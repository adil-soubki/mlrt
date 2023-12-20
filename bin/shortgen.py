#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generates short training data.

Output Stucture:
    |- outdir
        |- OnlyShort
        |   |- 04.02.TcoSL.2.1.0_TrainOS.ff [FlexFringe format]
        |   |- 04.02.TcoSL.2.1.0_TrainOS.mlrt [MLRegTest format]
        |   |- ...
        |- PlusShort
            |- Small
            |   |- 04.02.TcoSL.2.1.0_TrainPS.txt [FlexFringe format]
            |   |- ...
            |- Mid
            |   |- 04.02.TcoSL.2.1.0_TrainPS.txt [FlexFringe format]
            |   |- ...
            |- Large
                |- 04.02.TcoSL.2.1.0_TrainPS.txt [FlexFringe format]
                |- ...

Example Usage:
    $ shortgen.py -o /path/to/outdir
    $ shortgen.py -o /path/to/outdir --force

Note:
    Requires pynini 2.1.2.
    >>> conda install -c conda-forge pynini=2.1.2
"""
import os
import sys
from glob import glob

import pynini  # pylint: disable=import-error
import pandas as pd

from src.core.context import Context
from src.core.app import harness
from src.data.flexfringe import FFData
from src.data.mlrt import MLRT_DIR, MLRegTestFile


SRL_DIR = "/gpfs/projects/HeinzGroup/subregular-learning/"
FST_DIR = os.path.join(SRL_DIR, "src", "fstlib", "fst_format")
ON_LOAD = lambda fn: fn()


@ON_LOAD
def import_pathogen_as_pgn():
    sys.path.insert(1, os.path.join(SRL_DIR, "src"))
    globals()["pgn"] = __import__("pathogen")


def get_short_data_from_ltag(ltag: str):
    # pylint: disable=undefined-variable
    ret = []
    # Build FSAs
    fsa = pynini.Fst.read(os.path.join(FST_DIR, f"{ltag}.fst"))
    fsa.optimize()
    cofsa = pynini.difference(pgn.sigma(fsa).star, fsa)
    cofsa.optimize()
    # Create training data with duplicates.
    num_samples = 25
    min_len, max_len = 1, 19
    for ldx in range(min_len, max_len + 1):
        _, pos = pgn.create_data_with_duplicate(fsa, ldx, num_samples)
        _, neg = pgn.create_data_with_duplicate(cofsa, ldx, num_samples)
        ret.extend(zip(pos, [True] * len(pos)))
        ret.extend(zip(neg, [False] * len(neg)))
    return ret


def get_ltags() -> set[str]:
    ret = set()
    for path in glob(os.path.join(MLRT_DIR, "*", "*Train.txt")):
        ret.add(os.path.basename(path).split("_")[0])
    return ret


def main(ctx: Context) -> None:
    ctx.parser.add_argument("-o", "--outdir", required=True)
    ctx.parser.add_argument(
        "-f", "--force", action="store_true", help="overwrite existing"
    )
    args = ctx.parser.parse_args()

    # Generate OnlyShort data.
    outdir_os = os.path.join(args.outdir, "OnlyShort")
    os.makedirs(outdir_os, exist_ok=True)
    for ltag in get_ltags():
        if "co" in ltag:
            continue  # Skip compliment classes on first pass.
        outpath_mlrt = os.path.join(outdir_os, f"{ltag}_TrainOS.mlrt")
        outpath_ff = os.path.join(outdir_os, f"{ltag}_TrainOS.ff")
        if not os.path.exists(outpath_mlrt) or args.force:
            data = get_short_data_from_ltag(ltag)
            pd.DataFrame(data).to_csv(outpath_mlrt, index=False, header=False, sep="\t")
            ctx.log.info("wrote: %s", outpath_mlrt)
        if not os.path.exists(outpath_ff) or args.force:
            ff_string = MLRegTestFile.from_path(outpath_mlrt).to_string()
            with open(outpath_ff, "w") as fd:
                fd.write(ff_string)
            ctx.log.info("wrote: %s", outpath_ff)
    # Generate OnlyShort compliment classes on second pass.
    for ltag in get_ltags():
        if "co" not in ltag:
            continue  # Skip non-compliment classes.
        outpath_mlrt = os.path.join(outdir_os, f"{ltag}_TrainOS.mlrt")
        outpath_ff = os.path.join(outdir_os, f"{ltag}_TrainOS.ff")
        if not os.path.exists(outpath_mlrt) or args.force:
            df = (
                MLRegTestFile.from_path(
                    inpath := outpath_mlrt.replace(ltag, ltag.replace("co", ""))
                )
                .to_df()
                .eval("label = not label")
            )
            ctx.log.info("read: %s", inpath)
            df.to_csv(outpath_mlrt, index=False, header=False, sep="\t")
            ctx.log.info("wrote: %s", outpath_mlrt)
        if not os.path.exists(outpath_ff) or args.force:
            ff_string = MLRegTestFile.from_path(outpath_mlrt).to_string()
            ctx.log.info("read: %s", outpath_mlrt)
            with open(outpath_ff, "w") as fd:
                fd.write(ff_string)
            ctx.log.info("wrote: %s", outpath_ff)
    # Generate PlusShort data (FlexFringe format only).
    for data_size in ("Small", "Mid", "Large"):
        outdir_ps = os.path.join(args.outdir, "PlusShort", data_size)
        os.makedirs(outdir_ps, exist_ok=True)
        for path in glob(os.path.join(MLRT_DIR, data_size, "*Train.txt")):
            ltag = os.path.basename(path).split("_")[0]
            split = os.path.basename(path).split("_")[1].split(".")[0]
            assert split == "Train"
            outpath_ps = os.path.join(outdir_ps, f"{ltag}_TrainPS.txt")
            if os.path.exists(outpath_ps) and not args.force:
                continue  # Skip existing files.
            og_data = list(
                pd.read_csv(path, sep="\t", names=["sample", "label"]).itertuples(
                    index=False, name=None
                )
            )
            ctx.log.info("read: %s", path)
            sh_path = os.path.join(outdir_os, f"{ltag}_TrainOS.mlrt")
            sh_data = list(
                pd.read_csv(sh_path, sep="\t", names=["sample", "label"]).itertuples(
                    index=False, name=None
                )
            )
            ctx.log.info("read: %s", sh_path)
            data = sh_data + og_data
            pd.DataFrame(data).to_csv(outpath_ps, index=False, header=False, sep="\t")
            ff_string = MLRegTestFile.from_path(outpath_ps).to_string()
            with open(outpath_ps, "w") as fd:
                fd.write(ff_string)
            ctx.log.info("wrote: %s", outpath_ps)


if __name__ == "__main__":
    harness(main)
