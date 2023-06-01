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
    $ sgen.py -o /path/to/outdir
    $ sgen.py -o /path/to/outdir --force

Note:
    Requires pynini 2.1.2.
    >>> conda install -c conda-forge pynini=2.1.2
"""
import os
from glob import glob

import pynini
import pandas as pd

from src.core.context import Context
from src.core.app import harness
from src.data.flexfringe import FFData
from src.data.mlrt import MLRT_DIR, MLRegTestFile


SRL_DIR = "/gpfs/projects/HeinzGroup/subregular-learning/"
FST_DIR = os.path.join(SRL_DIR, "src", "fstlib", "fst_format")


_A_FSA = pynini.acceptor("a", token_type="utf8")
ZERO = _A_FSA - _A_FSA
ZERO.optimize()


def alph(fsa):
    symtable = fsa.input_symbols()
    i = iter(symtable)
    next(i)  # skip over epsilon, ie first entry in symbol table
    temp = ''
    for sympair in i:  # table entries are pairs of form (num,symbol)
        temp = temp + sympair[1]
    return temp


def sigma(fsa):
    ret = ZERO
    for ch in alph(fsa):
        ret = pynini.acceptor(ch, token_type="utf8") | ret
    return ret.optimize()


def sigmastar(fsa):
    return (sigma(fsa).star).optimize()


def list_string_set(fsa):
    """
    Utility function that outputs all strings of an fsa

    Note:
        The fsa must recognize a finite language.
    """
    my_list = []
    paths = fsa.paths(input_token_type="utf8", output_token_type="utf8")
    for s in paths.ostrings():
        my_list.append(s)
    my_list.sort(key=len)
    return my_list


def make_string_dict(fsa, min_len, max_len, sigma):
    """
    Builds a dict of FSAs for strings of the given length range.

    Args:
        fsa (Fst): An FSA.
        min_len (int): The minimum string length.
        max_len (int): The maximum string length.

    Returns:
        dict[int, Fst]: Maps string length to acceptor.
    """
    fsa_dict = {}
    for i in range(min_len, max_len + 1):
        fsa_dict[i] = pynini.intersect(fsa, pynini.closure(sigma, i, i))
    return fsa_dict


# create {num} random strings of positive/negative examples.
# This may be duplicates.
def create_data_with_duplicate(pos_dict, neg_dict, min_len, max_len, num):
    ret = []
    for i in range(min_len, max_len + 1):
        # get num strings of length i from the pos_dict
        pos_fsa = pynini.randgen(
            pos_dict[i],
            npath=num,
            seed=0,
            select="uniform",
            max_length=(2 ** 31) - 1,
            weighted=False
        )
        # write them into the files
        for ele in list_string_set(pos_fsa):
            ret.append((ele, True))
        # update the pos_dict by subtracting the strings in pos_fsa
        pos_dict[i] = pynini.difference(pos_dict[i], pos_fsa)

        # get num strings of length i from the neg_dict
        neg_fsa = pynini.randgen(
            neg_dict[i],
            npath=num,
            seed=0,
            select="uniform",
            max_length=2147483647,
            weighted=False
        )
        # write them into the files
        for ele in list_string_set(neg_fsa):
            ret.append((ele, False))
        # update the neg_dict by subtracting the strings in neg_fsa
        neg_dict[i] = pynini.difference(neg_dict[i], neg_fsa)
    return ret


def get_short_data_from_ltag(ltag: str):
    # Build FSAs
    the_fsa = pynini.Fst.read(os.path.join(FST_DIR, f"{ltag}.fst"))
    the_s = sigma(the_fsa)
    the_ss = sigmastar(the_fsa)
    the_cofsa = pynini.difference(sigmastar(the_fsa), the_fsa)
    the_cofsa.optimize()
    # Set up dictionary for string lengths.
    num_samples = 25
    min_len, max_len = 1, 19 
    pos_dict = make_string_dict(the_fsa, min_len - 1, max_len + 1, the_s)
    neg_dict = make_string_dict(the_cofsa, min_len - 1, max_len + 1, the_s)
    # Create training data with duplicates.
    return create_data_with_duplicate(
        pos_dict,
        neg_dict,
        min_len,
        max_len,
        num_samples
    )


def get_ltags() -> set[str]:
    ret = set()
    for path in glob(os.path.join(MLRT_DIR, "*", "*Train.txt")):
        ret.add(os.path.basename(path).split("_")[0])
    return ret


def main(ctx: Context) -> None:
    ctx.parser.add_argument("-o", "--outdir", required=True)
    ctx.parser.add_argument("-f", "--force", action="store_true", help="overwrite existing")
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
            df = MLRegTestFile.from_path(
                inpath := outpath_mlrt.replace(ltag, ltag.replace("co", ""))
            ).to_df().eval("label = not label")
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
            og_data = list(pd.read_csv(
                path, sep="\t", names=["sample", "label"]
            ).itertuples(index=False, name=None))
            ctx.log.info("read: %s", path)
            sh_path = os.path.join(outdir_os, f"{ltag}_TrainOS.mlrt")
            sh_data = list(pd.read_csv(
                sh_path, sep="\t", names=["sample", "label"]
            ).itertuples(index=False, name=None))
            ctx.log.info("read: %s", sh_path)
            data = sh_data + og_data
            pd.DataFrame(data).to_csv(outpath_ps, index=False, header=False, sep="\t")
            ff_string = MLRegTestFile.from_path(outpath_ps).to_string()
            with open(outpath_ps, "w") as fd:
                fd.write(ff_string)
            ctx.log.info("wrote: %s", outpath_ps)


if __name__ == "__main__":
    harness(main)
