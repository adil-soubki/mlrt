#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate flexfringe job script to build models and send to slurm.

Example Usage:
    $ ffgen.py --ini alergia \
               --modeldir ../FlexFringe/models-short \
               --datadir ../../tmp/sgen/PlusShort/ \
               --data-size Mid \
               --data-type TrainPS \
               --partitions short-28core
"""
import os
import sys
import time
from glob import glob

import pandas as pd
from more_itertools import distribute

from src.bin.flexfringe import FF_DIR
from src.core.context import Context
from src.core.app import harness
from src.core import slurm

# TODO: https://docs-research-it.berkeley.edu/services/high-performance-computing
#                                            /user-guide/running-your-jobs/gnu-parallel/


JOBS_DIR = "/gpfs/projects/HeinzGroup/Jobs"
TMP_DIR = "/gpfs/projects/HeinzGroup/tmp"
SF_DIR = "/gpfs/projects/HeinzGroup/asoubki/SiccoFringe"


def get_models(ini: str, modeldir: str):
    gstr = os.path.join(modeldir, f"{ini}_*.final.json")
    for path in glob(gstr):
        yield path


def get_training_data(data_type: str, datadir: str):
    gstr = os.path.join(datadir, f"*/*_{data_type}.txt")
    for path in glob(gstr):
        yield path


def get_missing_models(ini: str, data_type: str, datadir: str, modeldir: str):
    """Returns the paths to training data for all models not yet generated."""

    def parse_models(mstr):
        bname = ".".join(os.path.basename(mstr).split(".")[:-2])
        return {
            "dstr": bname.replace(f"{ini}_", ""),
            "data_size": bname.split("_")[-1],
            "model_path": mstr,
        }

    models = pd.DataFrame(map(parse_models, get_models(ini, modeldir)))
    if models.empty:
        models = pd.DataFrame([], columns=["dstr", "data_size", "model_path"])

    def parse_training_data(tstr):
        bname = os.path.basename(tstr)
        msize = os.path.basename(os.path.dirname(tstr))
        return {
            "dstr": bname.replace(f"{data_type}.txt", msize),
            "data_size": msize,
            "data_path": tstr,
        }

    tdata = pd.DataFrame(
        map(parse_training_data, get_training_data(data_type, datadir))
    )
    tdata = tdata.merge(models, how="left", on=["dstr", "data_size"])
    return tdata[tdata.model_path.isna()].data_path.tolist()


def main(ctx: Context) -> None:
    inis = ("edsm", "rpni", "alergia")
    data_sizes = ("Small", "Mid", "Large")
    data_types = ("Train", "TrainPS")
    partitions = slurm.sinfo().PARTITION.unique()
    ctx.parser.add_argument("-i", "--ini", choices=inis, required=True)
    ctx.parser.add_argument("-m", "--modeldir", type=os.path.realpath, required=True)
    ctx.parser.add_argument("-d", "--datadir", type=os.path.realpath, required=True)
    ctx.parser.add_argument("-s", "--data-size", choices=data_sizes, required=True)
    ctx.parser.add_argument("-t", "--data-type", choices=data_types, required=True)
    ctx.parser.add_argument("-p", "--partitions", choices=partitions, nargs="+")
    ctx.parser.add_argument(
        "-y", "--dryrun", action="store_true", help="don't send to slurm"
    )
    ctx.parser.set_defaults(
        modules=["shared", "gnu-parallel/6.0", "anaconda/3", "gcc/12.1.0"]
    )
    args = ctx.parser.parse_args()

    # Use a different binary for rpni since it is not on the main branch.
    ff_dir = SF_DIR if args.ini == "rpni" else FF_DIR
    ff_bin = os.path.join(ff_dir, "flexfringe")
    ini_file = os.path.join(ff_dir, "ini", f"{args.ini}.ini")
    ctx.log.info("FlexFringe Binary: %s", ff_bin)
    # Generate FlexFringe commands.
    cmds = []
    for path in get_missing_models(
        args.ini, args.data_type, args.datadir, args.modeldir
    ):
        ini = os.path.splitext(os.path.basename(ini_file))[0]
        bname = os.path.splitext(os.path.basename(path))[0].split("_")[0]
        dsize = os.path.basename(os.path.dirname(path))
        if ini != args.ini or dsize != args.data_size:
            continue  # Skip incorrect ini/size.
        out_file = os.path.join(args.modeldir, f"{ini}_{bname}_{dsize}")
        cmds.append(f"{ff_bin} {path} --ini {ini_file} --outputfile {out_file}")
    cmd_grps = distribute(len(args.partitions) * 2, cmds)
    # Generate slurm job files.
    for idx, grp in enumerate(cmd_grps):
        grp = list(grp)
        partition = args.partitions[idx % len(args.partitions)]
        cores = (28 if "28" in partition else 24) // 2
        nodes = 8 if "medium" in partition else 1
        nodes = 24 if "large" in partition else nodes
        jid = f"{args.ini}-{args.data_size}-{args.data_type}-{idx + 1}"
        cmdpath = os.path.join(JOBS_DIR, f"{jid}.txt")
        ctx.log.info(f"writing: {cmdpath} ({len(grp)} commands)")
        with open(cmdpath, "w") as fd:
            fd.write("\n".join(grp))
        slurmpath = os.path.join(JOBS_DIR, f"{jid}.slurm")
        slurm.sbatch(
            #  f"cat {cmdpath} | parallel --tmpdir={TMP_DIR} -l1 srun -N1 -n1 sh -c '$@' --",
            f"cat {cmdpath} | parallel --tmpdir={TMP_DIR} -P {cores}",
            flags={
                "ntasks-per-node": cores,
                "nodes": nodes,
                "time": slurm.timelimit(partition),
                "partition": partition,
                "mail-type": "BEGIN,END",
                "mail-user": "adil.soubki@stonybrook.edu",
            },
            modules=args.modules,
            dryrun=args.dryrun,
        )


if __name__ == "__main__":
    harness(main)
