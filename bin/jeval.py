#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate eval job script and send to slurm."""
import os
import sys
import time
from glob import glob

import pandas as pd

from src.core.context import Context
from src.core.app import harness
from src.core.slurm import sbatch
from src.bin.flexfringe import FF_DIR


EVAL_BIN = os.path.abspath(os.path.join(os.path.dirname(__file__), "eval.py"))
TMP_DIR = "/gpfs/projects/HeinzGroup/tmp"


# XXX: Only does edsm right now.
def get_missing_models(evalpath: str) -> set[str]:
    gstr = os.path.join(FF_DIR, "models/edsm*.final.json")
    all_models = {p for p in glob(gstr)}
    try:
        evals = pd.read_csv(evalpath)
    except FileNotFoundError:
        return all_models
    else:
        return all_models - set(evals.model_path)


# TODO: Maybe this should just merge with eval.py?
def main(ctx: Context) -> None:
    ctx.parser.add_argument("-o", "--outpath", default=os.path.join(FF_DIR, "evals.csv"))
    ctx.parser.add_argument("-y", "--dryrun", action="store_true", help="don't send to slurm")
    ctx.parser.set_defaults(modules=["shared"])
    args = ctx.parser.parse_args()
    # Generate file for gnu-parallel.
    scriptname = os.path.basename(sys.argv[0]).replace(".py", "")
    cmdpath = time.strftime(os.path.join(TMP_DIR, f"{scriptname}.%Y%m%d.%H%M%S.txt"))
    cmds = [] 
    for path in get_missing_models(args.outpath):
        cmds.append(f"{EVAL_BIN} {path} -o {args.outpath}")
    ctx.log.info("writing: %s", cmdpath)
    with open(cmdpath, "w") as fd:
        fd.write("\n".join(cmds))
    # Submit job to slurm. 
    sbatch(
        f"cat {cmdpath} | parallel --tmpdir={TMP_DIR} -l1 srun -N1 -n1 sh -c '$@' --",
        flags={
            "ntasks-per-node": 28,
            "nodes": 2,
            "time": "7-00:00:00",
            "partition": "extended-28core",
        },
        dryrun=args.dryrun
    )


if __name__ == "__main__":
    harness(main)
