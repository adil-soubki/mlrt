#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate eval job script and send to slurm.

Example Usage:
    $ jeval.py --ini edsm -o ../FlexFringe/evals-short -m ../FlexFringe/models-short
"""
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


def get_missing_models(evaldir: str, modeldir: str, ini: str) -> set[str]:
    gstr = os.path.join(os.path.realpath(modeldir), f"{ini}*.final.json")
    all_models = {p for p in glob(gstr)}
    evals = pd.concat([
        pd.read_csv(p) for p in glob(os.path.join(evaldir, "*.csv"))
    ] or [pd.DataFrame()])
    if not evals.empty:
        return all_models - set(evals.model_path)
    return all_models


# TODO: Maybe this should just merge with eval.py?
def main(ctx: Context) -> None:
    inis = ["edsm", "rpni", "alergia"]
    ctx.parser.add_argument("-i", "--ini", default=inis[0])
    ctx.parser.add_argument("-o", "--outdir", default=os.path.join(FF_DIR, "evals"))
    ctx.parser.add_argument("-m", "--modeldir", default=os.path.join(FF_DIR, "models"))
    ctx.parser.add_argument("-y", "--dryrun", action="store_true", help="don't send to slurm")
    ctx.parser.set_defaults(modules=["shared"])
    args = ctx.parser.parse_args()
    # Generate file for gnu-parallel.
    scriptname = os.path.basename(sys.argv[0]).replace(".py", "")
    cmdpath = time.strftime(os.path.join(TMP_DIR, f"{scriptname}.%Y%m%d.%H%M%S.txt"))
    cmds = [] 
    for path in get_missing_models(args.outdir, args.modeldir, args.ini):
        dstr = ".".join(os.path.basename(path).split(".")[:-2])
        outpath = os.path.join(args.outdir, f"{dstr}.csv")
        cmds.append(f"{EVAL_BIN} {path} -o {outpath}")
    ctx.log.info("writing: %s", cmdpath)
    with open(cmdpath, "w") as fd:
        fd.write("\n".join(cmds))
    # Submit job to slurm. 
    os.makedirs(args.outdir, exist_ok=True)
    sbatch(
        f"cat {cmdpath} | parallel --tmpdir={TMP_DIR} -l1 srun -N1 -n1 sh -c '$@' --",
        flags={
            "ntasks-per-node": 28,
            "nodes": 1,
            "time": "2-00:00:00",
            "partition": "long-28core",
        },
        dryrun=args.dryrun
    )


if __name__ == "__main__":
    harness(main)
