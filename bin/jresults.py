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
from src.bin.flexfringe import FF_DIR, FF_BIN


TMP_DIR = "/gpfs/projects/HeinzGroup/tmp"
MLRT_DIR = os.path.join(FF_DIR, "data", "MLRegTest")
INI_PATH = os.path.join(FF_DIR, "ini", "likelihood.ini")


# XXX: Only does edsm right now.
def get_missing_models(evaldir: str, ini: str) -> set[str]:
    gstr = os.path.join(FF_DIR, f"models/{ini}*.final.json")
    all_models = {p for p in glob(gstr)}
    #  evals = pd.concat([
    #      pd.read_csv(p) for p in glob(os.path.join(evaldir, "*"))
    #  ] or [pd.DataFrame()])
    #  if not evals.empty:
    #      return all_models - set(evals.model_path)
    return all_models


#  (mlrt) [asoubki@login2 FlexFringe]$ #../FlexFringe/flexfringe ../FlexFringe/data/MLRegTest/Large/16.04.TLT.4.1.6_TestSR.txt --ini ini/likelihood.ini --aptafile="/gpfs/projects/HeinzGroup/asoubki/FlexFringe/models/alergia_16.04.TLT.4.1.6_Small.final.json" --mode predict --predicttype 1
#  (mlrt) [asoubki@login2 FlexFringe]$ #column -t -s';' < /gpfs/projects/HeinzGroup/asoubki/FlexFringe/models/alergia_16.04.TLT.4.1.6_Small.final.json.result | less -F -S -X -K


# TODO: Maybe this should just merge with eval.py?
def main(ctx: Context) -> None:
    inis = ["edsm", "rpni", "alergia"]
    ctx.parser.add_argument("-i", "--ini", default=inis[0])
    ctx.parser.add_argument("-o", "--outdir", default=os.path.join(FF_DIR, "results"))
    ctx.parser.add_argument("-y", "--dryrun", action="store_true", help="don't send to slurm")
    ctx.parser.set_defaults(modules=["shared"])
    args = ctx.parser.parse_args()
    # Generate file for gnu-parallel.
    scriptname = os.path.basename(sys.argv[0]).replace(".py", "")
    cmdpath = time.strftime(os.path.join(TMP_DIR, f"{scriptname}.%Y%m%d.%H%M%S.txt"))
    cmds = [] 
    for path in get_missing_models(args.outdir, args.ini):
        assert path.endswith(".final.json")
        ini, dstr, msize = os.path.basename(path).replace(".final.json", "").split("_")
        dpaths =  glob(os.path.join(MLRT_DIR, "Large", f"{dstr}_Test*"))
        dpaths += glob(os.path.join(MLRT_DIR, msize, f"{dstr}_Train*"))
        for dpath in dpaths:
#  (mlrt) [asoubki@login2 FlexFringe]$ #../FlexFringe/flexfringe ../FlexFringe/data/MLRegTest/Large/16.04.TLT.4.1.6_TestSR.txt --ini ini/likelihood.ini --aptafile="/gpfs/projects/HeinzGroup/asoubki/FlexFringe/models/alergia_16.04.TLT.4.1.6_Small.final.json" --mode predict --predicttype 1
#  (mlrt) [asoubki@login2 FlexFringe]$ #column -t -s';' < /gpfs/projects/HeinzGroup/asoubki/FlexFringe/models/alergia_16.04.TLT.4.1.6_Small.final.json.result | less -F -S -X -K
            rpath = os.path.join(args.outdir, f"{ini}_{dstr}_{msize}_{}")
            cmds.append(
                f"{FF_BIN} {dpath} --aptafile {path} --ini {INI_PATH} --mode predict --predicttype 1"
            )
        #  outpath = os.path.join(args.outdir, f"{dstr}.csv")
        #  cmds.append(f"{EVAL_BIN} {path} -o {outpath}")
    ctx.log.info("writing: %s", cmdpath)
    with open(cmdpath, "w") as fd:
        fd.write("\n".join(cmds))
    # Submit job to slurm. 
    os.makedirs(args.outdir, exist_ok=True)
    sbatch(
        f"cat {cmdpath} | parallel --tmpdir={TMP_DIR} -l1 srun -N1 -n1 sh -c '$@' --",
        flags={
            "ntasks-per-node": 24,
            "nodes": 1,
            "time": "7-00:00:00",
            "partition": "extended-24core",
        },
        dryrun=args.dryrun
    )


if __name__ == "__main__":
    harness(main)
