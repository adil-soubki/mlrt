#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""An example script"""
import os
from dataclasses import asdict
from glob import glob
from typing import Any

import evaluate
import pandas as pd
from tqdm import tqdm

from src.core.context import Context
from src.core.app import harness
from src.bin.flexfringe import FF_DIR
from src.data.flexfringe import FFModel


MLRT_DIR = os.path.join(FF_DIR, "data", "MLRegTest")


def compute(model: FFModel, path: str) -> dict[str, Any]:
    ret = {k: v for k, v in vars(model).items() if k not in ("path", "machine", "dfa")}
    ret["split"] = os.path.splitext(os.path.basename(path))[0].split("_")[-1]
    df = pd.DataFrame(map(asdict, model.evaluate(path)))
    for metric in ("precision", "recall", "f1", "brier_score"): #  "roc_auc"?
        ret |= evaluate.load(metric).compute(predictions=df.pred, references=df.label)
    return ret


def main(ctx: Context) -> None:
    ctx.parser.add_argument("paths", nargs="+", help="model paths")
    ctx.parser.add_argument("-o", "--outpath", default=os.path.join(FF_DIR, "evals.csv"))
    args = ctx.parser.parse_args()

    results = []
    for path in tqdm(args.paths):
        assert path.endswith(".final.json")
        _, dstr, msize = os.path.basename(path).replace(".final.json", "").split("_")
        model = FFModel.from_path(path)
        for dpath in tqdm(glob(os.path.join(MLRT_DIR, msize, f"{dstr}_T*")), leave=False):
            results.append(compute(model, dpath))
    ctx.log.info("writing: %s", args.outpath)
    pd.DataFrame(results).to_csv(args.outpath, index=False)


if __name__ == "__main__":
    harness(main)
