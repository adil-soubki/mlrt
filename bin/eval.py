#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate flexfringe models on all corresponding data."""
import os
from contextlib import suppress
from dataclasses import asdict
from glob import glob
from typing import Any

import evaluate
import pandas as pd
from tqdm import tqdm

from src.core.context import Context
from src.core.app import harness
from src.core.df import update
from src.bin.flexfringe import FF_DIR
from src.data.flexfringe import FFModel


MLRT_DIR = os.path.join(FF_DIR, "data", "MLRegTest")
CACHE_DIR = os.path.join(os.path.dirname(FF_DIR), ".cache")


def compute(model: FFModel, path: str) -> dict[str, Any]:
    ret = {k: v for k, v in vars(model).items() if k not in ("path", "machine", "dfa")}
    ret["split"] = os.path.splitext(os.path.basename(path))[0].split("_")[-1]
    df = pd.DataFrame(map(asdict, model.evaluate(path)))
    #  df = pd.DataFrame(map(asdict, model.results(path)))  # Uses sicco method.
    for metric in ("accuracy", "precision", "recall", "f1", "brier_score"): #  "roc_auc"?
        ret |= evaluate.load(
            metric,
            experiment_id=str(pd.Timestamp.now()),
            cache_dir=CACHE_DIR,  # My home runs out of Disk...
        ).compute(predictions=df.pred, references=df.label)
    ret["model_path"] = os.path.abspath(model.path)
    ret["data_path"] = os.path.abspath(path)
    return ret


def main(ctx: Context) -> None:
    ctx.parser.add_argument("paths", nargs="+", help="model paths")
    ctx.parser.add_argument("-o", "--outpath", default=os.path.join(FF_DIR, "evals.csv"))
    args = ctx.parser.parse_args()
    ctx.log.info("model paths: %s", args.paths)
    ctx.log.info("outpath: %s", args.outpath)
    # Read existing file if it exists.
    current = pd.DataFrame()
    with suppress(FileNotFoundError):
        current = pd.read_csv(args.outpath)
    # Evaluate the given models.
    results = []
    for path in tqdm(args.paths):
        assert path.endswith(".final.json")
        _, dstr, msize = os.path.basename(path).replace(".final.json", "").split("_")
        # Always evaluate on large but include same msize train as a sanity check.
        dpaths =  glob(os.path.join(MLRT_DIR, "Large", f"{dstr}_Test*"))
        dpaths += glob(os.path.join(MLRT_DIR, msize, f"{dstr}_Train*"))
        model = FFModel.from_path(path)
        for dpath in tqdm(dpaths, leave=False):
            results.append(compute(model, dpath))
    # Update and write results.
    new = pd.DataFrame(results)
    new["last_modified"] = pd.Timestamp.now()
    if current.empty:
        current = pd.DataFrame(columns=new.columns)
    ctx.log.info("writing: %s", args.outpath)
    update(current, new, on=["model_path", "data_path"]).to_csv(args.outpath, index=False)


if __name__ == "__main__":
    harness(main)
