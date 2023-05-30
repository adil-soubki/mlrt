#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert MLRegTest files to FlexFringe format."""
import os
import glob

from src.bin.flexfringe import FF_DIR
from src.core.context import Context
from src.core.app import harness
from src.data.mlrt import MLRT_DIR, MLRegTestFile


def main(ctx: Context) -> None:
    default_outdir = os.path.join(FF_DIR, "data", "MLRegTest")
    ctx.parser.add_argument("-o", "--outdir", default=default_outdir)
    args = ctx.parser.parse_args()

    for data_size in ("Small", "Mid", "Large"):
        outdir = os.path.join(args.outdir, data_size)
        os.makedirs(outdir, exist_ok=True)
        for path in glob.glob(os.path.join(MLRT_DIR, data_size, "*")):
            outpath = os.path.join(outdir, os.path.basename(path))
            if os.path.exists(outpath):
                continue  # Skip existing files.
            mlrt_file = MLRegTestFile.from_path(path)
            with open(outpath, "w") as fd:
                fd.write(mlrt_file.to_string())
            ctx.log.info(f"wrote: %s", outpath)


if __name__ == "__main__":
    harness(main)
