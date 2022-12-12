#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A wrapper around flexfringe."""
import os

from src.core.context import Context
from src.core.app import harness
from src.bin.flexfringe import flexfringe, flexfringify, FF_DIR


# TODO: Maybe some of this should go into src.bin.flexfringe?
def main(ctx: Context) -> None:
    ctx.parser.add_argument("tracefiles", nargs="+")
    ctx.parser.add_argument("-y", "--dryrun", action="store_true", help="don't execute")
    args = flexfringify(ctx.parser)

    ffflags = {k: v for k, v in vars(args).items() if k.startswith("ff_") and v is not None}
    mdir = os.path.join(FF_DIR, "models")
    os.makedirs(mdir, exist_ok=True)
    ini = os.path.splitext(os.path.basename(args.ff_ini))[0]
    for path in args.tracefiles:
        bname = os.path.splitext(os.path.basename(path))[0].split("_")[0]
        dsize = os.path.basename(os.path.dirname(path))
        ffflags["ff_outputfile"] = os.path.join(mdir, f"{ini}_{bname}_{dsize}")
        flexfringe(
            tracefile=path,
            flags={k.replace("ff_", ""): v for k, v in ffflags.items()},
            dryrun=args.dryrun
        )


if __name__ == "__main__":
    harness(main)
