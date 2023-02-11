#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Batch eval.py jobs and send them to slurm."""
import os
import sys
from itertools import islice

from src.core.context import Context
from src.core.app import harness
from src.core.slurm import sbatch
from src.bin.flexfringe import FF_DIR


def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while (batch := tuple(islice(it, n))):
        yield batch


# TODO: Maybe this should just merge with eval.py?
def main(ctx: Context) -> None:
    ctx.parser.add_argument("paths", nargs="+", help="model paths")
    ctx.parser.add_argument("-o", "--outpath", default=os.path.join(FF_DIR, "evals.csv"))
    ctx.parser.add_argument("-n", "--batch-size", type=int, default=32)
    ctx.parser.add_argument("-y", "--dryrun", action="store_true", help="don't send to slurm")
    ctx.parser.set_defaults(modules=["shared"])
    args = ctx.parser.parse_args()

    for batch in batched(args.paths, args.batch_size):
        cmd = " ".join(list(batch) + [f"-o {args.outpath}"])
        sbatch(
            f"python -u {os.path.abspath(sys.argv[0]).replace('beval', 'eval')} {cmd}",
            dryrun=args.dryrun
        )


if __name__ == "__main__":
    harness(main)
