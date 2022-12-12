# -*- coding: utf-8 -*
import argparse
import logging
import os
import subprocess
import sys
from typing import Optional


_FLAGS: dict[str, str] = {}  # XXX: unused for default arguments.
FF_DIR = os.path.abspath("/gpfs/projects/HeinzGroup/asoubki/FlexFringe/")
FF_BIN = os.path.join(FF_DIR, "flexfringe")


def flexfringe(
    tracefile: str,
    flags: Optional[dict[str, str]] = None,
    dryrun: bool = False,
) -> "subprocess.CompletedProcess[bytes]":
    # Parse inputs.
    flags = flags or _FLAGS
    for key in _FLAGS:
        if key not in flags:
            flags[key] = _FLAGS[key]
    # Prepare command
    cmd = [FF_BIN, tracefile]
    for key, val in flags.items():
        cmd.append(f"--{key}={val}")
    log = logging.getLogger(__name__)
    # Handle dryruns.
    if dryrun:
        log.info("=" * 32 + " DRYRUN " + "=" * 32)
        log.info("Would execute command.\n%s", "\n\t".join(cmd))
        log.info("=" * 32 + " DRYRUN " + "=" * 32)
        return subprocess.CompletedProcess("", 0)
    log.info("Executing command.\n%s", "\n\t".join(cmd))
    # Submit the job.
    return subprocess.run(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=True,
    )


def flexfringify(parser: argparse.ArgumentParser) -> argparse.Namespace:
    # Add flexfringe arguments.
    tkns = (
        subprocess.run([FF_BIN, "--help"], check=True, capture_output=True)
        .stdout.decode("utf-8")
        .split()
    )
    flags = set(
        [tkn.split("=")[0][2:].replace("[", "") for tkn in tkns if tkn.startswith("--")]
    )
    grp = parser.add_argument_group("flexfringe arguments")
    for flag in sorted(list(flags)):
        grp.add_argument("--ff-" + flag, help=argparse.SUPPRESS)
    parser.epilog = "\n".join(
        [
            parser.epilog or "",
            "flexfringe arguments can be modified using --ff-<argument_name>",
        ]
    )
    return parser.parse_args()
