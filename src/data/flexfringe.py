# -*- coding: utf-8 -*
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterator

import pandas as pd  # type: ignore

from ..bin.flexfringe import FF_BIN, FF_DIR


PROBABALISTIC_INIS = ["alergia"]


@dataclass
class FFData:
    path: str
    header: list[int]
    samples: list[tuple[int, int, list[int]]]

    @classmethod
    def from_path(cls, path: str) -> FFData:
        samples = []
        with open(path, "r") as fd:
            header = list(map(int, next(fd).strip().split()))
            for ln in fd:
                line = list(map(int, ln.strip().split()))
                assert line[1] == len(line[2:])
                samples.append((line[0], line[1], line[2:]))
            assert header[0] == len(samples)
            return cls(path=path, header=header, samples=samples)


@dataclass
class FFModelResult:
    seq: list[int]
    label: bool
    pred: bool


@dataclass
class FFModel:
    path: str
    machine: dict[str, Any]
    dfa: dict[str, dict[str, Any]]
    ini: str  # E.g. esdm
    data_size: str  # Small, Mid, Large
    alphabet_size: int  # 4, 16, 64
    tier_size: int
    language_class: str
    factor_width: int
    threshold: int
    index: int

    @classmethod
    def from_path(cls, path: str) -> FFModel:
        ini = os.path.basename(path).split("_")[
            0
        ]  # XXX: Code repetition here... clean up.
        with open(path, "r") as fd:
            machine = json.load(fd)
        # {src1: {inp1: tgt1, ..., is_final: True/False}, ...}
        dfa: dict[str, dict[str, Any]] = defaultdict(dict)
        # dfa[start_state][input] = end_state
        for edge in machine["edges"]:
            dfa[edge["source"]][edge["name"]] = edge["target"]
        # dfa[state_id]["is_final"] = True/False
        assert set(machine["types"]) == {"0", "1"}
        bad, good = machine["types"]
        for node in machine["nodes"]:
            if ini in PROBABALISTIC_INIS:
                fcnts = {"0": 0, "1": 0} | node["data"].get("final_counts", {})
                dfa[str(node["id"])]["is_final"] = fcnts[good] > fcnts[bad]
            else:
                fcnts = {"0": 0, "1": 0} | node["data"]["final_counts"]
                assert not (fcnts[bad] > 0 and fcnts[good] > 0)
                dfa[str(node["id"])]["is_final"] = fcnts[good] > 0
            assert set(fcnts.keys()) == {"0", "1"}
        # assign metadata
        bname = os.path.basename(path)
        mdata = bname.split("_")[1].split(".")
        return cls(
            path=path,
            machine=machine,
            dfa=dfa,
            ini=bname.split("_")[0],
            data_size=bname.split("_")[-1].split(".")[0],
            alphabet_size=int(mdata[0]),
            tier_size=int(mdata[1]),
            language_class=mdata[2],
            factor_width=int(mdata[3]),
            threshold=int(mdata[4]),
            index=int(mdata[5]),
        )

    def __call__(self, seq: list[int]) -> bool:
        state = "0"
        for sym in seq:
            state = self.dfa[state].get(str(sym), None)
            if state is None:
                return False
        assert isinstance(self.dfa[state]["is_final"], bool)
        return bool(self.dfa[state]["is_final"])

    def evaluate(self, path: str) -> Iterator[FFModelResult]:
        for label, _, seq in FFData.from_path(path).samples:
            assert label in (0, 1)
            yield FFModelResult(seq, bool(label), self(seq))

    def results(self, path: str) -> Iterator[FFModelResult]:
        _, dstr, _ = os.path.basename(self.path).replace(".final.json", "").split("_")
        dsize = os.path.basename(os.path.dirname(path))
        split = os.path.basename(path).replace(".txt", "").split("_")[-1]
        assert dsize in {"Small", "Mid", "Large"}
        assert split in {"Train", "TestSR", "TestSA", "TestLR", "TestLA"}
        rdir = os.path.join(FF_DIR, "results")
        rpath = os.path.join(
            rdir, f"{self.ini}-{self.data_size}_{dstr}_{dsize}-{split}.result"
        )
        os.makedirs(rdir, exist_ok=True)
        inipath = os.path.join(FF_DIR, "ini", "likelihood.ini")
        cmd = [
            f"{FF_BIN} {path} --aptafile {self.path} "
            f"--ini {inipath} --mode predict --predicttype 1; "
            f"mv {self.path}.result {rpath}"
        ]
        subprocess.run(
            cmd, stdout=sys.stdout, stderr=sys.stderr, shell=True, check=True
        )
        for row in pd.read_csv(rpath, sep=";").to_dict("records"):
            yield FFModelResult(
                list(
                    map(
                        int, row[" abbadingo trace"].replace('"', "").strip().split(" ")
                    )
                ),
                label=row[" trace type"],
                pred=row[" predicted trace type"],
            )
