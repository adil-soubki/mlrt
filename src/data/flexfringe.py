# -*- coding: utf-8 -*
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterator


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
            fcnts = node["data"]["final_counts"]
            assert set(fcnts.keys()) == {"0", "1"}
            assert not (fcnts[bad] > 0 and fcnts[good] > 0)
            dfa[str(node["id"])]["is_final"] = fcnts[good] > 0
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
