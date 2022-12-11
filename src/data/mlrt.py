# -*- coding: utf-8 -*
import functools
import glob
import os
from dataclasses import dataclass 
from typing import Any

import pandas as pd


FF_DIR = os.path.abspath("/gpfs/projects/HeinzGroup/asoubki/FlexFringe/")
FF_DATA_DIR = os.path.join(FF_DIR, "data")
MLRT_DIR = "/gpfs/projects/HeinzGroup/subregular-learning/data_gen/"


@functools.lru_cache
def validate_alphabet(alphabet_size: int) -> list[str]:
    alphabet = set()
    for path in glob.glob(os.path.join(MLRT_DIR, "Small", f"{alphabet_size}.*")):
        df = MLRegTestFile.from_path(path).to_df()
        alphabet |= set("".join(df["sample"]))
    assert alphabet_size == len(alphabet)
    return sorted(list(alphabet), key=lambda s: (not s.isascii(), not s.islower(), s))


@dataclass
class MLRegTestFile:
    path: str
    data_size: str      # Small, Mid, Large
    split: str          # Train, Dev, Test(SR/LR/SA/LA)
    alphabet_size: int  # 4, 16, 64
    tier_size: int
    language_class: str
    factor_width: int
    threshold: int
    index: int
    
    @classmethod
    def from_path(cls, path: str) -> MLRegTestFile:
        bname = os.path.basename(path)
        mdata = bname.split("_")[0].split(".")
        return cls(
            path=os.path.abspath(path),
            data_size=os.path.basename(os.path.dirname(path)),
            split=bname.split("_")[1].split(".")[0],
            alphabet_size=int(mdata[0]),
            tier_size=int(mdata[1]),
            language_class=mdata[2],
            factor_width=int(mdata[3]),
            threshold=int(mdata[4]),
            index=int(mdata[5])
        )
    
    def to_string(self) -> str:
        lines = []
        header, samples = self.to_flexfringe()
        lines.append(" ".join(map(str, header)))
        for smpl in samples:
            lines.append(" ".join(map(str, [smpl[0], smpl[1]] + smpl[2])))
        return "\n".join(lines)
    
    def to_flexfringe(self) -> tuple[list[Any], list[Any]]:
        # TODO: validate small, mid, large have 1k, 10k, and 100k samples.
        alphabet = validate_alphabet(64)
        df = self.to_df()
        header = [len(df), self.alphabet_size]
        samples = []
        for dct in df.to_dict("records"):
            smpl = list(map(lambda s: alphabet.index(s), dct["sample"]))
            assert len(smpl) == len(dct["sample"])
            samples.append((int(dct["label"]), len(smpl), smpl))
        return header, samples
            
    def to_df(self) -> pd.DataFrame:
        return pd.read_csv(self.path, sep="\t", names=["sample", "label"])
