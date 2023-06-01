#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generates short training examples."""
import os
from glob import glob

import pynini
import pandas as pd

from src.core.context import Context
from src.core.app import harness
from src.data.flexfringe import FFData
from src.data.mlrt import MLRT_DIR, MLRegTestFile


SRL_DIR = "/gpfs/projects/HeinzGroup/subregular-learning/"
FST_DIR = os.path.join(SRL_DIR, "src", "fstlib", "fst_format")


_A_FSA = pynini.acceptor("a", token_type="utf8")
ZERO = _A_FSA - _A_FSA
ZERO.optimize()


def alph(fsa):
    symtable = fsa.input_symbols()
    i = iter(symtable)
    next(i)  # skip over epsilon, ie first entry in symbol table
    temp = ''
    for sympair in i:  # table entries are pairs of form (num,symbol)
        temp = temp + sympair[1]
    return temp


def sigma(fsa):
    ret = ZERO
    for ch in alph(fsa):
        ret = pynini.acceptor(ch, token_type="utf8") | ret
    return ret.optimize()


def sigmastar(fsa):
    return (sigma(fsa).star).optimize()


def list_string_set(fsa):
    """
    Utility function that outputs all strings of an fsa

    Note:
        The fsa must recognize a finite language.
    """
    my_list = []
    paths = fsa.paths(input_token_type="utf8", output_token_type="utf8")
    for s in paths.ostrings():
        my_list.append(s)
    my_list.sort(key=len)
    return my_list


def make_string_dict(fsa, min_len, max_len, sigma):
    """
    Builds a dict of FSAs for strings of the given length range.

    Args:
        fsa (Fst): An FSA.
        min_len (int): The minimum string length.
        max_len (int): The maximum string length.

    Returns:
        dict[int, Fst]: Maps string length to acceptor.
    """
    fsa_dict = {}
    for i in range(min_len, max_len + 1):
        fsa_dict[i] = pynini.intersect(fsa, pynini.closure(sigma, i, i))
    return fsa_dict


# create {num} random strings of positive/negative examples.
# This may be duplicates.
def create_data_with_duplicate(pos_dict, neg_dict, min_len, max_len, num):
    ret = []
    for i in range(min_len, max_len + 1):
        # get num strings of length i from the pos_dict
        pos_fsa = pynini.randgen(
            pos_dict[i],
            npath=num,
            seed=0,
            select="uniform",
            max_length=(2 ** 31) - 1,
            weighted=False
        )
        # write them into the files
        for ele in list_string_set(pos_fsa):
            ret.append((ele, True))
        # update the pos_dict by subtracting the strings in pos_fsa
        pos_dict[i] = pynini.difference(pos_dict[i], pos_fsa)

        # get num strings of length i from the neg_dict
        neg_fsa = pynini.randgen(
            neg_dict[i],
            npath=num,
            seed=0,
            select="uniform",
            max_length=2147483647,
            weighted=False
        )
        # write them into the files
        for ele in list_string_set(neg_fsa):
            ret.append((ele, False))
        # update the neg_dict by subtracting the strings in neg_fsa
        neg_dict[i] = pynini.difference(neg_dict[i], neg_fsa)
    return ret


def get_short_data_from_ltag(ltag: str):
    # Build FSAs
    the_fsa = pynini.Fst.read(os.path.join(FST_DIR, f"{ltag}.fst"))
    the_s = sigma(the_fsa)
    the_ss = sigmastar(the_fsa)
    the_cofsa = pynini.difference(sigmastar(the_fsa), the_fsa)
    the_cofsa.optimize()
    # Set up dictionary for string lengths.
    num_samples = 25
    min_len, max_len = 1, 19 
    pos_dict = make_string_dict(the_fsa, min_len - 1, max_len + 1, the_s)
    neg_dict = make_string_dict(the_cofsa, min_len - 1, max_len + 1, the_s)
    # Create training data with duplicates.
    return create_data_with_duplicate(
        pos_dict,
        neg_dict,
        min_len,
        max_len,
        num_samples
    )


def generate_compliment(path: str) -> str:
    ff_data = FFData.from_path(path)
    # Flip samples
    header, samples = ff_data.header, []
    for label, length, seq in ff_data.samples:
        assert label in (0, 1)
        assert length == len(seq)
        label = 0 if label == 1 else 1
        samples.append((label, length, seq))
    assert header == ff_data.header
    assert len(samples) == len(ff_data.samples)
    # Convert to string.
    lines = []
    lines.append(" ".join(map(str, header)))
    for smpl in samples:
        lines.append(" ".join(map(str, [smpl[0], smpl[1]] + smpl[2])))
    return "\n".join(lines)


def main(ctx: Context) -> None:
    ctx.parser.add_argument("-o", "--outdir", required=True)
    ctx.parser.add_argument("-f", "--force", action="store_true", help="overwrite existing")
    args = ctx.parser.parse_args()

    for data_size in ("Small", "Mid", "Large"):
        outdir = os.path.join(args.outdir, data_size)
        os.makedirs(outdir, exist_ok=True)
        for path in glob(os.path.join(MLRT_DIR, data_size, "*Train.txt")):
            ltag = os.path.basename(path).split("_")[0]
            split = os.path.basename(path).split("_")[1].split(".")[0]
            assert split == "Train"
            if "co" in ltag:
                continue  # No fst files for these. Generate on second pass.
            outpath = os.path.join(outdir, f"{ltag}_TrainPS.txt")
            if os.path.exists(outpath) and not args.force:
                continue  # Skip existing files.
            og_data = list(pd.read_csv(
                path, sep="\t", names=["sample", "label"]
            ).itertuples(index=False, name=None))
            data = get_short_data_from_ltag(ltag) + og_data
            pd.DataFrame(data).to_csv(outpath, index=False, header=False, sep="\t")
            ff_string = MLRegTestFile.from_path(outpath).to_string()
            with open(outpath, "w") as fd:
                fd.write(ff_string)
            ctx.log.info("wrote: %s", outpath)
        # Generate compliment classes on second pass.
        for path in glob(os.path.join(MLRT_DIR, data_size, "*Train.txt")):
            ltag = os.path.basename(path).split("_")[0]
            split = os.path.basename(path).split("_")[1].split(".")[0]
            assert split == "Train"
            if "co" not in ltag:
                continue  # Only generate for compliment classes.
            outpath = os.path.join(outdir, f"{ltag}_TrainPS.txt")
            if os.path.exists(outpath) and not args.force:
                continue  # Skip existing files.
            ff_string = generate_compliment(
                inpath := outpath.replace(ltag, ltag.replace("co", ""))
            )
            ctx.log.info("read: %s", inpath)
            with open(outpath, "w") as fd:
                fd.write(ff_string)
            ctx.log.info("wrote: %s", outpath)


if __name__ == "__main__":
    harness(main)
