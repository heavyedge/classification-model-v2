import argparse
import pathlib

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("knees", type=pathlib.Path, help="Knee labels.")
parser.add_argument("canonical", type=pathlib.Path, help="Canonical labels.")
parser.add_argument("-o", "--out", type=pathlib.Path, help="Output csv file")
args = parser.parse_args()


def combine_labels(knees, canonical):
    canonical = canonical.astype(bool)
    out = np.empty_like(knees, dtype=object)
    out[knees == 0] = "Type 0"
    out[(knees == 1) & canonical] = "Type 1a"
    out[(knees == 1) & (~canonical)] = "Type 1b"
    out[(knees == 2) & canonical] = "Type 2a"
    out[(knees == 2) & (~canonical)] = "Type 2b"
    out[(knees == 3) & canonical] = "Type 3a"
    out[(knees == 3) & (~canonical)] = "Type 3b"
    return out


knees = pd.read_csv(args.knees)["Type"].to_numpy()
canonical = pd.read_csv(args.canonical)["Type"].to_numpy()
labels = combine_labels(knees, canonical)

pd.DataFrame({"Type": labels}).to_csv(args.out, index=False)
