import argparse
import pathlib

import numpy as np
import pandas as pd
from heavyedge import ProfileData
from sklearn.model_selection import StratifiedKFold

parser = argparse.ArgumentParser(
    description="Write a CSV file containing train/test split flags.",
)
parser.add_argument(
    "profiles",
    type=pathlib.Path,
    help="HDF5 file containing preprocessed profile data.",
)
parser.add_argument(
    "labels",
    type=pathlib.Path,
    help="CSV file containing the true class labels in its 'Type' column.",
)
parser.add_argument(
    "--n-splits",
    type=int,
    required=True,
    help="Number of stratified folds.",
)
parser.add_argument(
    "-o",
    "--out",
    type=pathlib.Path,
    required=True,
    help="Split flags. 0 means train data and 1 means test data.",
)
args = parser.parse_args()

with ProfileData(args.profiles) as file:
    x = file.x()
    X, _, _ = file[:]
    X /= np.trapezoid(X, x, axis=1)[..., np.newaxis]
y = pd.read_csv(args.labels)["Type"].to_numpy()
flags = np.zeros((len(y), args.n_splits), dtype=int)

outer_fold = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=0)
for i, (_, test_idx) in enumerate(outer_fold.split(X, y)):
    flags[test_idx, i] = 1

df = pd.DataFrame(flags, columns=[f"split_{i}" for i in range(args.n_splits)])
df.to_csv(args.out, index=False)
