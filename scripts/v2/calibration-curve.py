import argparse
import pathlib

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

parser = argparse.ArgumentParser()
parser.add_argument("labels", type=pathlib.Path, help="Label csv file")
parser.add_argument("cv", type=pathlib.Path, help="CV csv files")
parser.add_argument(
    "-o", "--out", type=pathlib.Path, required=True, help="Output csv file"
)
parser.add_argument(
    "--n-bins", type=int, default=10, help="Number of bins for calibration curve"
)
args = parser.parse_args()

probabilities = pd.read_csv(args.cv)
y_prob_oof = probabilities.to_numpy()

y = pd.read_csv(args.labels)["Type"]
labels = np.sort(y.unique())
n_classes = len(labels)

results = []

for k, label in enumerate(labels):
    mask = ~np.isnan(y_prob_oof[:, k])
    frac_pos, mean_pred = calibration_curve(
        (y[mask] == label).astype(int),
        y_prob_oof[mask, k],
        n_bins=args.n_bins,
        strategy="uniform",
    )

    results.append(
        pd.DataFrame(
            {
                "class": label,
                "frac_pos": frac_pos,
                "mean_pred": mean_pred,
            }
        )
    )

pd.concat(results, ignore_index=True).to_csv(args.out, index=False)
