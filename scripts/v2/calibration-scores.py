import argparse
import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

parser = argparse.ArgumentParser()
parser.add_argument("labels", type=pathlib.Path, help="Label csv file")
parser.add_argument(
    "split",
    type=pathlib.Path,
    help="CSV file containing the train/test split flags (0 for train, 1 for test).",
)
parser.add_argument("cv", type=pathlib.Path, help="CV csv files")
parser.add_argument(
    "-o", "--out", type=pathlib.Path, required=True, help="Output csv file"
)
args = parser.parse_args()

scores = {
    "F1": [],
    "Precision": [],
    "Recall": [],
    "ROCAUC": [],
    "NLL": [],
    "Brier": [],
}

split = pd.read_csv(args.split).to_numpy()
y = LabelEncoder().fit_transform(pd.read_csv(args.labels)["Type"])
cv = pd.read_csv(args.cv).to_numpy()
n_classes = cv.shape[1]

for flag in split.T:
    is_test = flag.astype(bool)
    y_test = y[is_test]
    y_prob = cv[is_test]
    y_pred = y_prob.argmax(axis=1)

    scores["F1"].append(f1_score(y_test, y_pred, average="weighted"))
    scores["Precision"].append(
        precision_score(y_test, y_pred, average="weighted", zero_division=0)
    )
    scores["Recall"].append(recall_score(y_test, y_pred, average="weighted"))
    scores["ROCAUC"].append(
        roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
    )
    scores["NLL"].append(log_loss(y_test, y_prob))
    scores["Brier"].append(
        np.mean([brier_score_loss(y_test == k, y_prob[:, k]) for k in range(n_classes)])
    )

pd.DataFrame(scores).to_csv(args.out, index=False)
