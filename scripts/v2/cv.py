import argparse
import logging
import pathlib

import numpy as np
import pandas as pd
from heavyedge import ProfileData
from heavyedge_classify.model import minirocket_classifier
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description=(
        "Evaluate a calibrated MiniRocket classifier with nested stratified "
        "cross-validation. The input profiles are area-normalized, a model is "
        "trained for each outer fold, and out-of-fold class probabilities are "
        "written to a CSV file."
    ),
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=(
        "The labels CSV must contain a 'Type' column. The output CSV has one "
        "column per class (named with the corresponding label) and one row per "
        "input profile; each value is that profile's out-of-fold probability.\n\n"
        "Example:\n"
        "  python cv.py profiles.h5 labels.csv --calibration sigmoid "
        "--n-splits 5 -o benchmarks/CV.sigmoid.csv"
    ),
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
    "split",
    type=pathlib.Path,
    help="CSV file containing the train/test split flags (0 for train, 1 for test).",
)
parser.add_argument(
    "--calibration",
    choices=[
        "sigmoid",
        "isotonic",
        "temperature",
        "sigmoid_ovo",
        "isotonic_ovo",
    ],
    required=True,
    help="Probability calibration method used by the classifier.",
)
parser.add_argument(
    "--n-splits",
    type=int,
    required=True,
    help="Number of stratified folds for inner cross-validation.",
)
parser.add_argument(
    "-o",
    "--out",
    type=pathlib.Path,
    required=True,
    help="Destination CSV file for the out-of-fold class probabilities.",
)
args = parser.parse_args()

with ProfileData(args.profiles) as file:
    x = file.x()
    X, _, _ = file[:]
    X /= np.trapezoid(X, x, axis=1)[..., np.newaxis]

y = pd.read_csv(args.labels)["Type"].to_numpy()
labels = np.sort(np.unique(y))
n_classes = len(labels)

split_flags = pd.read_csv(args.split).to_numpy()
outer_splits = [
    (np.where(split_flags[:, i] == 0)[0], np.where(split_flags[:, i] == 1)[0])
    for i in range(split_flags.shape[1])
]

log.info("Calibration method: %s", args.calibration)

y_prob_oof = np.full((len(y), n_classes), np.nan)

for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_splits):
    log.info(
        "  Outer fold %d/%d (train=%d, test=%d)",
        fold_idx + 1,
        len(outer_splits),
        len(outer_train_idx),
        len(outer_test_idx),
    )
    X_outer_train = X[outer_train_idx]
    y_outer_train = y[outer_train_idx]

    inner_fold = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    inner_splits = list(inner_fold.split(X_outer_train, y_outer_train))

    model = minirocket_classifier(
        cv=inner_splits,
        calibration=args.calibration,
        verbose=True,
        random_state=42,
    )
    model.fit(X_outer_train, y_outer_train)

    y_prob_oof[outer_test_idx] = model.predict_proba(X[outer_test_idx])
    log.info("  Outer fold %d/%d done", fold_idx + 1, len(outer_splits))

log.info("Method %s complete", args.calibration)

pd.DataFrame(y_prob_oof, columns=labels).to_csv(args.out, index=False)
