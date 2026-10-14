"""Paired CPU logistic baseline; select resolution and L2 inside outer training."""

from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from pathlib import Path

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_model(x, y, c, config):
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=c, solver=config["solver"], class_weight=config["class_weight"],
        max_iter=config["max_iter"], tol=config["tol"],
    ))
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y)
    converged = not any(issubclass(item.category, ConvergenceWarning) for item in recorded)
    for item in recorded:
        print(str(item.message), flush=True)
    return model, converged


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(repo_root: Path, normal_root: Path, so2_root: Path, output_root: Path):
    started = time.monotonic()
    config = json.loads((repo_root / "docs/data/spec0054_classical_wsi.json").read_text())
    roots = dict(zip(config["branches"], (normal_root, so2_root), strict=True))
    data = {branch: dict(np.load(root / "wsi_features.npz")) for branch, root in roots.items()}
    # Align to the authoritative cohort rather than trusting positional equality.
    with (repo_root / "docs/data/spec0054_cohort_folds.csv").open(newline="") as handle:
        cohort = sorted(csv.DictReader(handle), key=lambda row: int(row["wsi_id"]))
    ids = np.array([int(row["wsi_id"]) for row in cohort])
    labels = np.array([int(row["diagnosis_index"]) for row in cohort])
    folds = np.array([int(row["fold"]) for row in cohort])
    classes = np.unique(labels)
    for branch, arrays in data.items():
        np.testing.assert_array_equal(np.sort(arrays["wsi_id"]), ids)
        positions = {int(wsi): row for row, wsi in enumerate(arrays["wsi_id"])}
        order = np.array([positions[int(wsi)] for wsi in ids])
        data[branch] = {name: value[order] for name, value in arrays.items()}
        np.testing.assert_array_equal(data[branch]["diagnosis_index"], labels)
        np.testing.assert_array_equal(data[branch]["fold"], folds)
    train = np.flatnonzero(folds != config["outer_fold"])
    holdout = np.flatnonzero(folds == config["outer_fold"])
    cv = list(StratifiedKFold(
        n_splits=config["inner_folds"], shuffle=True, random_state=config["seed"],
    ).split(train, labels[train]))
    output_root.mkdir(parents=True, exist_ok=False)
    assignments = []
    for inner_fold, (_, valid) in enumerate(cv):
        assignments.extend({"wsi_id": int(ids[train[row]]), "inner_fold": inner_fold} for row in valid)
    write_csv(output_root / "inner_folds.csv", assignments)
    scores = []
    candidates = []
    for grid in config["pool_grids"]:
        for c in config["C_candidates"]:
            paired_ce = []
            for branch in config["branches"]:
                x = data[branch][f"grid{grid}"]
                for inner_fold, (fit, valid) in enumerate(cv):
                    model, converged = fit_model(x[train[fit]], labels[train[fit]], c, config)
                    ce = log_loss(labels[train[valid]], model.predict_proba(x[train[valid]]), labels=classes)
                    paired_ce.append(ce)
                    scores.append({"branch": branch, "grid": grid, "C": c,
                                   "inner_fold": inner_fold, "unweighted_ce": ce,
                                   "converged": converged,
                                   "iterations": int(model[-1].n_iter_.max())})
            candidate = {"grid": grid, "C": c, "paired_inner_ce": float(np.mean(paired_ce))}
            candidates.append(candidate)
            print(json.dumps(candidate), flush=True)
    write_csv(output_root / "inner_scores.csv", scores)
    write_csv(output_root / "candidate_scores.csv", candidates)
    selected = min(candidates, key=lambda row: row["paired_inner_ce"])
    results = {}
    predictions = []
    for branch in config["branches"]:
        x = data[branch][f"grid{selected['grid']}"]
        model, converged = fit_model(x[train], labels[train], selected["C"], config)
        scaler, classifier = model
        np.savez(output_root / f"{branch}_model.npz", mean=scaler.mean_, scale=scaler.scale_,
                 coef=classifier.coef_, intercept=classifier.intercept_, classes=classifier.classes_)
        results[branch] = {"converged": converged, "iterations": int(classifier.n_iter_.max())}
        for split, rows in (("train", train), ("validation", holdout)):
            probabilities = model.predict_proba(x[rows])
            predicted = classifier.classes_[probabilities.argmax(axis=1)]
            truth = labels[rows]
            results[branch][split] = {
                "count": len(rows),
                "unweighted_ce": log_loss(truth, probabilities, labels=classes),
                "macro_f1": f1_score(truth, predicted, average="macro", labels=classes, zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(truth, predicted),
                "multiclass_brier_sum": float(np.mean(np.sum(
                    (probabilities - (truth[:, None] == classifier.classes_)) ** 2, axis=1,
                ))),
                "classification_report": classification_report(
                    truth, predicted, labels=classes, output_dict=True, zero_division=0,
                ),
                "confusion_matrix": [[int(np.sum((truth == a) & (predicted == b)))
                                      for b in classes] for a in classes],
            }
            for i, row in enumerate(rows):
                predictions.append({"branch": branch, "split": split, "wsi_id": int(ids[row]),
                                    "target": int(truth[i]), "predicted": int(predicted[i]),
                                    **{f"p_{int(k)}": float(p) for k, p in
                                       zip(classifier.classes_, probabilities[i], strict=True)}})
    write_csv(output_root / "predictions.csv", predictions)
    summary = {"config": config, "selected": selected, "results": results,
               "numpy_version": np.__version__, "sklearn_version": sklearn.__version__,
               "elapsed_seconds": time.monotonic() - started,
               "extractions": {branch: json.loads((root / "identity.json").read_text())
                               for branch, root in roots.items()}}
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"selected": selected, "results": results}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--normal-root", type=Path, required=True)
    parser.add_argument("--so2-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.repo_root, args.normal_root, args.so2_root, args.output_root)
