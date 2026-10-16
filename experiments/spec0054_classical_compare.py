"""Paired CPU logistic baseline; select resolution and L2 inside outer training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import PCA
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


def classification_metrics(truth, probabilities, classes):
    predicted = classes[probabilities.argmax(axis=1)]
    return {
        "count": len(truth),
        "unweighted_ce": log_loss(truth, probabilities, labels=classes),
        "macro_f1": f1_score(truth, predicted, average="macro", labels=classes, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(truth, predicted),
        "multiclass_brier_sum": float(np.mean(np.sum(
            (probabilities - (truth[:, None] == classes)) ** 2, axis=1,
        ))),
        "classification_report": classification_report(
            truth, predicted, labels=classes, output_dict=True, zero_division=0,
        ),
        "confusion_matrix": [[int(np.sum((truth == a) & (predicted == b)))
                              for b in classes] for a in classes],
    }


def nested_partitions(folds, labels, config):
    """Identical WSI partitions for dictionary fitting and paired classifiers."""
    for outer_fold in config["outer_folds"]:
        train = np.flatnonzero(folds != outer_fold)
        holdout = np.flatnonzero(folds == outer_fold)
        inner = [(train[fit], train[valid]) for fit, valid in StratifiedKFold(
            n_splits=config["inner_folds"], shuffle=True, random_state=config["seed"],
        ).split(train, labels[train])]
        yield outer_fold, train, holdout, inner


def run_fold(repo_root, normal_root, so2_root, output_root, config, outer_fold):
    started = time.monotonic()
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
    train = np.flatnonzero(folds != outer_fold)
    holdout = np.flatnonzero(folds == outer_fold)
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
            print(json.dumps({"outer_fold": outer_fold, **candidate}), flush=True)
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
            results[branch][split] = classification_metrics(truth, probabilities, classifier.classes_)
            for i, row in enumerate(rows):
                predictions.append({"branch": branch, "split": split, "wsi_id": int(ids[row]),
                                    "target": int(truth[i]), "predicted": int(predicted[i]),
                                    **{f"p_{int(k)}": float(p) for k, p in
                                       zip(classifier.classes_, probabilities[i], strict=True)}})
    write_csv(output_root / "predictions.csv", predictions)
    summary = {"outer_fold": outer_fold, "config": config, "selected": selected, "results": results,
               "numpy_version": np.__version__, "sklearn_version": sklearn.__version__,
               "elapsed_seconds": time.monotonic() - started,
               "extractions": {branch: json.loads((root / "identity.json").read_text())
                               for branch, root in roots.items()}}
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"outer_fold": outer_fold, "event": "fold_done", "selected": selected,
                      "validation": {b: {k: r["validation"][k] for k in
                                         ("unweighted_ce", "macro_f1", "balanced_accuracy")}
                                     for b, r in results.items()}}), flush=True)
    return summary


def run(repo_root: Path, normal_root: Path, so2_root: Path, output_root: Path):
    started = time.monotonic()
    config = json.loads((repo_root / "docs/data/spec0054_classical_wsi.json").read_text())
    output_root.mkdir(parents=True, exist_ok=False)
    folds = []
    out_of_fold = []
    for outer_fold in config["outer_folds"]:
        fold_root = output_root / f"fold{outer_fold}"
        folds.append(run_fold(repo_root, normal_root, so2_root, fold_root, config, outer_fold))
        with (fold_root / "predictions.csv").open(newline="") as handle:
            out_of_fold.extend({"outer_fold": outer_fold, **row} for row in csv.DictReader(handle)
                               if row["split"] == "validation")
    write_csv(output_root / "out_of_fold_predictions.csv", out_of_fold)
    results = {}
    scalar_metrics = ("unweighted_ce", "macro_f1", "balanced_accuracy", "multiclass_brier_sum")
    for branch in config["branches"]:
        rows = sorted((row for row in out_of_fold if row["branch"] == branch),
                      key=lambda row: int(row["wsi_id"]))
        truth = np.array([int(row["target"]) for row in rows])
        classes = np.unique(truth)
        probabilities = np.array([[float(row[f"p_{k}"]) for k in classes] for row in rows])
        results[branch] = {
            "fold_mean": {k: float(np.mean([f["results"][branch]["validation"][k] for f in folds]))
                          for k in scalar_metrics},
            "fold_std": {k: float(np.std([f["results"][branch]["validation"][k] for f in folds], ddof=1))
                         for k in scalar_metrics},
            "out_of_fold": classification_metrics(truth, probabilities, classes),
        }
    summary = {
        "config": config, "results": results,
        "selections": [{"outer_fold": f["outer_fold"], **f["selected"]} for f in folds],
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy_version": np.__version__, "sklearn_version": sklearn.__version__,
        "elapsed_seconds": time.monotonic() - started,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"event": "comparison_done", **summary}), flush=True)


def pattern_representations(x, histograms, key, fit, config, pca_path=None):
    """Fit the dimension-matched WSI control only on this partition's training."""
    scaler = StandardScaler().fit(x[fit])
    pca = PCA(n_components=config["pattern_histogram"]["pca_components"], svd_solver="full")
    pca.fit(scaler.transform(x[fit]))
    projected = pca.transform(scaler.transform(x))
    if pca_path is not None:
        np.savez(pca_path, mean=scaler.mean_, scale=scaler.scale_,
                 pca_mean=pca.mean_, components=pca.components_)
    return {"mean_std_grid8": x, "mean_std_pca64": projected, "histogram64": histograms[key]}


def aggregate_pattern_folds(folds, predictions, branches):
    results = {}
    metrics = ("unweighted_ce", "macro_f1", "balanced_accuracy", "multiclass_brier_sum")
    for branch in branches:
        selected = sorted((r for r in predictions if r["branch"] == branch), key=lambda r: int(r["wsi_id"]))
        truth = np.array([int(r["target"]) for r in selected])
        classes = np.unique(truth)
        probabilities = np.array([[float(r[f"p_{k}"]) for k in classes] for r in selected])
        # CSV retains FP32 probabilities; normalize their FP64 sum before log_loss.
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        results[branch] = {
            "fold_mean": {m: float(np.mean([f["results"][branch]["validation"][m] for f in folds]))
                          for m in metrics},
            "fold_std": {m: float(np.std([f["results"][branch]["validation"][m] for f in folds], ddof=1))
                         for m in metrics},
            "out_of_fold": classification_metrics(truth, probabilities, classes),
        }
    return results


def run_pattern_comparison(repo_root, normal_root, so2_root, normal_hist_root, so2_hist_root,
                           output_root, representation="pattern_histogram"):
    started = time.monotonic()
    config = json.loads((repo_root / "docs/data/spec0054_classical_wsi.json").read_text())
    branches = config["branches"]
    variants = config[representation]["variants"]
    roots = dict(zip(branches, (normal_root, so2_root), strict=True))
    hist_roots = dict(zip(branches, (normal_hist_root, so2_hist_root), strict=True))
    data = {b: dict(np.load(p / "wsi_features.npz")) for b, p in roots.items()}
    filename = "h2t_features.npz" if representation == "h2t" else "histograms.npz"
    hist = {b: dict(np.load(p / filename)) for b, p in hist_roots.items()}
    training_rows = {b: dict(np.load(p / "training_rows.npz")) for b, p in hist_roots.items()}
    with (repo_root / "docs/data/spec0054_cohort_folds.csv").open(newline="") as handle:
        cohort = sorted(csv.DictReader(handle), key=lambda row: int(row["wsi_id"]))
    ids = np.array([int(row["wsi_id"]) for row in cohort])
    labels = np.array([int(row["diagnosis_index"]) for row in cohort])
    folds = np.array([int(row["fold"]) for row in cohort])
    classes = np.unique(labels)
    for branch in branches:
        np.testing.assert_array_equal(data[branch]["wsi_id"], ids)
        np.testing.assert_array_equal(data[branch]["diagnosis_index"], labels)
        np.testing.assert_array_equal(data[branch]["fold"], folds)
        np.testing.assert_array_equal(hist[branch]["wsi_id"], ids)
    output_root.mkdir(parents=True, exist_ok=False)
    all_folds = {v: [] for v in variants}
    oof = {v: [] for v in variants}
    for outer_fold, train, holdout, inner in nested_partitions(folds, labels, config):
        fold_root = output_root / f"fold{outer_fold}"
        fold_root.mkdir()
        assignments = [{"wsi_id": int(ids[row]), "inner_fold": i}
                       for i, (_, valid) in enumerate(inner) for row in valid]
        write_csv(fold_root / "inner_folds.csv", assignments)
        prepared = {b: [] for b in branches}
        prepared_outer = {}
        for branch in branches:
            x = data[branch]["grid8"]
            for i, (fit, _) in enumerate(inner):
                key = f"fold{outer_fold}_inner{i}"
                np.testing.assert_array_equal(training_rows[branch][key], fit)
                if representation == "h2t":
                    prepared[branch].append({"h2t_mean16": hist[branch][key].astype(np.float64)})
                else:
                    prepared[branch].append(pattern_representations(x, hist[branch], key, fit, config))
            key = f"fold{outer_fold}_outer"
            np.testing.assert_array_equal(training_rows[branch][key], train)
            if representation == "h2t":
                prepared_outer[branch] = {"h2t_mean16": hist[branch][key].astype(np.float64)}
            else:
                prepared_outer[branch] = pattern_representations(
                    x, hist[branch], key, train, config, fold_root / f"{branch}_pca.npz",
                )
        for variant in variants:
            variant_root = fold_root / variant
            variant_root.mkdir()
            scores = []
            candidates = []
            for c in config["C_candidates"]:
                losses = []
                for branch in branches:
                    for i, (fit, valid) in enumerate(inner):
                        x = prepared[branch][i][variant]
                        model, converged = fit_model(x[fit], labels[fit], c, config)
                        ce = log_loss(labels[valid], model.predict_proba(x[valid]), labels=classes)
                        losses.append(ce)
                        scores.append({"branch": branch, "inner_fold": i, "C": c,
                                       "unweighted_ce": ce, "converged": converged,
                                       "iterations": int(model[-1].n_iter_.max())})
                candidates.append({"C": c, "paired_inner_ce": float(np.mean(losses))})
            selected = min(candidates, key=lambda r: r["paired_inner_ce"])
            write_csv(variant_root / "inner_scores.csv", scores)
            write_csv(variant_root / "candidate_scores.csv", candidates)
            results = {}
            predictions = []
            for branch in branches:
                x = prepared_outer[branch][variant]
                model, converged = fit_model(x[train], labels[train], selected["C"], config)
                scaler, classifier = model
                np.savez(variant_root / f"{branch}_model.npz", mean=scaler.mean_, scale=scaler.scale_,
                         coef=classifier.coef_, intercept=classifier.intercept_, classes=classifier.classes_)
                results[branch] = {"converged": converged, "iterations": int(classifier.n_iter_.max())}
                for split, rows in (("train", train), ("validation", holdout)):
                    probabilities = model.predict_proba(x[rows])
                    results[branch][split] = classification_metrics(labels[rows], probabilities, classes)
                    for i, row in enumerate(rows):
                        prediction = {"outer_fold": outer_fold, "branch": branch, "split": split,
                                      "wsi_id": int(ids[row]), "target": int(labels[row]),
                                      "predicted": int(classes[probabilities[i].argmax()]),
                                      **{f"p_{k}": float(p) for k, p in zip(classes, probabilities[i], strict=True)}}
                        predictions.append(prediction)
                        if split == "validation":
                            oof[variant].append(prediction)
            write_csv(variant_root / "predictions.csv", predictions)
            summary = {"outer_fold": outer_fold, "variant": variant, "selected": selected, "results": results}
            (variant_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            all_folds[variant].append(summary)
            print(json.dumps({"event": "pattern_fold_done", **summary}), flush=True)
    results = {}
    for variant in variants:
        write_csv(output_root / f"{variant}_out_of_fold.csv", oof[variant])
        results[variant] = {
            "selections": [{"outer_fold": f["outer_fold"], **f["selected"]} for f in all_folds[variant]],
            "branches": aggregate_pattern_folds(all_folds[variant], oof[variant], branches),
        }
    summary = {"config": config, "representation": representation,
               "results": results, "elapsed_seconds": time.monotonic() - started,
               "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "numpy_version": np.__version__, "sklearn_version": sklearn.__version__,
               "histogram_sources": {b: json.loads((p / "identity.json").read_text())
                                     for b, p in hist_roots.items()}}
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"event": "pattern_comparison_done", "results": results}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--normal-root", type=Path, required=True)
    parser.add_argument("--so2-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--normal-hist-root", type=Path)
    parser.add_argument("--so2-hist-root", type=Path)
    parser.add_argument("--representation", choices=("pattern_histogram", "h2t"), default="pattern_histogram")
    args = parser.parse_args()
    if args.normal_hist_root is not None:
        run_pattern_comparison(args.repo_root, args.normal_root, args.so2_root,
                               args.normal_hist_root, args.so2_hist_root, args.output_root, args.representation)
    else:
        run(args.repo_root, args.normal_root, args.so2_root, args.output_root)
