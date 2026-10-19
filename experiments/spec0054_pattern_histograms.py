"""Train-only patch vocabularies, bag proportions and H2T means on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import FunctionTransformer, Normalizer, StandardScaler
from threadpoolctl import threadpool_limits

from experiments.spec0054_classical_compare import nested_partitions


def sampled_patch_rows(data, config, representation="pattern_histogram"):
    rows, owners = [], []
    for wsi_row, (wsi, offset, count) in enumerate(zip(
        data["wsi_id"], data["patch_offset"], data["patch_count"], strict=True,
    )):
        generator = np.random.default_rng(np.random.SeedSequence([config["seed"], int(wsi)]))
        selected = np.sort(generator.choice(
            int(count), size=min(int(count), config[representation]["patches_per_wsi"]),
            replace=False,
        )) + int(offset)
        rows.extend(selected.tolist())
        owners.extend([wsi_row] * len(selected))
    return np.array(rows, dtype=np.int64), np.array(owners, dtype=np.int64)


def fit_vocabulary(samples, owners, training_rows, config, representation="pattern_histogram"):
    """Only the indicated WSI contribute to scaling and centroid estimation."""
    values = samples[np.isin(owners, training_rows)]
    if representation == "panther":
        scaler = FunctionTransformer()
    else:
        scaler = Normalizer(norm="l2") if representation == "h2t" else StandardScaler()
    normalized = scaler.fit_transform(values)
    recipe = config[representation]
    if representation == "panther":
        model = KMeans(n_clusters=recipe["clusters"], max_iter=recipe["max_iter"],
                       n_init=recipe["n_init"], random_state=config["seed"]).fit(normalized)
    else:
        model = MiniBatchKMeans(
            n_clusters=recipe["clusters"], batch_size=recipe["batch_size"],
            max_iter=recipe["max_iter"], n_init=recipe["n_init"], random_state=config["seed"],
        ).fit(normalized)
    if representation == "h2t":
        model.cluster_centers_ = scaler.transform(model.cluster_centers_)
    return scaler, model


def bag_histograms(cache, data, vocabularies, config):
    recipe = config["pattern_histogram"]
    tables = {key: np.empty((len(data["wsi_id"]), recipe["clusters"]), dtype=np.float32)
              for key in vocabularies}
    for row, (wsi, offset, count) in enumerate(zip(
        data["wsi_id"], data["patch_offset"], data["patch_count"], strict=True,
    )):
        totals = {key: np.zeros(recipe["clusters"], dtype=np.int64) for key in vocabularies}
        end = int(offset + count)
        for first in range(int(offset), end, recipe["assignment_batch"]):
            # Materialize each block once; all fitted vocabularies reuse it.
            values = np.array(cache[first:min(first + recipe["assignment_batch"], end)], copy=True)
            for key, (scaler, model) in vocabularies.items():
                assigned = model.predict(scaler.transform(values))
                totals[key] += np.bincount(assigned, minlength=recipe["clusters"])
        for key in vocabularies:
            tables[key][row] = totals[key] / count
        print(json.dumps({"event": "wsi_histograms_done", "wsi_id": int(wsi),
                          "wsi_done": row + 1, "vocabularies": len(vocabularies)}), flush=True)
    return tables


def bag_h2t_features(cache, data, vocabularies, config):
    """Assign normalized patches; concatenate per-pattern means of raw features."""
    recipe = config["h2t"]
    clusters, features = recipe["clusters"], cache.shape[1]
    tables = {key: np.empty((len(data["wsi_id"]), clusters * features), dtype=np.float32)
              for key in vocabularies}
    counts = {key: np.empty((len(data["wsi_id"]), clusters), dtype=np.int64)
              for key in vocabularies}
    for row, (wsi, offset, count) in enumerate(zip(
        data["wsi_id"], data["patch_offset"], data["patch_count"], strict=True,
    )):
        totals = {key: np.zeros((clusters, features), dtype=np.float64) for key in vocabularies}
        occupied = {key: np.zeros(clusters, dtype=np.int64) for key in vocabularies}
        end = int(offset + count)
        for first in range(int(offset), end, recipe["assignment_batch"]):
            values = np.array(cache[first:min(first + recipe["assignment_batch"], end)], copy=True)
            normalized = Normalizer(norm="l2").transform(values)
            for key, (_, model) in vocabularies.items():
                assigned = model.predict(normalized)
                occupied[key] += np.bincount(assigned, minlength=clusters)
                for group in range(clusters):
                    totals[key][group] += values[assigned == group].sum(axis=0, dtype=np.float64)
        for key in vocabularies:
            means = np.zeros_like(totals[key])
            np.divide(totals[key], occupied[key][:, None], out=means,
                      where=occupied[key][:, None] != 0)
            tables[key][row] = means.reshape(-1)
            counts[key][row] = occupied[key]
        print(json.dumps({"event": "wsi_h2t_done", "wsi_id": int(wsi),
                          "wsi_done": row + 1, "vocabularies": len(vocabularies)}), flush=True)
    return tables, counts


def run(repo_root: Path, cache_root: Path, output_root: Path, branch: str,
        representation="pattern_histogram"):
    started = time.monotonic()
    config_path = repo_root / "docs/data/spec0054_classical_wsi.json"
    config = json.loads(config_path.read_text())
    recipe = config[representation]
    source_identity = json.loads((cache_root / "identity.json").read_text())
    observed_hash = hashlib.sha256((cache_root / "identity.json").read_bytes()).hexdigest()
    np.testing.assert_equal(observed_hash, recipe["sources"][branch]["identity_sha256"])
    with np.load(cache_root / "wsi_features.npz") as stored:
        data = {key: stored[key] for key in stored.files}
    cache = np.load(cache_root / "patches_grid8.npy", mmap_mode="r")
    selected_rows, owners = sampled_patch_rows(data, config, representation)
    samples = np.array(cache[selected_rows], copy=True)
    output_root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output_root / config_path.name)
    np.savez(output_root / "sampled_patches.npz", patch_row=selected_rows,
             wsi_row=owners, wsi_id=data["wsi_id"])
    vocabularies = {}
    training = {}
    sample_audit = {}
    with threadpool_limits(limits=recipe["cpu_threads"]):
        for outer_fold, train, _, inner in nested_partitions(data["fold"], data["diagnosis_index"], config):
            splits = [(f"fold{outer_fold}_inner{i}", fit) for i, (fit, _) in enumerate(inner)]
            splits.append((f"fold{outer_fold}_outer", train))
            for key, fit in splits:
                fit_started = time.monotonic()
                scaler, model = fit_vocabulary(samples, owners, fit, config, representation)
                vocabularies[key] = scaler, model
                training[key] = fit
                parameters = {"training_wsi_id": data["wsi_id"][fit], "centers": model.cluster_centers_}
                if representation == "pattern_histogram":
                    parameters.update(patch_mean=scaler.mean_, patch_scale=scaler.scale_)
                np.savez(output_root / f"{key}_vocabulary.npz", **parameters)
                print(json.dumps({"event": "vocabulary_fitted", "branch": branch, "key": key,
                                  "training_wsi": len(fit), "iterations": int(model.n_iter_),
                                  "kmeans_fit_inertia": float(model.inertia_),
                                  "seconds": time.monotonic() - fit_started}), flush=True)
                if representation == "panther":
                    from experiments.spec0054_panther import sample_diagnostics
                    sample_audit[key] = sample_diagnostics(samples, owners, fit, model.cluster_centers_, config)
        if representation == "panther":
            from experiments.spec0054_panther import bag_panther_features
            (output_root / "prototype_sample_audit.json").write_text(json.dumps(sample_audit, indent=2) + "\n")
            tables, masses, diagnostics = bag_panther_features(cache, data, vocabularies, config)
            np.savez(output_root / "panther_masses.npz", wsi_id=data["wsi_id"], **masses)
            (output_root / "panther_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
        elif representation == "h2t":
            tables, counts = bag_h2t_features(cache, data, vocabularies, config)
            np.savez(output_root / "h2t_counts.npz", wsi_id=data["wsi_id"], **counts)
        else:
            tables = bag_histograms(cache, data, vocabularies, config)
    filename = {"panther": "panther_features.npz", "h2t": "h2t_features.npz",
                "pattern_histogram": "histograms.npz"}[representation]
    np.savez(output_root / filename, wsi_id=data["wsi_id"], **tables)
    np.savez(output_root / "training_rows.npz", **training)
    identity = {
        "branch": branch, "representation": representation,
        "config": config, "source_identity": source_identity,
        "source_identity_sha256": observed_hash,
        "source_commit": subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True,
        ).strip(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sampled_rows_sha256": hashlib.sha256(selected_rows.astype("<i8").tobytes()).hexdigest(),
        "sampled_patches": len(samples), "wsi_rows": len(data["wsi_id"]),
        "vocabularies": len(vocabularies), "elapsed_seconds": time.monotonic() - started,
        "numpy_version": np.__version__, "sklearn_version": sklearn.__version__,
    }
    (output_root / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    print(json.dumps({"event": "bag_representations_done", **identity}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--branch", choices=("normal_vae", "so2_vae"), required=True)
    parser.add_argument("--representation", choices=("pattern_histogram", "h2t", "panther"), default="pattern_histogram")
    args = parser.parse_args()
    run(args.repo_root, args.cache_root, args.output_root, args.branch, args.representation)
