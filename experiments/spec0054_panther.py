"""Chunked one-step MAP-EM statistics following the PANTHER reference recipe.

Reference: mahmoodlab/PANTHER e7ffae402e146363fe1ca3813ffe68d248ec570f,
src/mil_models/PANTHER/networks.py and src/scripts/embedding/panther.sh.
Independent NumPy formulation; FP64 moments use a common numerical origin.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.special import logsumexp


def expectation(values, centers, initial_variance):
    """Uniform initial mixture with isotropic diagonal covariance."""
    cross = values @ centers.T
    center_norm = np.sum(centers ** 2, axis=1)
    scores = (cross - 0.5 * center_norm) / initial_variance
    normalizer = logsumexp(scores, axis=1)
    log_responsibilities = scores - normalizer[:, None]
    responsibilities = np.exp(log_responsibilities)
    distances = np.maximum(np.sum(values ** 2, axis=1)[:, None] + center_norm - 2 * cross, 0)
    likelihood = (normalizer - np.log(len(centers))
                  - 0.5 * values.shape[1] * np.log(2 * np.pi * initial_variance)
                  - 0.5 * np.sum(values ** 2, axis=1) / initial_variance)
    return responsibilities, log_responsibilities, likelihood, distances


def map_parameters(mass, first, second, centers, origin, tau, initial_variance):
    """Reference MAP M-step; prior contributes tau virtual observations."""
    denominator = mass + tau
    prior_mean = centers - origin
    mean = (first + tau * prior_mean) / denominator[:, None]
    variance = ((second + tau * (initial_variance + prior_mean ** 2))
                / denominator[:, None] - mean ** 2)
    proportions = denominator / denominator.sum()
    return proportions, mean + origin, variance


def sample_diagnostics(samples, owners, training, centers, config):
    """Observe train/held-out samples without updating reference prototypes."""
    recipe = config["panther"]
    origin = centers.mean(axis=0, dtype=np.float64)
    values = samples.astype(np.float64) - origin
    shifted = centers.astype(np.float64) - origin
    q, log_q, likelihood, distances = expectation(values, shifted, recipe["initial_variance"])
    result = {}
    train_mask = np.isin(owners, training)
    for split, mask in (("train", train_mask), ("heldout", ~train_mask)):
        selected = q[mask]
        result[split] = {
            "patches": int(mask.sum()),
            "mean_log_likelihood": float(likelihood[mask].mean()),
            "quantization_residual": float(distances[mask].min(axis=1).mean()),
            "responsibility_weighted_residual": float(np.sum(selected * distances[mask], axis=1).mean()),
            "responsibility_entropy": float(-np.sum(selected * log_q[mask], axis=1).mean()),
            "component_mass": selected.sum(axis=0).tolist(),
        }
    return result


def bag_panther_features(cache, data, vocabularies, config):
    """One reference MAP-EM update on every complete cached WSI bag."""
    recipe = config["panther"]
    clusters, features = recipe["clusters"], cache.shape[1]
    tables = {key: np.empty((len(data["wsi_id"]), clusters * (1 + 2 * features)), np.float32)
              for key in vocabularies}
    masses = {key: np.empty((len(data["wsi_id"]), clusters), np.float64) for key in vocabularies}
    diagnostics = []
    for row, (wsi, offset, count) in enumerate(zip(
        data["wsi_id"], data["patch_offset"], data["patch_count"], strict=True,
    )):
        # A change of numerical origin leaves probabilities and moments invariant.
        origin = np.array(cache[int(offset)], dtype=np.float64, copy=True)
        centers = {key: model.cluster_centers_.astype(np.float64) for key, (_, model) in vocabularies.items()}
        moments = {key: [np.zeros(clusters), np.zeros((clusters, features)),
                         np.zeros((clusters, features)), 0., 0.] for key in vocabularies}
        end = int(offset + count)
        for first in range(int(offset), end, recipe["assignment_batch"]):
            values = np.array(cache[first:min(first + recipe["assignment_batch"], end)], dtype=np.float64)
            values -= origin
            squared = values ** 2
            for key in vocabularies:
                q, log_q, likelihood, _ = expectation(values, centers[key] - origin, recipe["initial_variance"])
                moments[key][0] += q.sum(axis=0)
                moments[key][1] += q.T @ values
                moments[key][2] += q.T @ squared
                moments[key][3] -= float(np.sum(q * log_q))
                moments[key][4] += float(likelihood.sum())
        for key in vocabularies:
            mass, first, second, entropy, likelihood = moments[key]
            pi, mu, variance = map_parameters(
                mass, first, second, centers[key], origin, recipe["tau"], recipe["initial_variance"],
            )
            # Official allcat order: all proportions, all means, all variances.
            tables[key][row] = np.concatenate((pi, mu.reshape(-1), variance.reshape(-1)))
            masses[key][row] = mass
            diagnostics.append({
                "key": key, "wsi_id": int(wsi), "patches": int(count),
                "responsibility_entropy": entropy / count,
                "initial_mean_log_likelihood": likelihood / count,
                "effective_components": float(np.exp(-np.sum(pi * np.log(pi)))),
                "minimum_component_mass": float(mass.min()),
                "variance_min": float(variance.min()), "variance_max": float(variance.max()),
                "max_diagonal_condition": float(np.max(variance.max(axis=1) / variance.min(axis=1))),
            })
        print(json.dumps({"event": "wsi_panther_done", "wsi_id": int(wsi),
                          "wsi_done": row + 1, "vocabularies": len(vocabularies)}), flush=True)
    return tables, masses, diagnostics


def panther_representations(values, config):
    clusters = config["panther"]["clusters"]
    means_end = clusters + (values.shape[1] - clusters) // 2
    return {"panther_all16": values.astype(np.float64),
            "panther_mean16": values[:, clusters:means_end].astype(np.float64)}
