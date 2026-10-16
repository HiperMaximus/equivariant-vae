"""Training isolation and complete-bag proportions for patch vocabularies."""

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.spec0054_classical_compare import pattern_representations
from experiments.spec0054_pattern_histograms import (
    bag_h2t_features, bag_histograms, fit_vocabulary, sampled_patch_rows,
)


def recipe():
    return {"seed": 1701, "pattern_histogram": {
        "clusters": 3, "patches_per_wsi": 3, "batch_size": 16,
        "max_iter": 20, "n_init": 2, "assignment_batch": 3, "pca_components": 2,
    }}


def test_patch_sampling_is_capped_and_wsi_seeded():
    data = {"wsi_id": np.array([4, 281, 286]), "patch_offset": np.array([0, 5, 7]),
            "patch_count": np.array([5, 2, 7])}
    rows, owners = sampled_patch_rows(data, recipe())
    again, again_owners = sampled_patch_rows(data, recipe())
    np.testing.assert_array_equal(rows, again)
    np.testing.assert_array_equal(owners, again_owners)
    np.testing.assert_array_equal(np.bincount(owners), [3, 2, 3])
    for wsi in range(3):
        selected = rows[owners == wsi]
        assert len(np.unique(selected)) == len(selected)
        assert np.all(selected >= data["patch_offset"][wsi])
        assert np.all(selected < data["patch_offset"][wsi] + data["patch_count"][wsi])


def test_holdout_patches_do_not_change_scaling_or_centers():
    samples = np.random.default_rng(1701).normal(size=(80, 5)).astype(np.float32)
    owners = np.repeat(np.arange(8), 10)
    training = np.arange(6)
    changed = samples.copy()
    changed[owners >= 6] += 1000
    with threadpool_limits(limits=1):
        scaler, centers = fit_vocabulary(samples, owners, training, recipe())
        second_scaler, second_centers = fit_vocabulary(changed, owners, training, recipe())
    np.testing.assert_allclose(scaler.mean_, samples[owners < 6].mean(axis=0, dtype=np.float64))
    np.testing.assert_array_equal(scaler.mean_, second_scaler.mean_)
    np.testing.assert_array_equal(centers.cluster_centers_, second_centers.cluster_centers_)


def test_chunked_histogram_matches_direct_complete_bag_counts():
    samples = np.random.default_rng(1711).normal(size=(80, 5)).astype(np.float32)
    data = {"wsi_id": np.array([4, 281]), "patch_offset": np.array([0, 31]),
            "patch_count": np.array([31, 49])}
    with threadpool_limits(limits=1):
        scaler, model = fit_vocabulary(samples, np.repeat([0, 1], [31, 49]), np.array([0]), recipe())
        table = bag_histograms(samples, data, {"vocabulary": (scaler, model)}, recipe())["vocabulary"]
        for row, (offset, count) in enumerate(zip(data["patch_offset"], data["patch_count"], strict=True)):
            assigned = model.predict(scaler.transform(samples[offset:offset + count]))
            expected = np.bincount(assigned, minlength=3) / count
            np.testing.assert_allclose(table[row], expected, atol=1e-7)
    np.testing.assert_allclose(table.sum(axis=1), 1, atol=1e-7)


def test_pca_control_does_not_learn_from_holdout(tmp_path):
    values = np.random.default_rng(1721).normal(size=(20, 8))
    training = np.arange(15)
    changed = values.copy()
    changed[15:] += 1000
    hist = {"split": np.ones((20, 3)) / 3}
    first = pattern_representations(values, hist, "split", training, recipe(), tmp_path / "first.npz")
    second = pattern_representations(changed, hist, "split", training, recipe(), tmp_path / "second.npz")
    np.testing.assert_array_equal(first["mean_std_pca64"][training], second["mean_std_pca64"][training])
    with np.load(tmp_path / "first.npz") as a, np.load(tmp_path / "second.npz") as b:
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])


def test_h2t_unit_centers_use_only_training_patches():
    config = recipe()
    config["h2t"] = dict(config["pattern_histogram"])
    samples = np.random.default_rng(1731).normal(size=(80, 5)).astype(np.float32)
    owners = np.repeat(np.arange(8), 10)
    training = np.arange(6)
    changed = samples.copy()
    changed[owners >= 6] *= -1000
    with threadpool_limits(limits=1):
        normalizer, model = fit_vocabulary(samples, owners, training, config, "h2t")
        _, second = fit_vocabulary(changed, owners, training, config, "h2t")
    normalized = normalizer.transform(samples)
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1, atol=2e-7)
    np.testing.assert_allclose(np.linalg.norm(model.cluster_centers_, axis=1), 1, atol=2e-7)
    np.testing.assert_array_equal(model.cluster_centers_, second.cluster_centers_)
    np.testing.assert_array_equal(normalizer.transform(np.zeros((1, 5))), np.zeros((1, 5)))


def test_h2t_chunked_raw_means_and_empty_patterns_match_direct_formula():
    config = recipe()
    config["h2t"] = dict(config["pattern_histogram"])
    samples = np.array([[10., 0.], [2., 0.], [0., 3.], [0., 4.],
                        [-2., 0.], [-6., 0.]], dtype=np.float32)
    data = {"wsi_id": np.array([4, 281]), "patch_offset": np.array([0, 4]),
            "patch_count": np.array([4, 2])}
    with threadpool_limits(limits=1):
        normalizer, model = fit_vocabulary(samples, np.zeros(6, dtype=int), np.array([0]), config, "h2t")
        tables, counts = bag_h2t_features(samples, data, {"split": (normalizer, model)}, config)
        for row, (offset, count) in enumerate(zip(data["patch_offset"], data["patch_count"], strict=True)):
            values = samples[offset:offset + count]
            assigned = model.predict(normalizer.transform(values))
            expected = np.zeros((3, 2))
            for group in np.unique(assigned):
                expected[group] = values[assigned == group].mean(axis=0)
            np.testing.assert_allclose(tables["split"][row], expected.reshape(-1))
            np.testing.assert_array_equal(counts["split"][row], np.bincount(assigned, minlength=3))
    assert np.any(counts["split"] == 0)
    assert np.max(np.abs(tables["split"])) > 1  # Pool raw embeddings, not unit vectors.
