"""Direct Gaussian and MAP moment checks for the cached PANTHER experiment."""

import numpy as np
from scipy.special import softmax
from threadpoolctl import threadpool_limits

from experiments.spec0054_panther import bag_panther_features, expectation, panther_representations
from experiments.spec0054_pattern_histograms import fit_vocabulary


def recipe():
    return {"seed": 1701, "panther": {"clusters": 3, "patches_per_wsi": 4,
            "max_iter": 20, "n_init": 1, "assignment_batch": 2,
            "tau": 1., "initial_variance": 1.}}


def test_expectation_matches_explicit_gaussian_distances():
    values = np.array([[1., 2.], [4., -3.], [0., 0.]])
    centers = np.array([[0., 1.], [2., -1.], [-4., 2.]])
    distance = np.sum((values[:, None] - centers[None]) ** 2, axis=-1)
    q, log_q, likelihood, distances = expectation(values, centers, 1.)
    np.testing.assert_allclose(q, softmax(-distance / 2, axis=1))
    np.testing.assert_allclose(np.exp(log_q), q)
    np.testing.assert_allclose(distances, distance)
    expected_density = np.exp(-distance / 2) / (2 * np.pi)
    np.testing.assert_allclose(likelihood, np.log(expected_density.mean(axis=1)))


def test_chunked_map_matches_full_bag_reference_and_output_slices():
    rng = np.random.default_rng(1781)
    values = rng.normal(size=(17, 4)).astype(np.float32)
    owners = np.repeat([0, 1], [9, 8])
    data = {"wsi_id": np.array([4, 281]), "patch_offset": np.array([0, 9]),
            "patch_count": np.array([9, 8])}
    config = recipe()
    with threadpool_limits(limits=1):
        scaler, model = fit_vocabulary(values, owners, np.array([0]), config, "panther")
        table, masses, _ = bag_panther_features(values, data, {"split": (scaler, model)}, config)
    centers = model.cluster_centers_.astype(np.float64)
    for row, selected in enumerate((values[:9].astype(np.float64), values[9:].astype(np.float64))):
        q = softmax(-np.sum((selected[:, None] - centers[None]) ** 2, axis=-1) / 2, axis=1)
        mass = q.sum(axis=0)
        pi = (mass + 1) / (len(selected) + 3)
        mu = (q.T @ selected + centers) / (mass + 1)[:, None]
        variance = ((q.T @ (selected ** 2) + 1 + centers ** 2) / (mass + 1)[:, None] - mu ** 2)
        expected = np.concatenate((pi, mu.reshape(-1), variance.reshape(-1)))
        np.testing.assert_allclose(table["split"][row], expected, atol=1e-7)
        np.testing.assert_allclose(masses["split"][row], mass)
    split = panther_representations(table["split"], config)
    np.testing.assert_array_equal(split["panther_all16"], table["split"])
    np.testing.assert_array_equal(split["panther_mean16"], table["split"][:, 3:15])


def test_holdout_does_not_change_raw_training_centers():
    samples = np.random.default_rng(1791).normal(size=(30, 4)).astype(np.float32)
    owners = np.repeat([0, 1, 2], 10)
    changed = samples.copy()
    changed[owners == 2] += 1000
    with threadpool_limits(limits=1):
        scaler, model = fit_vocabulary(samples, owners, np.array([0, 1]), recipe(), "panther")
        _, other = fit_vocabulary(changed, owners, np.array([0, 1]), recipe(), "panther")
    np.testing.assert_array_equal(scaler.transform(samples), samples)
    np.testing.assert_array_equal(model.cluster_centers_, other.cluster_centers_)


def test_prior_keeps_negligible_components_finite_and_positive():
    values = np.array([[0., 0.], [0.1, 0.1]], dtype=np.float32)
    config = recipe()

    class Vocabulary:
        cluster_centers_ = np.array([[0., 0.], [100., 100.], [-100., -100.]])

    data = {"wsi_id": np.array([4]), "patch_offset": np.array([0]), "patch_count": np.array([2])}
    table, mass, _ = bag_panther_features(values, data, {"split": (None, Vocabulary())}, config)
    result = table["split"][0]
    np.testing.assert_allclose(result[:3], [3 / 5, 1 / 5, 1 / 5])
    np.testing.assert_allclose(result[5:9], [100, 100, -100, -100])
    np.testing.assert_allclose(result[-4:], 1)
    np.testing.assert_allclose(mass["split"].sum(), 2)
    assert np.isfinite(result).all()
    assert np.all(result[9:] > 0)
