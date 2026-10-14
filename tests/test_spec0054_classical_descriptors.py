"""Check spatial/channel ordering and streaming population dispersion."""

import numpy as np
import pytest

from experiments.spec0054_classical_extract import merge_moments, pool_patches


@pytest.mark.parametrize("grid", [4, 8, 16])
def test_block_means_preserve_channel_and_spatial_order(grid):
    values = np.arange(2 * 16 * 32 * 32, dtype=np.float32).reshape(2, 16, 32, 32) / 100
    values = values.astype(np.float16)
    block = 32 // grid
    expected = np.stack([
        np.stack([values[:, channel, y:y + block, x:x + block].mean(axis=(1, 2), dtype=np.float32)
                  for y in range(0, 32, block) for x in range(0, 32, block)], axis=1)
        for channel in range(16)
    ], axis=1).reshape(2, -1)
    np.testing.assert_allclose(pool_patches(values, grid), expected, rtol=1e-6, atol=1e-6)


def test_chunked_moments_match_full_population_std():
    values = np.random.default_rng(1701).normal(size=(23, 17)) + 10000
    moments = (0, np.zeros(17), np.zeros(17))
    for chunk in np.array_split(values, [1, 8, 13]):
        moments = merge_moments(*moments, chunk)
    count, mean, m2 = moments
    np.testing.assert_allclose(mean, values.mean(axis=0), atol=1e-10, rtol=0)
    np.testing.assert_allclose(np.sqrt(m2 / count), values.std(axis=0, ddof=0), atol=1e-11, rtol=0)
