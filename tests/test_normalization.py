import numpy as np

from src.normalization import z_standardize


def test_z_standardize_produces_zero_mean_and_unit_std():
    # Arrange
    rng = np.random.default_rng(42)
    cube = rng.normal(loc=300.0, scale=15.0, size=(5, 5, 200))

    # Act
    result = z_standardize(cube)

    # Assert
    assert result.shape == cube.shape
    assert np.allclose(np.mean(result, axis=2), 0.0, atol=1e-6)
    assert np.allclose(np.std(result, axis=2), 1.0, atol=1e-5)


def test_z_standardize_handles_constant_signal_without_nan():
    # Arrange
    cube = np.full((3, 3, 50), 310.0)

    # Act
    result = z_standardize(cube)

    # Assert
    assert not np.isnan(result).any()
    assert np.allclose(result, 0.0)