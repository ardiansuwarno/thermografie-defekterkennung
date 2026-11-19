import numpy as np

from src.harmonization import harmonize_time_series


def test_harmonize_pads_short_series_to_target_length():
    # Arrange
    cube = np.ones((4, 4, 50))

    # Act
    result = harmonize_time_series(cube, target_length=100, padding_value=-99.0)

    # Assert
    assert result.shape == (4, 4, 100)
    assert np.all(result[:, :, 50:] == -99.0)


def test_harmonize_truncates_long_series_to_target_length():
    # Arrange
    cube = np.arange(2 * 2 * 120).reshape(2, 2, 120)

    # Act
    result = harmonize_time_series(cube, target_length=100)

    # Assert
    assert result.shape == (2, 2, 100)
    assert np.array_equal(result, cube[:, :, :100])