import numpy as np
import pytest

from src.flattening import flatten_for_ml, reconstruct_image


def test_flatten_for_ml_returns_2d_matrix_and_spatial_shape():
    # Arrange
    cube = np.zeros((10, 12, 8))

    # Act
    matrix, spatial_shape = flatten_for_ml(cube)

    # Assert
    assert matrix.shape == (120, 8)
    assert spatial_shape == (10, 12)


def test_reconstruct_image_restores_2d_shape():
    # Arrange
    predictions = np.arange(20)

    # Act
    image = reconstruct_image(predictions, spatial_shape=(4, 5))

    # Assert
    assert image.shape == (4, 5)


def test_reconstruct_image_raises_on_mismatched_shape():
    # Arrange
    predictions = np.zeros(10)

    # Act & Assert
    with pytest.raises(ValueError):
        reconstruct_image(predictions, spatial_shape=(4, 5))