import numpy as np

from src.wavelet_features import extract_wavelet_features


def test_extract_wavelet_features_reduces_time_dimension():
    # Arrange
    rng = np.random.default_rng(0)
    cube = rng.standard_normal((8, 10, 128))

    # Act
    features, spatial_shape = extract_wavelet_features(cube, wavelet_name="haar", level=4)

    # Assert
    assert spatial_shape == (8, 10)
    assert features.shape[0] == 80
    assert features.shape[1] < 128
