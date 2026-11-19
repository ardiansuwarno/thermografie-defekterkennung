import numpy as np

from src.flattening import reconstruct_image
from src.harmonization import harmonize_time_series
from src.normalization import z_standardize
from src.wavelet_features import extract_wavelet_features


def test_signal_processing_pipeline_end_to_end():
    # Arrange
    rng = np.random.default_rng(7)
    log_cube = np.log(rng.uniform(300.0, 320.0, size=(6, 8, 90)))

    # Act
    harmonized = harmonize_time_series(log_cube, target_length=128)
    standardized = z_standardize(harmonized)
    features, spatial_shape = extract_wavelet_features(standardized, wavelet_name="haar", level=4)
    dummy_preds = np.zeros(features.shape[0], dtype=int)
    mask = reconstruct_image(dummy_preds, spatial_shape)

    # Assert
    assert mask.shape == (6, 8)