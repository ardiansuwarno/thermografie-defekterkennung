import importlib.util
from pathlib import Path

import numpy as np


def _load_postprocessing():
    path = Path("src/ML/05_postprocessing.py")
    spec = importlib.util.spec_from_file_location("postprocessing", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_morphological_filter_removes_single_noise_pixel():
    # Arrange
    mod = _load_postprocessing()
    mask = np.zeros((15, 15), dtype=np.uint8)
    mask[2, 2] = 1  # Einzelnes Rauschpixel
    mask[7:12, 7:12] = 1  # Echter Defektblock

    # Act
    cleaned = mod.apply_morphological_filters(mask, filter_size=3)

    # Assert
    assert cleaned[2, 2] == 0
    assert cleaned[9, 9] == 1