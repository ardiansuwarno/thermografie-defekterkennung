import h5py
import numpy as np

from src.log_scaling import load_and_scale_data


def test_load_and_scale_data_applies_lut_and_logarithm(tmp_path):
    # Arrange: Winzige synthetische HDF5-Datei mit gleicher Struktur wie KIS2_Data
    hdf5_file = tmp_path / "dummy.hdf5"
    lut = np.array([300.0, 310.0, 320.0], dtype=np.float64)
    tdata = np.array([[[0, 1], [1, 2]], [[2, 0], [0, 1]]], dtype=np.uint16)
    mask = np.array([[0, 1], [0, 0]], dtype=np.uint8)
    domain = np.array([0.0, 1.0], dtype=np.float64)

    with h5py.File(hdf5_file, "w") as f:
        f.create_dataset("Data/Tdata", data=tdata)
        f.create_dataset("MetaData/LookUpTable", data=lut)
        f.create_dataset("MetaData/DomainValues", data=domain)
        f.create_dataset("GroundTruth/DefectMask", data=mask)

    # Act
    log_cube, ground_truth, time_vec = load_and_scale_data(hdf5_file)

    # Assert
    assert log_cube.shape == (2, 2, 2)
    assert np.allclose(log_cube[0, 0, 0], np.log(300.0 + 1e-8))
    assert np.array_equal(ground_truth, mask)
    assert np.array_equal(time_vec, domain)