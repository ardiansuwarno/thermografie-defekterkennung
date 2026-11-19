import h5py
import numpy as np


def load_and_scale_data(file_path):
    """
    Liest eine HDF5-Datei ein, wandelt die Rohdaten über die LookUpTable
    in absolute Temperaturwerte um und wendet eine logarithmische Skalierung an.

    Returns:
        data_cube_log (np.ndarray): Der logarithmisch skalierte Bildwürfel.
        ground_truth (np.ndarray): Die Defekt-Maske (Labels).
        time_vector (np.ndarray): Der Zeitvektor in Sekunden.
    """
    with h5py.File(file_path, "r") as f:
        # Rohdaten und Metadaten laden
        data_cube_raw = np.array(f["Data"]["Tdata"])
        lut = np.array(f["MetaData"]["LookUpTable"])
        time_vector = np.array(f["MetaData"]["DomainValues"])
        ground_truth = np.array(f["GroundTruth"]["DefectMask"])

    # 1. Rohdaten (uint16) in echte Temperaturwerte (float64) umwandeln
    # Dies ist wichtig, da die physikalischen Werte für den Logarithmus benötigt werden
    data_cube_temp = lut[data_cube_raw]

    # 2. Logarithmische Skalierung anwenden (Linearisierung der Abkühlkurve)
    # Ein winziges Epsilon (1e-8) verhindert den Absturz durch log(0),
    # auch wenn 0 Kelvin physikalisch in diesem Datensatz nicht vorkommen sollte.
    epsilon = 1e-8
    data_cube_log = np.log(data_cube_temp + epsilon)

    return data_cube_log, ground_truth, time_vector


# --- Test-Block zur direkten Überprüfung ---
if __name__ == "__main__":
    # Passe den Pfad an, falls du das Skript aus dem src-Ordner heraus aufrufst
    test_file = "../KIS2_Data/defect/Simulation_Data10004.hdf5"

    try:
        log_data, gt, t_vec = load_and_scale_data(test_file)
        print("Erfolgreich skaliert!")
        print(f"Shape der skalierten Daten: {log_data.shape}")
        print(f"Minimale Log-Temperatur:    {np.min(log_data):.4f}")
        print(f"Maximale Log-Temperatur:    {np.max(log_data):.4f}")
    except FileNotFoundError:
        print(f"Fehler: Die Datei {test_file} wurde nicht gefunden. Prüfe den Pfad!")
