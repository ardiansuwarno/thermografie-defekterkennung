import os
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.harmonization import harmonize_time_series
from src.log_scaling import load_and_scale_data
from src.normalization import z_standardize
from src.wavelet_features import extract_wavelet_features


def prepare_data(file_paths, target_length=1000):
    """
    Lädt mehrere Dateien, verarbeitet sie durch die Pipeline und kombiniert sie.
    """
    all_features = []
    all_labels = []

    for file_path in file_paths:
        print(f"Verarbeite: {file_path.name}...")

        # 1. Laden und Log-Skalieren
        log_cube, ground_truth, time_vec = load_and_scale_data(file_path)

        # 2. Zeitreihen harmonisieren
        harmonized_cube = harmonize_time_series(log_cube, target_length=target_length)

        # 3. Z-Standardisierung
        standardized_cube = z_standardize(harmonized_cube)

        # 4. Wavelet-Features extrahieren (gibt bereits eine flache 2D-Matrix zurück)
        features, spatial_shape = extract_wavelet_features(
            standardized_cube, wavelet_name="haar", level=4
        )

        # 5. Labels (Ground Truth) für das ML-Modell abflachen (Flattening)
        # Aus (100, 100) wird ein 1D-Array mit 10.000 Einträgen
        labels_flat = ground_truth.flatten()

        # Binäre Klassifikation erzwingen: Alles größer 0 wird als Defekt (1) markiert
        labels_flat = np.where(labels_flat > 0, 1, 0)

        all_features.append(features)
        all_labels.append(labels_flat)

    # Listen zu großen Matrizen zusammensetzen
    X_combined = np.vstack(all_features)
    y_combined = np.concatenate(all_labels)

    return X_combined, y_combined


if __name__ == "__main__":
    try:
        # 1. Pfade definieren
        base_data_path = Path("KIS2_Data")
        output_dir = Path("src/ML/data")

        # Ordnerstruktur für saubere Speicherung anlegen
        train_dir = output_dir / "train"
        eval_dir = output_dir / "eval"
        train_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)

        # 2. Dateien auswählen (3 Defekt- und 3 Intakt-Dateien für einen robusten Datensatz)
        files_to_process = [
            base_data_path / "defect" / "Simulation_Data10004.hdf5",
            base_data_path / "defect" / "Simulation_Data10005.hdf5",
            base_data_path / "defect" / "Simulation_Data10009.hdf5",
            base_data_path / "no-defect" / "Simulation_Data1000.hdf5",
            base_data_path / "no-defect" / "Simulation_Data1001.hdf5",
            base_data_path / "no-defect" / "Simulation_Data10006.hdf5",
        ]

        print("Starte Datenvorbereitung...\n")

        # 3. Pipeline durchlaufen
        X, y = prepare_data(files_to_process)

        print(f"\nGesamtdatensatz erstellt: X={X.shape}, y={y.shape}")

        # 4. Train-Test-Split (80% Training, 20% Evaluierung)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 5. Speichern der Arrays
        np.save(train_dir / "X_train.npy", X_train)
        np.save(train_dir / "y_train.npy", y_train)
        np.save(eval_dir / "X_test.npy", X_test)
        np.save(eval_dir / "y_test.npy", y_test)

        print(f"\nErfolgreich gespeichert in {output_dir.absolute()}!")
        print(f"Trainingsdaten: {X_train.shape[0]} Pixel")
        print(f"Testdaten:      {X_test.shape[0]} Pixel")

    except Exception as e:
        # Strikte rote Konsolenausgabe bei Fehlern
        print(f"\033[91mFehler bei der Datenvorbereitung:\n{e}\033[0m")
