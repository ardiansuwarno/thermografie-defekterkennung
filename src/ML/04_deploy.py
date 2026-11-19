import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt

# Importiere deine Module aus dem Hauptverzeichnis
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.flattening import reconstruct_image
from src.harmonization import harmonize_time_series
from src.log_scaling import load_and_scale_data
from src.normalization import z_standardize
from src.wavelet_features import extract_wavelet_features


def deploy_model_on_file(file_path, model_path, target_length=1000):
    """
    Lädt eine neue HDF5-Datei, führt die gesamte Vorverarbeitung durch
    und erstellt eine 2D-Defektmaske mit dem trainierten Modell.
    """
    print(f"Lade und verarbeite Datei: {file_path.name}")

    # 1. Daten laden
    log_cube, ground_truth, _ = load_and_scale_data(file_path)

    # 2. Harmonisieren und Normalisieren
    harmonized_cube = harmonize_time_series(log_cube, target_length=target_length)
    standardized_cube = z_standardize(harmonized_cube)

    # 3. Features extrahieren (DWT)
    # spatial_shape merkt sich die Dimensionen (z. B. 100x100) für später
    features_flat, spatial_shape = extract_wavelet_features(standardized_cube)

    print("Vorverarbeitung abgeschlossen. Starte KI-Inferenz...")

    # 4. Modell laden und Vorhersage treffen
    # Die geladene Pipeline skaliert die Daten automatisch und wendet die PCA an
    model = joblib.load(model_path)
    predictions_flat = model.predict(features_flat)

    # 5. Flache Vorhersagen zurück in ein 2D-Bild wandeln
    segmentation_mask = reconstruct_image(predictions_flat, spatial_shape)

    return segmentation_mask, ground_truth


if __name__ == "__main__":
    try:
        # Pfade definieren (Wähle eine Datei, die NICHT im Training verwendet wurde)
        base_data_path = Path("KIS2_Data")
        test_file = base_data_path / "defect" / "Simulation_Data10008.hdf5"

        model_path = Path("src/ML/saved_models/best_logistic_regression.pkl")

        if not test_file.exists():
            raise FileNotFoundError(f"Die Testdatei {test_file} existiert nicht.")
        if not model_path.exists():
            raise FileNotFoundError(f"Das Modell {model_path} wurde nicht gefunden.")

        # Modelleinsatz
        predicted_mask, true_mask = deploy_model_on_file(test_file, model_path)

        print("Vorhersage erfolgreich! Erstelle Visualisierung...")

        # --- Visualisierung der Ergebnisse ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # origin='lower' stellt sicher, dass die y-Koordinaten nach oben zeigen
        axes[0].imshow(true_mask, cmap="gray", origin="lower")
        axes[0].set_title("Echte Defektmaske (Ground Truth)")
        axes[0].set_xlabel("X-Pixel")
        axes[0].set_ylabel("Y-Pixel")

        axes[1].imshow(predicted_mask, cmap="inferno", origin="lower")
        axes[1].set_title("KI-Segmentierung (Vorhersage)")
        axes[1].set_xlabel("X-Pixel")
        axes[1].set_ylabel("Y-Pixel")

        plt.suptitle(f"Defekterkennung für {test_file.name}", fontsize=16)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"\033[91mFehler beim Modelleinsatz:\n{e}\033[0m")
