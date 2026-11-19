import numpy as np
import pywt

def extract_wavelet_features(data_cube, wavelet_name='haar', level=4):
    """
    Wendet die Diskrete Wavelet-Transformation (DWT) auf die Zeitachse jedes Pixels an
    und formt die Daten für Machine-Learning-Algorithmen um.
    
    Args:
        data_cube (np.ndarray): Vorverarbeiteter Bildwürfel, Shape (Y, X, Zeit).
        wavelet_name (str): Name des Wavelets (z. B. 'haar', 'db4').
        level (int): Zerlegungsstufe (Je höher, desto stärker die Kompression).
        
    Returns:
        feature_matrix (np.ndarray): 2D-Matrix für ML-Modelle, Shape (Y*X, Anzahl_Features).
        original_shape (tuple): Ursprüngliche räumliche Shape (Y, X), 
                                wichtig für die spätere Rückwandlung ins Bild.
    """
    y_dim, x_dim, time_steps = data_cube.shape
    
    # 1. DWT entlang der Zeitachse (axis=-1) anwenden
    # Gibt eine Liste von Koeffizienten zurück: [cA_n, cD_n, cD_{n-1}, ..., cD_1]
    # cA = Approximation (Trend), cD = Details (schnelle Änderungen)
    coeffs = pywt.wavedec(data_cube, wavelet=wavelet_name, level=level, axis=-1)
    
    # Um die Dimensionen drastisch zu reduzieren, nehmen wir nur die 
    # Approximations-Koeffizienten und die letzte Detail-Stufe der höchsten Zerlegungsebene.
    cA_top = coeffs[0]  # Shape: (Y, X, reduzierte_zeit)
    cD_top = coeffs[1]  # Shape: (Y, X, reduzierte_zeit)
    
    # Koeffizienten entlang der Feature-Achse (letzte Achse) verketten
    combined_features = np.concatenate((cA_top, cD_top), axis=-1)
    num_features = combined_features.shape[-1]
    
    # 2. Flattening: Umformen in eine 2D-Tabelle für Machine Learning
    # Aus (100, 100, num_features) wird (10000, num_features)
    feature_matrix = combined_features.reshape(y_dim * x_dim, num_features)
    
    return feature_matrix, (y_dim, x_dim)

# --- Test-Block zur direkten Überprüfung ---
if __name__ == "__main__":
    # Simuliere einen bereits vorverarbeiteten (z.B. standardisierten) Bildwürfel
    dummy_cube = np.random.randn(100, 100, 1000)
    
    try:
        print(f"Originale Shape des Bildwürfels: {dummy_cube.shape}")
        
        # Merkmale extrahieren
        features, spatial_shape = extract_wavelet_features(dummy_cube, wavelet_name='haar', level=4)
        
        print("\nErfolgreich Wavelet-Features extrahiert!")
        print(f"Shape der Feature-Matrix für ML: {features.shape} (Pixel, Features)")
        print(f"Ursprüngliche Bild-Dimensionen:  {spatial_shape}")
        
    except Exception as e:
        print(f"\033[91mFehler bei der Feature-Extraktion: {e}\033[0m")