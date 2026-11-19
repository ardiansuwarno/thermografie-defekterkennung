import numpy as np

def z_standardize(data_cube):
    """
    Führt eine Z-Standardisierung auf dem Bildwürfel durch.
    Die Skalierung erfolgt für jeden Pixel individuell entlang der Zeitachse.
    
    Args:
        data_cube (np.ndarray): Der harmonisierte Bildwürfel mit Shape (Y, X, Zeit).
        
    Returns:
        np.ndarray: Der Z-standardisierte Bildwürfel.
    """
    # Mittelwert und Standardabweichung entlang der Zeitachse (axis=2) berechnen.
    # keepdims=True erhält die Dimensionen (Y, X, 1), was das Broadcasting 
    # für die nachfolgende Subtraktion und Division ermöglicht.
    mean_vals = np.mean(data_cube, axis=2, keepdims=True)
    std_vals = np.std(data_cube, axis=2, keepdims=True)
    
    # Ein kleines Epsilon verhindert die Division durch Null, 
    # falls ein Pixel ein komplett flaches Signal (Standardabweichung = 0) aufweist.
    epsilon = 1e-8
    
    # Z-Standardisierung anwenden: (Wert - Mittelwert) / Standardabweichung
    standardized_cube = (data_cube - mean_vals) / (std_vals + epsilon)
    
    return standardized_cube

# --- Test-Block zur direkten Überprüfung ---
if __name__ == "__main__":
    # Wir simulieren einen Bildwürfel (z. B. 10x10 Pixel, 1000 Frames)
    # mit fiktiven Temperaturwerten zwischen 295 und 345
    dummy_data = np.random.rand(10, 10, 1000) * 50 + 295
    
    try:
        standardized_data = z_standardize(dummy_data)
        
        print("Erfolgreich standardisiert!")
        print(f"Original-Mittelwert eines Test-Pixels:  {np.mean(dummy_data[5, 5, :]):.4f}")
        print(f"Standardisierter Mittelwert (sollte ~0 sein): {np.mean(standardized_data[5, 5, :]):.4f}")
        print(f"Standardisierte Std-Abw. (sollte ~1 sein):  {np.std(standardized_data[5, 5, :]):.4f}")
        
    except Exception as e:
        print(f"\033[91mFehler bei der Normalisierung: {e}\033[0m")