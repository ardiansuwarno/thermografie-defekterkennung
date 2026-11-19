import numpy as np

def harmonize_time_series(data_cube, target_length, padding_value=-99.0):
    """
    Harmonisiert die Zeitachse (3. Dimension) eines Bildwürfels durch 
    Padding (Auffüllen) oder Truncation (Abschneiden).
    
    Args:
        data_cube (np.ndarray): Der logarithmisch skalierte Bildwürfel mit Shape (Y, X, Zeit).
        target_length (int): Die gewünschte, einheitliche Anzahl an Frames.
        padding_value (float): Der Dummy-Wert, mit dem zu kurze Sequenzen aufgefüllt werden.
        
    Returns:
        np.ndarray: Der harmonisierte Bildwürfel mit der Shape (Y, X, target_length).
    """
    current_length = data_cube.shape[2]
    
    # Fall 1: Die Sequenz hat bereits die perfekte Länge
    if current_length == target_length:
        return data_cube
        
    # Fall 2: Truncation (Die Sequenz ist zu lang und wird abgeschnitten)
    elif current_length > target_length:
        return data_cube[:, :, :target_length]
        
    # Fall 3: Padding (Die Sequenz ist zu kurz und wird mit dem padding_value aufgefüllt)
    else:
        # Definieren, an welchen Achsen wie viel hinzugefügt wird:
        # (Y-Achse: 0, X-Achse: 0, Zeitachse: Differenz zur Ziellänge)
        pad_width = ((0, 0), (0, 0), (0, target_length - current_length))
        
        return np.pad(data_cube, pad_width, mode='constant', constant_values=padding_value)

# --- Test-Block zur direkten Überprüfung ---
if __name__ == "__main__":
    # Simulieren zwei Bildwürfel: einen zu kurzen (800 Frames) und einen zu langen (1200 Frames)
    dummy_short = np.random.rand(100, 100, 800)
    dummy_long = np.random.rand(100, 100, 1200)
    
    target = 1000
    
    try:
        harmonized_short = harmonize_time_series(dummy_short, target)
        harmonized_long = harmonize_time_series(dummy_long, target)
        
        print(f"Original kurz: {dummy_short.shape} -> Harmonisiert: {harmonized_short.shape}")
        print(f"Original lang: {dummy_long.shape} -> Harmonisiert: {harmonized_long.shape}")
    except Exception as e:
        # Fehlermeldung strikt in Rot ausgeben
        print(f"\033[91mFehler bei der Harmonisierung: {e}\033[0m")