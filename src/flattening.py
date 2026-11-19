import numpy as np

def flatten_for_ml(data_cube):
    """
    Wandelt einen 3D-Feature-Würfel in eine flache 2D-Matrix für das Machine Learning um.
    
    Args:
        data_cube (np.ndarray): 3D-Array mit der Shape (Y, X, Features).
                                Kann auch (Y, X, Zeit) sein.
        
    Returns:
        feature_matrix (np.ndarray): 2D-Matrix mit der Shape (Y * X, Features).
        spatial_shape (tuple): Die originalen (Y, X)-Dimensionen für die spätere Rekonstruktion.
    """
    # Dimensionen auslesen
    y_dim = data_cube.shape[0]
    x_dim = data_cube.shape[1]
    
    # Falls das Array nur 2 Dimensionen hat (z.B. nur ein Feature pro Pixel), 
    if len(data_cube.shape) == 2:
        data_cube = np.expand_dims(data_cube, axis=-1)
        
    num_features = data_cube.shape[2]
    
    # Flattening: (Y, X, Features) -> (Y * X, Features)
    feature_matrix = data_cube.reshape((y_dim * x_dim), num_features)
    
    return feature_matrix, (y_dim, x_dim)

def reconstruct_image(predictions, spatial_shape):
    """
    Wandelt die flachen Modell-Vorhersagen (1D-Array) wieder in ein 2D-Bild um.
    
    Args:
        predictions (np.ndarray): 1D-Array mit den Vorhersagen, Shape (Y * X,).
        spatial_shape (tuple): Das Tupel (Y, X) aus der flatten_for_ml-Funktion.
        
    Returns:
        np.ndarray: Das 2D-Segmentierungsbild mit der Shape (Y, X).
    """
    y_dim, x_dim = spatial_shape
    
    # Überprüfen, ob die Längen zusammenpassen
    if predictions.shape[0] != (y_dim * x_dim):
        raise ValueError(f"Shape-Fehler: Predictions ({predictions.shape[0]}) "
                         f"passen nicht zur spatial_shape ({y_dim * x_dim}).")
        
    # Zurück in ein 2D-Bild formen
    segmentation_image = predictions.reshape(y_dim, x_dim)
    
    return segmentation_image

# --- Test-Block zur direkten Überprüfung ---
if __name__ == "__main__":
    # Simuliere einen Bildwürfel nach der Wavelet-Transformation 
    # (z.B. 100x100 Pixel, 126 extrahierte Wavelet-Features)
    dummy_features = np.random.randn(100, 100, 126)
    
    try:
        # 1. Flattening für das Modell
        flat_matrix, original_shape = flatten_for_ml(dummy_features)
        print(f"1. Flattening erfolgreich:")
        print(f"   Original: {dummy_features.shape}")
        print(f"   Matrix für ML: {flat_matrix.shape} (Pixel, Features)")
        print(f"   Gespeicherte Bildgröße: {original_shape}")
        
        # Simuliere die Vorhersage eines ML-Modells (0 = Intakt, 1 = Defekt)
        # Für 10.000 Pixel gibt das Modell genau 10.000 Labels aus
        dummy_predictions = np.random.randint(0, 2, size=flat_matrix.shape[0])
        
        # 2. Rückwandlung in das Bild
        segmentation_map = reconstruct_image(dummy_predictions, original_shape)
        print(f"\n2. Rekonstruktion erfolgreich:")
        print(f"   Vorhersagen (1D): {dummy_predictions.shape}")
        print(f"   Segmentierungs-Maske (2D): {segmentation_map.shape}")
        
    except Exception as e:
        print(f"\033[91mFehler bei der Umformung: {e}\033[0m")