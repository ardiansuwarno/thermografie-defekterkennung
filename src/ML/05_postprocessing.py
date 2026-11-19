import numpy as np
from scipy.ndimage import binary_closing, binary_opening


def apply_morphological_filters(mask, filter_size=3):
    """
    Wendet morphologische Filter (Opening und Closing) auf eine binäre Maske an.

    Ablauf:
    1. Opening (Erosion -> Dilatation): Löscht kleine, isolierte Störpixel im Hintergrund.
    2. Closing (Dilatation -> Erosion): Schließt kleine Lücken innerhalb des Hauptdefekts.

    Args:
        mask (np.ndarray): Die vorhergesagte 2D-Defektmaske.
        filter_size (int): Die Größe der Nachbarschaft (z. B. 3x3 Pixel).

    Returns:
        np.ndarray: Die bereinigte Maske.
    """
    # Definiere den "Pinsel" (Strukturelement), mit dem wir über das Bild fahren
    structure = np.ones((filter_size, filter_size), dtype=int)

    # 1. Opening gegen das Rauschen am Bildrand (False Positives)
    mask_opened = binary_opening(mask, structure=structure)

    # 2. Closing gegen kleine Löcher und unsaubere Ränder im Defekt
    mask_cleaned = binary_closing(mask_opened, structure=structure)

    return mask_cleaned.astype(int)
