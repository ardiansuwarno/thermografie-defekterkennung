# Thermografische Defekterkennung mittels Machine Learning

Dieses Projekt implementiert eine vollständige Signalverarbeitungs- und Machine-Learning-Pipeline zur automatisierten Defekterkennung in thermografischen Messdaten (HDF5). Die zeitlichen Abkühlkurven der Bildpixel werden durch Wavelet-Transformationen analysiert und mithilfe einer Logistischen Regression klassifiziert.

## 📂 Projektstruktur

Das Projekt ist modular aufgebaut und trennt die klassische Signalverarbeitung strikt von der Machine-Learning-Pipeline:

```text
TEIL 2-20260817/
├── KIS2_Data/                      # Rohdaten (HDF5-Dateien, Defekt & Intakt)
├── src/
│   ├── flattening.py               # Transformation zwischen 1D-Arrays und 2D-Bildern
│   ├── harmonization.py            # Zeitreihen-Interpolation auf einheitliche Länge
│   ├── log_scaling.py              # Logarithmische Skalierung der Temperaturwerte
│   ├── normalization.py            # Z-Standardisierung der Abkühlkurven
│   ├── wavelet_features.py         # Merkmalsextraktion (Haar-Wavelets)
│   │
│   └── ML/                         # Machine-Learning-Pipeline
│       ├── data/                   # Gespeicherte Trainings- und Test-Arrays (.npy)
│       ├── saved_models/           # Trainierte Modelle und CV-Resultate (.pkl)
│       ├── 01_data_preps.py        # Daten laden, verarbeiten und splitten
│       ├── 02_train.py             # Modell-Training (GridSearch, PCA, Logistische Regression)
│       ├── 03_evaluate.ipynb       # Modell-Evaluierung (Overfitting-Analyse, Lernkurven)
│       ├── 04_deploy.py            # Inferenz-Skript für neue HDF5-Dateien
│       └── 05_postprocessing.py    # Morphologische Filter (Opening/Closing)
│
├── test_segmentation.ipynb         # Finale Visuelle Auswertung und IoU-Berechnung
└── thermografie_analyse.ipynb      # Initiale Datenexploration (Exploratory Data Analysis)
```

## 🛠️ Installation & Setup

Dieses Projekt verwendet eine isolierte Conda-Umgebung, um Konflikte mit anderen Python-Projekten zu vermeiden und eine reproduzierbare Ausführung sicherzustellen.

### 1. Conda-Umgebung erstellen
Erstelle zunächst eine neue Umgebung mit Python 3.10:

```bash
conda create --name kis2_env python=3.10 -y
```

### 2. Umgebung aktivieren
Aktiviere die Umgebung in deinem Terminal:

```bash
conda activate kis2_env
```

### 3. Abhängigkeiten installieren
Installiere alle benötigten Bibliotheken für die Signalverarbeitung, das maschinelle Lernen und die Datenvisualisierung:

```bash
conda install -c conda-forge h5py pandas matplotlib seaborn pywavelets scikit-learn -y
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```


## 🚀 Ausführung der Pipeline

Die Pipeline ist sequenziell aufgebaut. Die Skripte müssen in der folgenden Reihenfolge ausgeführt werden:

### 1. Datenpräparation (01_data_prep.py)

Führe folgenden Befehl im Terminal aus:
```bash
python src/ML/01_data_prep.py
```
Lädt ausgewählte HDF5-Dateien, jagt sie durch die Signalverarbeitungsmodule (Harmonisierung, Z-Standardisierung, Wavelet-Transformation) und speichert die finalen Features als flache NumPy-Matrizen im Ordner data/.

### 2. Modell-Training (02_train.py)

Führe folgenden Befehl im Terminal aus:
```bash
python src/ML/02_train.py
```
Trainiert eine Klassifikations-Pipeline bestehend aus:
- StandardScaler: Skalierung der Features.
- PCA (Principal Component Analysis): Dimensionsreduktion auf die 15 wichtigsten Hauptkomponenten zur Beschleunigung.
- Logistische Regression: Binäre Klassifikation mit automatischem Hyperparameter-Tuning über GridSearchCV (L1, L2, ElasticNet).

Hinweis: Bei Fehlern während des Trainings erfolgt eine rote Konsolenausgabe zur schnellen Fehlerbehebung.

### 3. Evaluierung (03_evaluate.ipynb)

Öffne die Datei src/ML/03_evaluate.ipynb direkt in VS Code oder starte sie über das Terminal:
```bash
jupyter notebook src/ML/03_evaluate.ipynb
```

Ein interaktives Jupyter Notebook zur Analyse des trainierten Modells. Es generiert:
- Overfitting/Underfitting-Graphen basierend auf den GridSearchCV-Resultaten.
- Feature-Importance-Balkendiagramme der PCA-Komponenten.
- Eine Lernkurve zur Bewertung der Datenmenge.

### 4. Einsatz & Post-Processing (test_segmentation.ipynb)

Öffne die Datei test_segmentation.ipynb im Hauptordner in VS Code oder über das Terminal:
```bash
jupyter notebook test_segmentation.ipynb
```

Wendet das austrainierte Modell auf eine völlig ungesehene HDF5-Messung an. Beinhaltet:
- Inferenz: Aufruf von 04_deploy.py.
- Post-Processing: Aufruf von 05_postprocessing.py (Morphologische Filter) zur Beseitigung von False-Positives (Rauschen) und Glättung der Defektkonturen.
- Visualisierung: Darstellung von Ground Truth, Vorhersage und Fehlerkarte (Error Map).

## 📊 Ergebnisse & Erkenntnisse

Das Modell erzielt auf ungesehenen Daten eine sehr robuste Segmentierungsleistung:

IoU-Score (Roh): 81.73%

IoU-Score (mit Post-Processing): 82.53%

Physikalische Interpretation der Fehlerkarte:
Das morphologische Opening entfernt das isolierte Hintergrundrauschen effektiv. Da die zugrundeliegenden HDF5-Messdaten aus einer Simulation stammen, weist der Temperaturgradient an den Defektgrenzen in der Ground Truth eine absolute, scharfe Trennung auf. Dass die verbleibenden Klassifikationsfehler fast ausschließlich direkt an diesen Rändern auftreten, ist primär auf Diskretisierungseffekte zurückzuführen. An den exakten Übergängen zwischen Defekt und intaktem Material entstehen im Pixelraster Mischsignale. Die eingesetzte Signalverarbeitung (insbesondere die Wavelet-Transformation) verarbeitet diese scharfen Sprünge, wodurch das Modell an den direkten Grenzübergängen leichte Unschärfen in der harten binären Zuweisung (0 oder 1) zeigt.