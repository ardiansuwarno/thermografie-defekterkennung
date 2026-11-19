# Thermografische Defekterkennung mittels Machine Learning

![CI Pipeline](https://github.com/ardiansuwarno/thermografie-defekterkennung/actions/workflows/ci.yml/badge.svg)

Dieses Projekt implementiert eine vollständige Signalverarbeitungs- und Machine-Learning-Pipeline zur automatisierten Defekterkennung in thermografischen Messdaten (HDF5). Die zeitlichen Abkühlkurven der Bildpixel werden durch Wavelet-Transformationen analysiert und mithilfe einer Logistischen Regression klassifiziert.

## 📂 Projektstruktur

Das Projekt ist modular aufgebaut und trennt Signalverarbeitung, Machine-Learning-Pipeline, automatisierte Tests und CI/CD-Konfiguration strikt voneinander:

```text
TEIL 2-20260817/
├── .github/
│   └── workflows/
│       └── ci.yml                       # GitHub Actions CI-Pipeline (Ruff & Pytest)
├── .gitignore
├── pyproject.toml                       # Konfiguration für Ruff und Pytest
├── requirements.txt                     # Python-Abhängigkeiten für CI & Setup
├── README.md
├── KIS2_Data/                           # Rohdaten (HDF5-Dateien, Defekt & Intakt)
├── src/
│   ├── __init__.py
│   ├── flattening.py                    # Transformation zwischen 1D-Arrays und 2D-Bildern
│   ├── harmonization.py                 # Zeitreihen-Harmonisierung (Padding/Truncation)
│   ├── log_scaling.py                   # Logarithmische Skalierung der Temperaturwerte
│   ├── normalization.py                 # Z-Standardisierung der Abkühlkurven
│   ├── wavelet_features.py              # Merkmalsextraktion (Haar-Wavelets)
│   │
│   └── ML/                              # Machine-Learning-Pipeline
│       ├── __init__.py
│       ├── data/                        # Gespeicherte Trainings- und Test-Arrays (.npy, lokal)
│       ├── saved_models/                # Trainierte Modelle und CV-Resultate (.pkl, lokal)
│       ├── 01_data_prep.py              # Daten laden, verarbeiten und splitten
│       ├── 02_train.py                  # Modell-Training (GridSearch, PCA, Logistische Regression)
│       ├── 03_evaluate.ipynb            # Modell-Evaluierung (Overfitting-Analyse, Lernkurven)
│       ├── 04_deploy.py                 # Inferenz-Skript für neue HDF5-Dateien
│       └── 05_postprocessing.py         # Morphologische Filter (Opening/Closing)
│
├── tests/                               # Automatisierte Unit- und Integrationstests (pytest)
│   ├── __init__.py
│   ├── test_flattening.py
│   ├── test_harmonization.py
│   ├── test_log_scaling.py
│   ├── test_normalization.py
│   ├── test_wavelet_features.py
│   └── ml/
│       ├── test_postprocessing.py
│       └── test_pipeline_integration.py
│
├── test_segmentation.ipynb              # Finale visuelle Auswertung und IoU-Berechnung
└── thermografie_analyse.ipynb           # Initiale Datenexploration (Exploratory Data Analysis)
```

## 🛠️ Installation & Setup

Dieses Projekt verwendet eine isolierte Conda-Umgebung, um Konflikte mit anderen Python-Projekten zu vermeiden und eine reproduzierbare Ausführung sicherzustellen.

### 1. Conda-Umgebung erstellen und aktivieren
```bash
conda create --name kis2_env python=3.10 -y
conda activate kis2_env
```

### 2. Abhängigkeiten installieren
Installiere alle benötigten Bibliotheken über `pip` (identisch zur CI-Umgebung):
```bash
pip install -r requirements.txt
pip install pytest pytest-cov ruff
```

Alternativ über Conda (inkl. optionaler PyTorch-Installation):
```bash
conda install -c conda-forge h5py pandas matplotlib seaborn pywavelets scikit-learn -y
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
pip install pytest pytest-cov ruff
```

## 🚀 Ausführung der Pipeline

Die Pipeline ist sequenziell aufgebaut. Führe alle Befehle direkt aus dem Projekt-Hauptordner (`TEIL 2-20260817`) bei aktivierter Umgebung `kis2_env` aus:

### 1. Datenpräparation (`01_data_prep.py`)
```bash
python src/ML/01_data_prep.py
```
Lädt ausgewählte HDF5-Dateien, verarbeitet sie durch die Signalverarbeitungsmodule (Log-Skalierung, Harmonisierung, Z-Standardisierung, Wavelet-Transformation) und speichert die finalen Features als flache NumPy-Matrizen im Ordner `src/ML/data/`.

### 2. Modell-Training (`02_train.py`)
```bash
python src/ML/02_train.py
```
Trainiert eine Klassifikations-Pipeline bestehend aus:
* **StandardScaler:** Skalierung der Features.
* **PCA (Principal Component Analysis):** Dimensionsreduktion auf die 15 wichtigsten Hauptkomponenten zur Beschleunigung.
* **Logistische Regression:** Binäre Klassifikation mit automatischem Hyperparameter-Tuning über `GridSearchCV` (L1, L2, ElasticNet).

*Hinweis: Bei Fehlern während des Trainings erfolgt eine rote Konsolenausgabe zur schnellen Fehlerbehebung.*

### 3. Evaluierung (`03_evaluate.ipynb`)
Öffne `src/ML/03_evaluate.ipynb` direkt in VS Code oder starte das Notebook über das Terminal:
```bash
jupyter notebook src/ML/03_evaluate.ipynb
```
Das Notebook generiert:
* Overfitting/Underfitting-Graphen basierend auf den `GridSearchCV`-Resultaten.
* Feature-Importance-Balkendiagramme der PCA-Komponenten.
* Eine Lernkurve zur Bewertung der Datenmenge.

### 4. Einsatz & Post-Processing (`test_segmentation.ipynb`)
Öffne `test_segmentation.ipynb` im Hauptordner in VS Code oder über das Terminal:
```bash
jupyter notebook test_segmentation.ipynb
```
Wendet das austrainierte Modell auf eine ungesehene HDF5-Messung an. Beinhaltet:
* **Inferenz:** Aufruf von `04_deploy.py`.
* **Post-Processing:** Aufruf von `05_postprocessing.py` (morphologische Filter) zur Beseitigung von False-Positives (Rauschen) und Glättung der Defektkonturen.
* **Visualisierung:** Darstellung von Ground Truth, Vorhersage und Fehlerkarte (Error Map).

## ✅ Tests, Code-Qualität & CI/CD

### Lokale Prüfung (Pytest & Ruff)
Die Unit- und Integrationstests arbeiten mit synthetischen NumPy-Arrays und temporären HDF5-Dateien, sodass sie schnell und unabhängig von lokalen Rohdaten laufen.

```bash
pytest                          # Alle Unit- und Integrationstests ausführen
pytest --cov=src                # Tests inkl. Code-Coverage-Bericht für src/
ruff check .                    # Linting prüfen
ruff check --fix .              # Einfache Lint-Fehler automatisch beheben
ruff format .                   # Code automatisch formatieren
ruff format --check .           # Formatierung (wie im CI) prüfen
```

### Automatisierte CI-Pipeline (GitHub Actions)
Die Workflow-Datei `.github/workflows/ci.yml` wird automatisch bei jedem `push` und `pull_request` auf `main` (sowie manuell via `workflow_dispatch`) auf einem `ubuntu-latest`-Runner ausgeführt und besteht aus zwei parallelen Jobs:
* **Ruff Lint & Format:** Prüft Code-Qualität (`ruff check .`) und einheitliche Formatierung (`ruff format --check .`).
* **Pytest & Coverage:** Installiert alle Abhängigkeiten aus `requirements.txt`, führt `pytest --cov=src` aus und lädt `coverage.xml` sowie `junit.xml` als Build-Artefakte hoch.

## 📊 Ergebnisse & Erkenntnisse

Das Modell erzielt auf ungesehenen Daten eine sehr robuste Segmentierungsleistung:

* **IoU-Score (Roh):** **81.73 %**
* **IoU-Score (mit Post-Processing):** **82.53 %**

**Physikalische Interpretation der Fehlerkarte:**
Das morphologische Opening entfernt das isolierte Hintergrundrauschen effektiv. Da die zugrundeliegenden HDF5-Messdaten aus einer Simulation stammen, weist der Temperaturgradient an den Defektgrenzen in der Ground Truth eine absolute, scharfe Trennung auf. Dass die verbleibenden Klassifikationsfehler fast ausschließlich direkt an diesen Rändern auftreten, ist primär auf Diskretisierungseffekte zurückzuführen. An den exakten Übergängen zwischen Defekt und intaktem Material entstehen im Pixelraster Mischsignale. Die eingesetzte Signalverarbeitung (insbesondere die Wavelet-Transformation) verarbeitet diese scharfen Sprünge, wodurch das Modell an den direkten Grenzübergängen leichte Unschärfen in der harten binären Zuweisung (0 oder 1) zeigt.