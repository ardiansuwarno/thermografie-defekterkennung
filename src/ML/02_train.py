import numpy as np
import joblib
import warnings
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.utils import resample

# Warnungen unterdrücken für eine saubere Konsole
warnings.filterwarnings("ignore")

def train_with_hyperparameter_tuning(X_train, y_train):
    """
    Trainiert eine Logistische Regression mit PCA-Dimensionsreduktion 
    und optimierten Grids für maximale Geschwindigkeit.
    """
    # 1. Pipeline: Skalierung -> PCA (auf 15 Features) -> Klassifikation
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=15, random_state=42)),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    # 2. Getrennte Grids (lbfgs für L2 ist extrem schnell)
    param_grid_l2 = {
        'classifier__penalty': ['l2'],
        'classifier__solver': ['lbfgs'],
        'classifier__C': [0.1, 1, 10],
        'classifier__tol': [1e-3, 1e-4]
    }

    param_grid_l1 = {
        'classifier__penalty': ['l1'],
        'classifier__solver': ['saga'],
        'classifier__C': [0.1, 1, 10],
        'classifier__tol': [1e-3] # Nur eine Toleranz, um Zeit zu sparen
    }

    param_grid_elastic = {
        'classifier__penalty': ['elasticnet'],
        'classifier__solver': ['saga'],
        'classifier__C': [0.1, 1],
        'classifier__l1_ratio': [0.5], # Nur ein mittleres Mischverhältnis
        'classifier__tol': [1e-3]
    }
    
    param_grid = [param_grid_l2, param_grid_l1, param_grid_elastic]
    
    # 3. KFold mit 3 statt 5 Splits beschleunigt zusätzlich um 40%
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    
    print("Starte GridSearch (mit PCA=15 Features und 3-Fold CV)...")
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=kfold,
        scoring='accuracy',
        n_jobs=-1,
        return_train_score=True
    )
    
    grid_search.fit(X_train, y_train)
    
    print("\nTraining abgeschlossen!")
    print(f"Beste Accuracy (Cross-Validation): {grid_search.best_score_ * 100:.2f} %")
    print(f"Beste Hyperparameter: {grid_search.best_params_}")
    
    return grid_search

if __name__ == "__main__":
    try:
        data_dir = Path("src/ML/data/train")
        model_dir = Path("src/ML/saved_models")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        print("Lade Trainingsdaten...")
        X_train_full = np.load(data_dir / "X_train.npy")
        y_train_full = np.load(data_dir / "y_train.npy")
        
        print(f"Volle Trainingsdaten: {X_train_full.shape[0]} Pixel")
        
        # Subsampling auf 10.000 Pixel
        X_train_sub, y_train_sub = resample(
            X_train_full, y_train_full, 
            n_samples=10000, 
            random_state=42, 
            stratify=y_train_full # Behält das Verhältnis von Defekt/Intakt bei
        )
        
        print(f"Reduzierte Trainingsdaten für GridSearch: {X_train_sub.shape[0]} Pixel")
        
        # Training starten (gibt nun das komplette GridSearch-Objekt zurück)
        fitted_grid = train_with_hyperparameter_tuning(X_train_sub, y_train_sub)
        
        # Bestes Modell und CV-Ergebnisse speichern
        best_model = fitted_grid.best_estimator_
        joblib.dump(best_model, model_dir / "best_logistic_regression.pkl")
        
        # Ergebnisse der GridSearchCV für Plotting speichern
        import pandas as pd
        results_df = pd.DataFrame(fitted_grid.cv_results_)
        results_df.to_pickle(model_dir / "cv_results.pkl")
        
        print(f"\nModelle und Resultate erfolgreich in {model_dir.absolute()} gespeichert!")
        
    except Exception as e:
        print(f"\033[91mFehler: {e}\033[0m")