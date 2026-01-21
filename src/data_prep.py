"""
data_prep.py
-------------
Préparation des données pour un projet bancaire de prédiction du churn.

- Chargement du CSV
- Nettoyage (colonnes inutiles)
- Gestion des valeurs manquantes
- Encodage catégoriel + standardisation numérique via ColumnTransformer
- Split train/test
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# -----------------------------
# Config
# -----------------------------
@dataclass
class PrepConfig:
    base_dir: Path
    data_filename: str = "Churn_Modelling.csv"
    target_candidates: Tuple[str, ...] = ("Exited", "Churn", "churn")
    drop_candidates: Tuple[str, ...] = ("RowNumber", "CustomerId", "Surname")
    test_size: float = 0.2
    random_state: int = 42


# -----------------------------
# Helpers
# -----------------------------
def find_target_column(df: pd.DataFrame, candidates: Tuple[str, ...]) -> str:
    """Trouve la colonne cible (churn) dans le dataset."""
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Aucune colonne cible trouvée parmi {candidates}. Colonnes: {list(df.columns)}")


def load_raw_data(cfg: PrepConfig) -> pd.DataFrame:
    """Charge le dataset brut depuis data/raw/."""
    data_path = cfg.base_dir / "data" / "raw" / cfg.data_filename
    if not data_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {data_path}")

    df = pd.read_csv(data_path)
    return df


def basic_cleaning(df: pd.DataFrame, cfg: PrepConfig) -> pd.DataFrame:
    """Nettoyage léger : suppression des colonnes identifiants / non utiles."""
    df = df.copy()
    drop_cols = [c for c in cfg.drop_candidates if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
    return df


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Construit un preprocessor sklearn:
    - numériques : imputation median + standardisation
    - catégorielles : imputation most_frequent + one-hot
    """
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )

    return preprocessor, num_cols, cat_cols


def prepare_train_test(
    cfg: PrepConfig,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, ColumnTransformer, dict]:
    """
    Pipeline de préparation complète :
    - load + clean
    - target selection
    - split train/test (stratifié)
    - build preprocessor

    Retourne:
      X_train, y_train, X_test, y_test, preprocessor, metadata
    """
    df = load_raw_data(cfg)
    df = basic_cleaning(df, cfg)

    target = find_target_column(df, cfg.target_candidates)

    # Séparer features / target
    X = df.drop(columns=[target])
    y = df[target].astype(int)

    # Split stratifié (important pour churn)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    preprocessor, num_cols, cat_cols = build_preprocessor(X_train)

    metadata = {
        "target": target,
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "train_size": X_train.shape[0],
        "test_size": X_test.shape[0],
        "churn_rate": float(y.mean()),
        "churn_rate_train": float(y_train.mean()),
        "churn_rate_test": float(y_test.mean()),
    }

    return X_train, y_train, X_test, y_test, preprocessor, metadata


# -----------------------------
# CLI (optionnel) : exécution directe
# -----------------------------
if __name__ == "__main__":
    cfg = PrepConfig(base_dir=Path(__file__).resolve().parents[1])
    X_train, y_train, X_test, y_test, preprocessor, meta = prepare_train_test(cfg)

    print("✅ Data prep OK")
    print("Target:", meta["target"])
    print("Rows:", meta["n_rows"], "Cols:", meta["n_cols"])
    print("Train:", meta["train_size"], "Test:", meta["test_size"])
    print("Churn rate (global/train/test):", meta["churn_rate"], meta["churn_rate_train"], meta["churn_rate_test"])
    print("Num cols:", len(meta["num_cols"]), "| Cat cols:", len(meta["cat_cols"]))
