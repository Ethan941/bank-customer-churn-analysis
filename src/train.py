"""
train.py
--------
Entraînement de modèles churn bancaire + évaluation "banque-ready" + tuning de threshold.

- Baseline: Logistic Regression (souvent préféré en banque pour interprétabilité)
- Modèle robuste: RandomForest
- Métriques: ROC-AUC, Precision, Recall, F1 + matrice de confusion + report
- Tuning du threshold (ex: viser Recall >= 0.65)
- Export d'un résumé dans reports/
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
)

from src.data_prep import PrepConfig, prepare_train_test


# -----------------------------
# Config
# -----------------------------
@dataclass
class TrainConfig:
    base_dir: Path
    threshold: float = 0.5
    random_state: int = 42
    # Objectif "banque": ne pas rater trop de churners
    min_recall_target: float = 0.65


# -----------------------------
# Metrics helpers
# -----------------------------
def evaluate_binary(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Calcule des métriques clés pour churn."""
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),  # [[TN, FP],[FN, TP]]
        "report": classification_report(y_true, y_pred, zero_division=0),
    }


def find_threshold_for_min_recall(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_recall: float = 0.65,
) -> Optional[Dict[str, float]]:
    """
    Trouve un threshold qui atteint au moins min_recall en maximisant la précision.
    Utile en banque pour piloter la taille de campagne de rétention.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # thresholds a taille -1 vs precision/recall -> on aligne
    thresholds = np.append(thresholds, 1.0)

    candidates = []
    for p, r, t in zip(precision, recall, thresholds):
        if r >= min_recall:
            candidates.append((t, p, r))

    if not candidates:
        return None

    best_t, best_p, best_r = max(candidates, key=lambda x: x[1])  # max precision
    return {"threshold": float(best_t), "precision": float(best_p), "recall": float(best_r)}


# -----------------------------
# Training
# -----------------------------
def build_models(preprocessor, random_state: int) -> Dict[str, Pipeline]:
    """Construit les pipelines modèles."""
    logreg = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )

    rf = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=400,
                random_state=random_state,
                class_weight="balanced",
                n_jobs=-1,
            )),
        ]
    )

    return {"logreg": logreg, "random_forest": rf}


def train_and_evaluate(cfg_prep: PrepConfig, cfg_train: TrainConfig) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Entraîne les modèles, calcule les métriques à threshold=0.5,
    puis calcule un threshold "banque" pour min_recall_target.
    Retourne (résumé dict, dataframe résultats).
    """
    X_train, y_train, X_test, y_test, preprocessor, meta = prepare_train_test(cfg_prep)

    models = build_models(preprocessor, random_state=cfg_train.random_state)

    rows = []
    y_test_np = y_test.to_numpy()

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]

        # métriques au threshold par défaut
        m_default = evaluate_binary(y_test_np, y_proba, threshold=cfg_train.threshold)

        # tuning threshold pour atteindre un recall minimum (si possible)
        tuned = find_threshold_for_min_recall(y_test_np, y_proba, min_recall=cfg_train.min_recall_target)

        row = {
            "model": name,
            "roc_auc": m_default["roc_auc"],
            "precision@0.5": m_default["precision"],
            "recall@0.5": m_default["recall"],
            "f1@0.5": m_default["f1"],
            "cm@0.5": str(m_default["confusion_matrix"]),
            "tuned_threshold_for_recall": tuned["threshold"] if tuned else np.nan,
            "precision@tuned": tuned["precision"] if tuned else np.nan,
            "recall@tuned": tuned["recall"] if tuned else np.nan,
        }
        rows.append(row)

    results_df = pd.DataFrame(rows).sort_values(by=["recall@0.5", "roc_auc"], ascending=False)

    # Choix "best" par défaut (banque): max recall à threshold=0.5, puis roc_auc
    best_model = results_df.iloc[0]["model"]

    summary = {
        **meta,
        "best_model_default": best_model,
        "threshold_default": cfg_train.threshold,
        "min_recall_target": cfg_train.min_recall_target,
    }

    return summary, results_df


def pretty_print_detailed(cfg_prep: PrepConfig, cfg_train: TrainConfig) -> None:
    """Affiche aussi les reports détaillés et matrices de confusion."""
    X_train, y_train, X_test, y_test, preprocessor, meta = prepare_train_test(cfg_prep)
    models = build_models(preprocessor, random_state=cfg_train.random_state)

    y_test_np = y_test.to_numpy()

    print("✅ Training finished")
    print("Target:", meta["target"])
    print("Churn rate (global/train/test):", round(meta["churn_rate"], 4), round(meta["churn_rate_train"], 4), round(meta["churn_rate_test"], 4))
    print()

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]

        m = evaluate_binary(y_test_np, y_proba, threshold=cfg_train.threshold)
        tuned = find_threshold_for_min_recall(y_test_np, y_proba, min_recall=cfg_train.min_recall_target)

        print("—" * 70)
        print(f"Model: {name}")
        print(f"ROC-AUC   : {m['roc_auc']:.4f}")
        print(f"Precision : {m['precision']:.4f}")
        print(f"Recall    : {m['recall']:.4f}")
        print(f"F1        : {m['f1']:.4f}")
        print("Confusion matrix [[TN, FP],[FN, TP]]:")
        print(m["confusion_matrix"])
        print()
        print(m["report"])

        print("Threshold tuning (objectif recall >= {:.2f})".format(cfg_train.min_recall_target))
        print(tuned if tuned else "Aucun threshold ne permet d'atteindre ce recall avec ces proba.")
        print()


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[1]

    cfg_prep = PrepConfig(base_dir=base_dir, data_filename="Churn_Modelling.csv")
    cfg_train = TrainConfig(base_dir=base_dir, threshold=0.5, min_recall_target=0.65)

    # Affichage complet console (comme tu as déjà)
    pretty_print_detailed(cfg_prep, cfg_train)

    # Résumé + export dans reports/
    summary, results_df = train_and_evaluate(cfg_prep, cfg_train)

    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    results_path = reports_dir / "model_results.csv"
    results_df.to_csv(results_path, index=False)

    summary_path = reports_dir / "run_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Bank churn modeling - run summary\n")
        f.write(f"Target: {summary['target']}\n")
        f.write(f"Rows: {summary['n_rows']} | Cols: {summary['n_cols']}\n")
        f.write(f"Churn rate (global/train/test): {summary['churn_rate']:.4f} / {summary['churn_rate_train']:.4f} / {summary['churn_rate_test']:.4f}\n")
        f.write(f"Default threshold: {summary['threshold_default']}\n")
        f.write(f"Min recall target (tuning): {summary['min_recall_target']}\n")
        f.write("\nResults (see model_results.csv)\n")

    print("✅ Saved:", results_path)
    print("✅ Saved:", summary_path)
