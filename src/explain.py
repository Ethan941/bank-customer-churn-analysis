"""
explain.py
----------
Explicabilité churn bancaire (banque/assurance).

Objectifs:
- Récupérer les noms des features après preprocessing (num + one-hot)
- Entraîner un RandomForest et fournir:
  1) Importance globale des variables (feature_importances_)
  2) Export CSV dans reports/
  3) (Optionnel) SHAP pour une explication plus avancée

Usage:
  python -m src.explain
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from src.data_prep import PrepConfig, prepare_train_test


def get_feature_names(preprocessor) -> List[str]:
    """
    Récupère les noms des features après ColumnTransformer,
    compatible avec différentes versions de scikit-learn.
    """
    feature_names: List[str] = []

    # Colonnes numériques
    if hasattr(preprocessor, "named_transformers_") and "num" in preprocessor.named_transformers_:
        # Les colonnes d'origine (avant scaler) restent les mêmes noms
        try:
            num_cols = list(preprocessor.transformers[0][2])
        except Exception:
            num_cols = []
        feature_names.extend(num_cols)

    # Colonnes catégorielles one-hot
    if hasattr(preprocessor, "named_transformers_") and "cat" in preprocessor.named_transformers_:
        cat_transformer = preprocessor.named_transformers_["cat"]
        ohe = cat_transformer.named_steps["onehot"]

        try:
            cat_cols = list(preprocessor.transformers[1][2])
        except Exception:
            cat_cols = []

        ohe_features = list(ohe.get_feature_names_out(cat_cols))
        feature_names.extend(ohe_features)

    return feature_names


def train_rf_model(preprocessor, random_state: int = 42) -> Pipeline:
    """Pipeline RandomForest + preprocessor."""
    return Pipeline(
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


def compute_rf_importance(model: Pipeline, feature_names: List[str]) -> pd.DataFrame:
    """Calcule l'importance des variables pour RandomForest."""
    rf = model.named_steps["model"]
    importances = rf.feature_importances_

    imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    imp["importance_pct"] = (imp["importance"] / imp["importance"].sum()) * 100
    return imp


def try_shap_explain(model: Pipeline, X_sample: pd.DataFrame, feature_names: List[str], out_dir: Path) -> None:
    """
    Optionnel: SHAP (si installé).
    Sauvegarde un summary plot (bar) dans reports/figures.
    """
    try:
        import shap  # noqa: F401
    except Exception:
        print("ℹ️ SHAP non installé -> skip. (Optionnel) pip install shap")
        return

    import shap
    import matplotlib.pyplot as plt

    # Transformer X_sample via preprocessor (matrice numérique)
    X_trans = model.named_steps["prep"].transform(X_sample)

    explainer = shap.TreeExplainer(model.named_steps["model"])
    shap_values = explainer.shap_values(X_trans)

    # binaire: on prend classe 1 (churn)
    if isinstance(shap_values, list):
        shap_vals_class1 = shap_values[1]
    else:
        shap_vals_class1 = shap_values

    shap.summary_plot(
        shap_vals_class1,
        features=X_trans,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    out_path = out_dir / "shap_summary_bar.png"
    plt.savefig(out_path, dpi=160)
    plt.close()

    print("✅ Saved SHAP plot:", out_path)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[1]
    reports_dir = base_dir / "reports"
    figures_dir = reports_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    cfg = PrepConfig(base_dir=base_dir, data_filename="Churn_Modelling.csv")
    X_train, y_train, X_test, y_test, preprocessor, meta = prepare_train_test(cfg)

    # IMPORTANT: on fit le preprocessor via le pipeline (fit du modèle)
    feature_names = None

    model = train_rf_model(preprocessor, random_state=42)
    model.fit(X_train, y_train)

    # Après fit, preprocessor a named_transformers_ disponible
    feature_names = get_feature_names(model.named_steps["prep"])

    # Vérification simple
    if len(feature_names) == 0:
        raise RuntimeError("Impossible de récupérer les noms de features. Vérifie le preprocessor.")

    # Feature importance
    imp = compute_rf_importance(model, feature_names)

    print("✅ Top 15 features (RandomForest importance)")
    print(imp.head(15).to_string(index=False))

    # Export CSV
    out_csv = reports_dir / "rf_feature_importance.csv"
    imp.to_csv(out_csv, index=False)
    print("\n✅ Saved:", out_csv)

    # Optionnel SHAP
    X_sample = X_test.sample(n=min(500, len(X_test)), random_state=42)
    try_shap_explain(model, X_sample, feature_names, figures_dir)

