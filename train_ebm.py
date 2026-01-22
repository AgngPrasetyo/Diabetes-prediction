# ============================================================
# EBM Explainer
# Training + Evaluation + Local Explainability
# (Clinical Decision Support)
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from interpret.glassbox import ExplainableBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve
)

from config import DATA_DIR, MODEL_DIR, REPORT_DIR, TARGET_COL, DATA_BALANCED_CLEAN_PATH

# ============================================================
# FEATURES
# ============================================================
FEATURE_COLS = [
    "HighBP",
    "HighChol",
    "Smoker",
    "Stroke",
    "HeartDisease",
    "PhysActivity",
    "Alcohol",
    "Sex",
    "BMI",
    "GenHealth",
    "MentalHealth",
    "PhysicalHealth",
    "Age",
    "BMI_HighBP_Interaction",
    "Age_GenHealth_Interaction"
]

EBM_DATA_PATH = DATA_DIR / "diabetes_clean_smote_dataset.csv"
SEED = 42
THRESHOLD = 0.5
TARGET_COL = 'Diabetes'

# ============================================================
# 1. TRAIN + EVALUATE EBM (GLOBAL)
# ============================================================
def train_and_evaluate_ebm():
    print(f"[INFO] Load training data: {EBM_DATA_PATH}")
    df = pd.read_csv(EBM_DATA_PATH)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=SEED
    )

    ebm = ExplainableBoostingClassifier(
        interactions=2,
        learning_rate=0.01,
        max_bins=256,
        random_state=SEED
    )

    print("[INFO] Training EBM model...")
    ebm.fit(X_train, y_train)

    # =========================
    # Evaluation
    # =========================
    print("[INFO] Evaluating EBM model...")
    proba_test = ebm.predict_proba(X_test)[:, 1]
    y_pred = (proba_test >= THRESHOLD).astype(int)

    roc_auc = roc_auc_score(y_test, proba_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"\nROC-AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Save metrics
    metrics_path = REPORT_DIR / "ebm_classification_report.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "roc_auc": float(roc_auc),
            "threshold": THRESHOLD,
            "classification_report": report
        }, f, indent=4)

    print(f"[INFO] Classification report saved: {metrics_path}")

    # Save model
    model_path = MODEL_DIR / "ebm_explainer.joblib"
    joblib.dump(ebm, model_path)
    print(f"[INFO] EBM model saved: {model_path}")

    return ebm


# ============================================================
# 2. LOCAL EXPLAINABILITY (FROM JSON INPUT)
# ============================================================
def explain_patient(json_input_path):

    ebm = joblib.load(MODEL_DIR / "ebm_explainer.joblib")

    with open(json_input_path, "r") as f:
        patient_data = json.load(f)

    X_patient = pd.DataFrame([patient_data], columns=FEATURE_COLS)

    # Local explanation
    explanation = ebm.explain_local(X_patient)

    local_scores = explanation.data(0)["scores"]
    feature_names = explanation.feature_names

    contrib_df = pd.DataFrame({
        "feature": feature_names,
        "contribution": local_scores
    }).sort_values(by="contribution", key=abs, ascending=False)

    # =========================
    # Visual output
    # =========================
    plt.figure(figsize=(10, 6))
    colors = ["red" if v > 0 else "green" for v in contrib_df["contribution"]]
    plt.barh(contrib_df["feature"], contrib_df["contribution"], color=colors)
    plt.xlabel("Contribution to Diabetes Risk (log-odds)")
    plt.title("EBM Local Explanation – Patient Risk Factors")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plot_path = REPORT_DIR / "ebm_local_explanation.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"[INFO] Local explanation plot saved: {plot_path}")

    # =========================
    # JSON output
    # =========================
    output_json = {
        "patient_input": patient_data,
        "feature_contributions": {
            f: float(v) for f, v in zip(feature_names, local_scores)
        },
        "total_risk_score_logit": float(np.sum(local_scores))
    }

    json_path = REPORT_DIR / "ebm_local_explanation.json"
    with open(json_path, "w") as f:
        json.dump(output_json, f, indent=4)

    print(f"[INFO] Local explanation JSON saved: {json_path}")

    return contrib_df


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":

    # Train + evaluate EBM
    train_and_evaluate_ebm()

    # Example local explanation
    explain_patient(
        json_input_path=DATA_DIR / "sample_patient.json"
    )
