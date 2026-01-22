# ============================================================
# NGBoost Training + Testing
# NON-SMOTE DATA + APSO CLASS WEIGHT
# ============================================================

from sklearnex import patch_sklearn
patch_sklearn()

import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split
)
from sklearn.metrics import (
    classification_report,
    roc_curve,
    confusion_matrix,
    f1_score
)
from sklearn.tree import DecisionTreeRegressor

from ngboost import NGBClassifier
from ngboost.distns import Bernoulli

from config import MODEL_DIR, SEED
from evaluation import get_classification_metrics
from utils import timer


# ============================================================
# CONFIG
# ============================================================
DATA_PATH = "diabetes_clean_non_smote_dataset.csv"
TARGET_COL = "Diabetes"

FEATURE_COLS = [
    "HighBP", "HighChol", "Smoker", "Stroke", "HeartDisease",
    "PhysActivity", "Alcohol", "Sex", "BMI", "GenHealth",
    "MentalHealth", "PhysicalHealth", "Age",
    "BMI_HighBP_Interaction", "Age_GenHealth_Interaction"
]


# ============================================================
# APSO PARAMETERS
# ============================================================
N_PARTICLES = 12
N_ITER_APSO = 20

W_NEG_RANGE = (0.5, 2.0)
W_POS_RANGE = (1.0, 6.0)


# ============================================================
# APSO UTILITIES
# ============================================================
def clip(x, bounds):
    return np.minimum(np.maximum(x, bounds[:, 0]), bounds[:, 1])


def build_sample_weight(y, w_neg, w_pos):
    return np.where(y == 0, w_neg, w_pos)


def fitness_function(X, y, w_neg, w_pos):
    """F1-score CV"""
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    scores = []

    for tr_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        sample_weight = build_sample_weight(y_tr.values, w_neg, w_pos)

        model = NGBClassifier(
            Dist=Bernoulli,
            Base=DecisionTreeRegressor(
                max_depth=4,
                min_samples_leaf=50,
                min_samples_split=100,
                random_state=SEED
            ),
            n_estimators=1200,
            learning_rate=0.03,
            natural_gradient=True,
            random_state=SEED,
            verbose=False
        )

        model.fit(X_tr, y_tr, sample_weight=sample_weight)
        y_pred = model.predict(X_val)
        scores.append(f1_score(y_val, y_pred))

    return np.mean(scores)


# ============================================================
# APSO OPTIMIZATION
# ============================================================
def run_apso(X, y):
    bounds = np.array([W_NEG_RANGE, W_POS_RANGE])

    pos = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(N_PARTICLES, 2))
    vel = np.zeros_like(pos)

    pbest = pos.copy()
    pbest_score = np.array([
        fitness_function(X, y, p[0], p[1]) for p in pos
    ])

    gbest_idx = np.argmax(pbest_score)
    gbest = pbest[gbest_idx].copy()
    gbest_score = pbest_score[gbest_idx]

    for t in range(N_ITER_APSO):
        w = 0.9 - t * (0.5 / N_ITER_APSO)
        c1, c2 = 2.0, 2.0

        for i in range(N_PARTICLES):
            r1, r2 = np.random.rand(2)

            vel[i] = (
                w * vel[i]
                + c1 * r1 * (pbest[i] - pos[i])
                + c2 * r2 * (gbest - pos[i])
            )

            pos[i] = clip(pos[i] + vel[i], bounds)

            score = fitness_function(X, y, pos[i][0], pos[i][1])

            if score > pbest_score[i]:
                pbest[i] = pos[i]
                pbest_score[i] = score

                if score > gbest_score:
                    gbest = pos[i].copy()
                    gbest_score = score

        print(
            f"[APSO] Iter {t+1}/{N_ITER_APSO} | "
            f"Best F1={gbest_score:.4f} | "
            f"w_neg={gbest[0]:.3f}, w_pos={gbest[1]:.3f}"
        )

    return gbest


# ============================================================
# PIPELINE
# ============================================================
@timer
def run_pipeline(output_name):

    print(f"[INFO] Load NON-SMOTE data: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    print("[PHASE 1] APSO optimize class weight")
    best_w_neg, best_w_pos = run_apso(X_train, y_train)

    class_weight = {0: float(best_w_neg), 1: float(best_w_pos)}
    sample_weight = build_sample_weight(y_train.values, best_w_neg, best_w_pos)

    print("[INFO] Best class weight:", class_weight)

    print("[PHASE 2] Train final NGBoost")

    final_model = NGBClassifier(
        Dist=Bernoulli,
        Base=DecisionTreeRegressor(
            max_depth=4,
            min_samples_leaf=50,
            min_samples_split=100,
            random_state=SEED
        ),
        n_estimators=1500,
        learning_rate=0.03,
        natural_gradient=True,
        random_state=SEED,
        verbose=False
    )

    final_model.fit(X_train, y_train, sample_weight=sample_weight)

    proba_test = final_model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, proba_test)
    best_thresh = thresholds[(tpr - fpr).argmax()]
    y_pred = (proba_test >= best_thresh).astype(int)

    metrics = get_classification_metrics(y_test.values, proba_test, best_thresh)
    cls_report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = MODEL_DIR / f"{output_name}.joblib"
    metric_path = MODEL_DIR / f"{output_name}_metrics.json"

    joblib.dump(final_model, model_path)

    with open(metric_path, "w") as f:
        json.dump({
            "model_type": "ngboost_apso_weight",
            "class_weight": class_weight,
            "best_threshold": float(best_thresh),
            "metrics_test": metrics,
            "classification_report": cls_report,
            "confusion_matrix": {
                "tn": int(conf_matrix[0, 0]),
                "fp": int(conf_matrix[0, 1]),
                "fn": int(conf_matrix[1, 0]),
                "tp": int(conf_matrix[1, 1]),
            },
            "features": FEATURE_COLS,
            "data_source": "preprocessed_non_smote"
        }, f, indent=4)

    print(f"[INFO] Model saved   : {model_path}")
    print(f"[INFO] Metrics saved : {metric_path}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    run_pipeline("ngboost_weighted")
