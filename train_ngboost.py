# ============================================================
# NGBoost Training + Testing
# ALIGNED WITH PREPROCESSING (SMOTEENN + SCALING + FE)
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
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split
)
from sklearn.metrics import (
    classification_report,
    roc_curve,
    confusion_matrix
)
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import uniform, randint as sp_randint

from ngboost import NGBClassifier
from ngboost.distns import Bernoulli

from config import (
    MODEL_DIR,
    DATA_DIR,
    SEED,
    DATA_BALANCED_CLEAN_PATH 
)

from evaluation import get_classification_metrics
from utils import timer


# ============================================================
# CONFIG
# ============================================================
DATA_DIR  = "diabetes_clean_smote_dataset.csv"
TARGET_COL = "Diabetes"

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iter", type=int, default=20)
    parser.add_argument("--cv_folds", type=int, default=3)
    parser.add_argument("--output_name",type=str,default="ngboost_non_weighted")
    return parser.parse_args()


# ============================================================
# PIPELINE
# ============================================================
@timer
def run_pipeline(n_iter, cv_folds, output_name):

    # =========================
    # 1. LOAD DATA
    # =========================
    print(f"[INFO] Load preprocessed data: {DATA_BALANCED_CLEAN_PATH }")
    df = pd.read_csv(DATA_BALANCED_CLEAN_PATH, sep=',' )

    X = df[FEATURE_COLS]
    y = df[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=SEED
    )

    print(f"[INFO] Train size : {X_train.shape}")
    print(f"[INFO] Test  size : {X_test.shape}")

    # =========================
    # 2. MODEL & SEARCH SPACE
    # =========================
    base_model = DecisionTreeRegressor(
        max_depth=4,
        min_samples_leaf=50,
        min_samples_split=100,
        random_state=SEED
    )

    param_dist = {
        "n_estimators": sp_randint(1500, 3000),
        "learning_rate": uniform(0.01, 0.02),
        "col_sample": uniform(0.7, 0.3),
        "minibatch_frac": uniform(0.6, 0.4),
    }

    ngb = NGBClassifier(
        Dist=Bernoulli,
        Base=base_model,
        natural_gradient=True,
        random_state=SEED,
        verbose=False
    )

    search = RandomizedSearchCV(
        estimator=ngb,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=SEED
        ),
        n_jobs=-1,
        verbose=2
    )

    # =========================
    # 3. TRAINING
    # =========================
    print("\n[PHASE 1] Training NGBoost")
    search.fit(X_train, y_train)

    best_params = search.best_params_
    print("\n[INFO] Best params:")
    print(best_params)

    final_model = NGBClassifier(
        Dist=Bernoulli,
        Base=base_model,
        natural_gradient=True,
        random_state=SEED,
        verbose=False,
        **best_params
    )

    final_model.fit(X_train, y_train)

    # =========================
    # 4. TESTING
    # =========================
    print("\n[PHASE 2] Testing")

    proba_test = final_model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, proba_test)
    best_idx = (tpr - fpr).argmax()
    best_thresh = thresholds[best_idx]

    y_pred = (proba_test >= best_thresh).astype(int)

    metrics = get_classification_metrics(
        y_test.values,
        proba_test,
        threshold=best_thresh
    )

    cls_report_dict = classification_report(
        y_test, y_pred, output_dict=True
    )

    conf_matrix = confusion_matrix(y_test, y_pred)

    # =========================
    # 5. OUTPUT
    # =========================
    print("\n=== FINAL METRICS ===")
    print(pd.Series(metrics))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("\nConfusion Matrix:")
    print(conf_matrix)

    # =========================
    # 6. SAVE
    # =========================
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = MODEL_DIR / f"{output_name}.joblib"
    metric_path = MODEL_DIR / f"{output_name}_metrics.json"

    joblib.dump(final_model, model_path)

    with open(metric_path, "w") as f:
        json.dump({
            "model_type": "ngboost_smoteenn_fe_scaled",
            "best_params": best_params,
            "best_threshold": float(best_thresh),
            "metrics_test": metrics,
            "classification_report": cls_report_dict,
            "confusion_matrix": {
                "tn": int(conf_matrix[0, 0]),
                "fp": int(conf_matrix[0, 1]),
                "fn": int(conf_matrix[1, 0]),
                "tp": int(conf_matrix[1, 1]),
            },
            "features": FEATURE_COLS,
            "data_source": "preprocessed_smoteenn"
        }, f, indent=4)

    print(f"\n[INFO] Model saved   : {model_path}")
    print(f"[INFO] Metrics saved : {metric_path}")


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        args.n_iter,
        args.cv_folds,
        args.output_name
    )
