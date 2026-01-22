import joblib
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score, precision_score,
    recall_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)
from config import MODEL_DIR, DATA_DIR, TARGET_COL, REPORT_DIR, DATA_BALANCED_CLEAN_PATH 

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

# Load data
print(f"[INFO] Load DATA (balanced): {DATA_BALANCED_CLEAN_PATH }")
df = pd.read_csv(DATA_BALANCED_CLEAN_PATH , sep=',')

X = df[FEATURE_COLS]
y = df[TARGET_COL].astype(int)

# Load models
model1 = joblib.load(MODEL_DIR / "ngboost_non_weighted.joblib")
model2 = joblib.load(MODEL_DIR / "ngboost_weighted.joblib")

# Predict probabilities
proba1 = model1.predict_proba(X)[:, 1]
proba2 = model2.predict_proba(X)[:, 1]

# Weighted soft voting
w1, w2 = 0.7, 0.3
proba_ensemble = (w1 * proba1 + w2 * proba2) / (w1 + w2)

# Cari threshold terbaik berdasarkan F1
thresholds = np.linspace(0.1, 0.99, 300)
best_f1 = 0
best_thresh = 0.5
candidates = []

for t in thresholds:
    pred = (proba_ensemble >= t).astype(int)
    precision = precision_score(y, pred)
    recall = recall_score(y, pred)
    f1 = f1_score(y, pred)

    if recall >= 0.80 and precision >= 0.80:
        candidates.append((t, recall, precision, f1))

    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

# Pilih threshold terbaik dari kandidat (jika ada)
if candidates:
    df_result = pd.DataFrame(candidates, columns=["threshold", "recall", "precision", "f1"])
    df_result = df_result.sort_values(by="f1", ascending=False)
    final_thresh = df_result.iloc[0]["threshold"]
else:
    final_thresh = best_thresh

# Final prediction
y_pred = (proba_ensemble >= final_thresh).astype(int)

# Metrik akhir
auc = roc_auc_score(y, proba_ensemble)
report = classification_report(y, y_pred, output_dict=True)
conf = confusion_matrix(y, y_pred)

# Simpan confusion matrix plot
conf_matrix_path = REPORT_DIR / "ensemble_ngboost_confusion_matrix.png"
disp = ConfusionMatrixDisplay(confusion_matrix=conf)
disp.plot()
plt.title(f"Confusion Matrix (Threshold = {final_thresh:.2f})")
plt.savefig(conf_matrix_path, dpi=300)
plt.show()

# Plot ROC AUC
roc_plot_path = REPORT_DIR / "ensemble_ngboost_roc_curve.png"
roc_disp = RocCurveDisplay.from_predictions(y, proba_ensemble)
roc_disp.ax_.set_title("ROC Curve - Ensemble NGBoost")
plt.tight_layout()
plt.savefig(roc_plot_path, dpi=300)
plt.show()

# Simpan metrik ke JSON
metrics = {
    "threshold": float(final_thresh),
    "roc_auc": float(auc),
    "f1_score": f1_score(y, y_pred),
    "precision": precision_score(y, y_pred),
    "recall": recall_score(y, y_pred),
    "classification_report": report,
    "confusion_matrix_path": str(conf_matrix_path),
    "roc_curve_path": str(roc_plot_path)
}

with open(REPORT_DIR / "ensemble_ngboost_dual_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Simpan model ensemble ke joblib
ensemble_model_path = MODEL_DIR / "ngboost_ensemble_model.joblib"
ensemble_model = {
    "model1": model1,
    "model2": model2,
    "weights": (w1, w2),
    "threshold": final_thresh
}
joblib.dump(ensemble_model, ensemble_model_path)

# Tampilkan hasil ke terminal
print("\n Ensemble NGBoost Evaluation Results")
print(f"Best Threshold: {metrics['threshold']:.3f}")
print(f"ROC AUC      : {metrics['roc_auc']:.4f}")
print(f"F1 Score     : {metrics['f1_score']:.4f}")
print(f"Precision    : {metrics['precision']:.4f}")
print(f"Recall       : {metrics['recall']:.4f}")
print("\nClassification Report:")
print(classification_report(y, y_pred, digits=4))