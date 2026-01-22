import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, RocCurveDisplay,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)
from config import MODEL_DIR, DATA_DIR, REPORT_DIR

# Load model dan data
model = joblib.load(MODEL_DIR / "ngboost_ensemble_model.joblib")  # atau model_not_weight
X_test = joblib.load(DATA_DIR / "X_test.joblib")
y_test = joblib.load(DATA_DIR / "y_test.joblib")

# Prediksi
proba = model.predict_proba(X_test)[:, 1]
threshold =  0.385752508361204 # ubah jika punya threshold optimal
y_pred = (proba >= threshold).astype(int)

# ROC Curve
roc_auc = roc_auc_score(y_test, proba)
RocCurveDisplay.from_predictions(y_test, proba)
plt.title(f"ROC Curve - NGBoost (AUC = {roc_auc:.4f})")
plt.savefig(REPORT_DIR / "roc_curve_weight_ngboost.png", dpi=300)
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.title("Confusion Matrix - NGBoost")
plt.savefig(REPORT_DIR / "conf_matrix_weight_ngboost.png", dpi=300)
plt.show()

# Tampilkan laporan klasifikasi
print("Classification Report:\n")
print(classification_report(y_test, y_pred, digits=4))
print(f"AUC: {roc_auc:.4f}")
