# -*- coding: utf-8 -*-
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import MODEL_DIR, DATA_DIR, TARGET_COL, REPORT_DIR
from sklearn.metrics import confusion_matrix
from matplotlib.patches import Patch
import scipy.stats

# === Load model ensemble and data ===
ensemble_model = joblib.load(MODEL_DIR / "ngboost_ensemble_model.joblib")
model1 = ensemble_model["model1"]
model2 = ensemble_model["model2"]
w1, w2 = ensemble_model["weights"]
features = ensemble_model["features"]
threshold = ensemble_model["threshold"]

df = pd.read_csv(DATA_DIR / "selected_features_rfe.csv")
X = df[features]
y = df[TARGET_COL].astype(int)

# === Ensemble Probabilities ===
proba1 = model1.predict_proba(X)[:, 1]
proba2 = model2.predict_proba(X)[:, 1]
proba_ensemble = (w1 * proba1 + w2 * proba2) / (w1 + w2)
y_pred = (proba_ensemble >= threshold).astype(int)

# --- Debug: pastikan panjang array konsisten
print("DEBUG lengths:", len(X), len(proba_ensemble), len(y))

# === 1) Predictive Distribution Table (Mean & Variance) ===
dist1 = model1.pred_dist(X)
dist2 = model2.pred_dist(X)

# dist.probs shape = (n_classes, n_samples), jadi ambil baris ke-1 (kelas=1)
mean1 = dist1.probs[1]  
mean2 = dist2.probs[1]

# Variansi biner: p * (1 - p)
var1 = mean1 * (1 - mean1)
var2 = mean2 * (1 - mean2)

# Ensemble (linear combination)
mean_ens = (w1 * mean1 + w2 * mean2) / (w1 + w2)
var_ens  = (w1 * var1  + w2 * var2 ) / (w1 + w2)

# Debug ulang
print("DEBUG dist lengths:", len(mean_ens), len(var_ens), len(y))

# Sekarang buat DataFrame—semua panjang harus sama
n_samples = len(X)
df_dist = pd.DataFrame({
    "Sample":             np.arange(n_samples),
    "Predicted Mean":     mean_ens,
    "Predicted Variance": var_ens,
    "True Label":         y.values
})

df_dist.to_csv(REPORT_DIR / "predictive_distribution_table.csv", index=False)

# === 2) Predicted Probability Distribution Plot (Gaussian Approx) ===
sample_idx = 10  # bisa diubah bebas
mu    = mean_ens[sample_idx]
sigma = np.sqrt(var_ens[sample_idx])

# gunakan 4*sigma di sekitar mu untuk visual yang lebih natural
x_min = max(0, mu - 4*sigma)
x_max = min(1, mu + 4*sigma)
x_vals = np.linspace(x_min, x_max, 500)
pdf_vals = scipy.stats.norm.pdf(x_vals, loc=mu, scale=sigma)

plt.figure(figsize=(8, 4))
plt.plot(x_vals, pdf_vals, label="Predictive Distribution (Normal Approx.)")
plt.axvline(threshold, color="orange", linestyle="--", label="Threshold")
plt.axvline(y.iloc[sample_idx], color="red", linestyle="-",      label="True Label")
plt.title(f"Predictive Distribution for Sample #{sample_idx}")
plt.xlabel("Probability of Diabetes")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig(REPORT_DIR / "predictive_distribution_sample.png", dpi=300)
plt.show()



# === 3b) Violin Plot of Predictive Means by Class ===
df_violin = pd.DataFrame({
    "Predicted Mean": mean_ens,
    "True Label":      y.map({0: "No Diabetes", 1: "Diabetes"})
})
plt.figure(figsize=(6, 5))
sns.violinplot(data=df_violin, x="True Label", y="Predicted Mean", palette="pastel")
plt.title("Distribution of Predictive Means by True Class")
plt.tight_layout()
plt.savefig(REPORT_DIR / "violin_plot_predictive_distribution.png", dpi=300)
plt.show()

# === 4) Confidence vs Error Scatter Plot ===
confidence = np.abs(proba_ensemble - 0.5)
error      = (y_pred != y).astype(int)

plt.figure(figsize=(6, 5))
sns.scatterplot(x=confidence, y=error, alpha=0.6)
plt.title("Confidence vs Prediction Error")
plt.xlabel("|P(Diabetes) - 0.5| (Confidence)")
plt.ylabel("Error (1 = Incorrect, 0 = Correct)")
plt.tight_layout()
plt.savefig(REPORT_DIR / "confidence_vs_error_scatter.png", dpi=300)
plt.show()
