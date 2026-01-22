import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from PIL import Image
from config import MODEL_DIR, DATA_DIR, TARGET_COL, REPORT_DIR

# --- Load model ensemble dan data ---
@st.cache_resource
def load_resources():
    model = joblib.load(MODEL_DIR / "ngboost_ensemble_model.joblib")
    df = pd.read_csv(DATA_DIR / "selected_features_rfe.csv")
    return model, df

ensemble_model, df = load_resources()
X = df[ensemble_model["features"]]
y = df[TARGET_COL]
model1, model2 = ensemble_model["model1"], ensemble_model["model2"]
w1, w2 = ensemble_model["weights"]
thresh = ensemble_model["threshold"]

# Hitung probabilitas ensemble
y_proba1 = model1.predict_proba(X)[:, 1]
y_proba2 = model2.predict_proba(X)[:, 1]
y_proba = (w1 * y_proba1 + w2 * y_proba2) / (w1 + w2)
y_pred = (y_proba >= thresh).astype(int)

st.title("Visualisasi Ensemble NGBoost untuk Deteksi Dini Diabetes")
st.markdown("Model utama: **Ensemble NGBoost**")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Evaluasi Model", "Probabilitas & Ketidakpastian", "Distribusi Ketidakpastian",
    "Korelasi Kesalahan", "Visualisasi ROC & Threshold"
])

# --- Evaluasi Model ---
with tab1:
    st.header("Evaluasi Klasifikasi")
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    st.write(f"Akurasi: **{acc:.4f}**")
    st.text(classification_report(y, y_pred))
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix – Ensemble NGBoost")
    st.pyplot(fig)

# --- Probabilitas & Ketidakpastian ---
with tab2:
    st.header("Distribusi Probabilitas Pasien")
    df_proba = pd.DataFrame({
        "P(No Diabetes)": 1 - y_proba,
        "P(Diabetes)": y_proba
    })
    st.line_chart(df_proba.iloc[:5])

    for i in range(5):
        kelas_pred = y_pred[i]
        kelas_true = y.iloc[i]
        uncert = abs(y_proba[i] - 0.5)
        level = (
            "Tinggi" if uncert < 0.1 else
            "Sedang" if uncert < 0.4 else
            "Rendah"
        )
        benar = kelas_pred == kelas_true
        warna_bg = "#218838" if benar else "#c82333"  # Hijau pekat & Merah pekat
        warna_teks = "#000000"  # Hitam

        st.markdown(
            f"""
            <div style='background-color:{warna_bg}; padding:10px; border-radius:10px;'>
            <h4 style='color:{warna_teks};'>Pasien {i+1} </h4>
            <p style='color:{warna_teks};'><strong>Prediksi:</strong> {'Diabetes' if kelas_pred == 1 else 'Tidak Diabetes'}</p>
            <p style='color:{warna_teks};'><strong>Label Asli:</strong> {'Diabetes' if kelas_true == 1 else 'Tidak Diabetes'}</p>
            <p style='color:{warna_teks};'>- Probabilitas Diabetes: <code>{y_proba[i]:.2%}</code></p>
            <p style='color:{warna_teks};'>- Probabilitas Tidak Diabetes: <code>{1 - y_proba[i]:.2%}</code></p>
            <p style='color:{warna_teks};'><strong>Tingkat Ketidakpastian:</strong> {level}</p>
            </div>
            <br>
            """,
            unsafe_allow_html=True
        )

    # Tabel perbandingan keseluruhan label vs prediksi
    st.subheader("Perbandingan Label Asli vs Prediksi")
    df_compare = pd.DataFrame({
        "Label Asli": y.iloc[:10].map({0: "Tidak Diabetes", 1: "Diabetes"}),
        "Prediksi Model": pd.Series(y_pred[:10]).map({0: "Tidak Diabetes", 1: "Diabetes"}),
        "Probabilitas Diabetes": [f"{p:.2%}" for p in y_proba[:10]]
    })
    df_compare["Benar/Salah"] = (y_pred[:10] == y.iloc[:10]).map(lambda x: "✅ Benar" if x else "❌ Salah")
    st.dataframe(df_compare)

# --- Distribusi Ketidakpastian ---
with tab3:
    st.header("Distribusi Ketidakpastian (Semua Data)")
    uncertainty_scores = np.abs(y_proba - 0.5)
    fig3, ax3 = plt.subplots()
    sns.histplot(uncertainty_scores, bins=20, kde=True, ax=ax3)
    ax3.set_title("Distribusi Ketidakpastian Prediksi")
    ax3.set_xlabel("|P(Diabetes) - 0.5|")
    st.pyplot(fig3)

# --- Korelasi Kesalahan ---
with tab4:
    st.header("Ketidakpastian vs Kesalahan Prediksi")
    errors = (y_pred != y).astype(int)
    fig4, ax4 = plt.subplots()
    sns.boxplot(x=errors, y=uncertainty_scores, ax=ax4)
    ax4.set_xticklabels(["Benar", "Salah"])
    ax4.set_title("Ketidakpastian Berdasarkan Akurasi")
    ax4.set_ylabel("Ketidakpastian")
    st.pyplot(fig4)

# --- ROC & Threshold ---
with tab5:
    st.header("Kurva ROC dan Threshold")
    roc_path = REPORT_DIR / "ensemble_ngboost_roc_curve.png"
    conf_path = REPORT_DIR / "ensemble_ngboost_confusion_matrix.png"

    st.subheader("ROC Curve")
    st.image(Image.open(roc_path))

    st.subheader("Confusion Matrix")
    st.image(Image.open(conf_path))

    st.markdown(f"**Threshold optimal:** `{thresh:.2f}` berdasarkan kombinasi F1/Recall/Precision")