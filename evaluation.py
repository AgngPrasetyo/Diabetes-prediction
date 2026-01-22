"""
Fungsi untuk evaluasi metrik klasifikasi.
"""
from typing import Dict
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def get_classification_metrics(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """
    Menghitung metrik klasifikasi utama dari probabilitas.

    Args:
        y_true: Nilai target sebenarnya.
        y_proba: Probabilitas prediksi dari kelas positif.
        threshold: Ambang batas untuk mengubah probabilitas menjadi kelas biner.
    """
    y_pred = (y_proba >= threshold).astype(int)
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }
    return metrics