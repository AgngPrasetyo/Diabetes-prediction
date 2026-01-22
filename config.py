"""
File konfigurasi terpusat untuk proyek deteksi diabetes.
"""
from pathlib import Path

# Path utama proyek
ROOT_DIR = Path(__file__).parent.resolve()

# Path ke folder-folder penting
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "best models"
REPORT_DIR = ROOT_DIR / "reports"

# ==========================================================
# --- TAMBAHAN UNTUK MEMBUAT FOLDER SECARA OTOMATIS ---
# Baris-baris ini akan membuat folder jika belum ada.
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================

# Konfigurasi data
# Dataset hasil preprocessing
DATA_PATH   = DATA_DIR / "diabetes_dataset.csv"
DATA_BALANCED_CLEAN_PATH   = DATA_DIR / "diabetes_clean_smote_dataset.csv"
DATA_NONSMOTE_CLEAN_PATH   = DATA_DIR / "diabetes_clean_non_smote_dataset.csv"



TARGET_COL = "Diabetes_binary"
TEST_SIZE = 0.2
SEED = 42
