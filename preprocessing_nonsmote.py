import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from collections import Counter
import sys
# Pastikan file config.py ada di folder yang sama
from config import DATA_PATH 

# 1. LOAD DATA
# ==============================================================================
print("--- [1] Memuat Dataset ---")
try:
    df = pd.read_csv(DATA_PATH, sep=',')
    print(f"Data awal dimuat: {df.shape}")
except FileNotFoundError:
    print(f"File tidak ditemukan di path: {DATA_PATH}")
    sys.exit()

# 2. FEATURE SELECTION & RENAMING
# ==============================================================================
print("\n--- [2] Memilih Fitur & Rename ---")

column_mapping = {
    'Diabetes_binary': 'Diabetes', # Target
    'HighBP': 'HighBP',
    'HighChol': 'HighChol',
    'BMI': 'BMI',
    'Smoker': 'Smoker',
    'Stroke': 'Stroke',
    'HeartDiseaseorAttack': 'HeartDisease',
    'PhysActivity': 'PhysActivity',
    'HvyAlcoholConsump': 'Alcohol',
    'GenHlth': 'GenHealth',
    'MentHlth': 'MentalHealth',
    'PhysHlth': 'PhysicalHealth',
    'Sex': 'Sex',
    'Age': 'Age'
}

available_cols = [c for c in column_mapping.keys() if c in df.columns]
df = df[available_cols].rename(columns=column_mapping)

print(f"Kolom terpilih: {list(df.columns)}")

# 3. DATA CLEANING (DUPLICATES & NULL)
# ==============================================================================
print("\n--- [3] Membersihkan Data (Duplikat & Null) ---")
initial_rows = df.shape[0]

# Hapus Duplikat
df.drop_duplicates(inplace=True)

# Hapus Null
df.dropna(inplace=True)

print(f"Baris dihapus: {initial_rows - df.shape[0]}")
print(f"Ukuran data setelah cleaning: {df.shape}")

# 4. OUTLIER HANDLING (IQR Method)
# ==============================================================================
print("\n--- [4] Menangani Outlier (Metode IQR) ---")
cols_to_check_outlier = ['BMI']

for col in cols_to_check_outlier:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    old_size = df.shape[0]
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    print(f"Outlier dihapus pada kolom {col}: {old_size - df.shape[0]} baris")

# 5. FEATURE ENGINEERING
# ==============================================================================
print("\n--- [5] Feature Engineering ---")

df['BMI_HighBP_Interaction'] = df['BMI'] * df['HighBP']
df['Age_GenHealth_Interaction'] = df['Age'] * df['GenHealth']

print("Fitur baru ditambahkan: 'BMI_HighBP_Interaction', 'Age_GenHealth_Interaction'")

# 6. STANDARDIZATION (Scaling)
# ==============================================================================
print("\n--- [6] Standarisasi Data ---")

numeric_features = [
    'BMI', 'GenHealth', 'MentalHealth', 'PhysicalHealth', 'Age', 
    'BMI_HighBP_Interaction', 'Age_GenHealth_Interaction'
]

# Menggunakan StandardScaler (Z-score scaling)
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

print("Fitur numerik telah distandarisasi (Mean=0, Std=1).")

# 7. MENYIAPKAN FINAL DATAFRAME (TANPA SMOTE)
# ==============================================================================
print("\n--- [7] Finalisasi Data (Tanpa SMOTE) ---")

# Karena kita TIDAK melakukan SMOTE, kita gunakan df yang ada
df_clean = df.copy()

# Memastikan kolom binary bertipe integer (bukan float 1.0/0.0)
binary_cols = ['Diabetes', 'HighBP', 'HighChol', 'Smoker', 'Stroke', 'HeartDisease', 
               'PhysActivity', 'Alcohol', 'Sex']

for col in binary_cols:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].astype(int)

# Cek distribusi akhir
print("Distribusi Target (Imbalanced):")
print(df_clean['Diabetes'].value_counts())

# 8. MENYIMPAN DATA
# ==============================================================================
print("\n--- [8] Menyimpan Data ---")
output_filename = 'diabetes_clean_non_smote_dataset.csv'

# Simpan ke folder yang sama dengan script atau sesuai path yang diinginkan
# Jika ingin menggunakan path dari config, gunakan: DATA_DIR / output_filename
df_clean.to_csv(output_filename, index=False)

print(f"File berhasil disimpan sebagai: {output_filename}")

# Optional: Visualisasi Distribusi Akhir
plt.figure(figsize=(6,4))
sns.countplot(x='Diabetes', data=df_clean)
plt.title('Distribusi Kelas Target (Tanpa SMOTE)')
plt.savefig('distribution_no_smote.png')
print("Grafik distribusi disimpan sebagai 'distribution_no_smote.png'")

print("\n=== Preprocessing Selesai ===")