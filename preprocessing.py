import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.combine import SMOTEENN
from collections import Counter
from config import DATA_PATH

# 1. LOAD DATA
# ==============================================================================
print("--- [1] Memuat Dataset ---")
try:
    df = pd.read_csv(DATA_PATH, sep=',')
    print(f"Data awal dimuat: {df.shape}")
except FileNotFoundError:
    print("File 'diabetes_dataset.csv' tidak ditemukan. Pastikan file ada di folder yang sama.")
    exit()

# 2. FEATURE SELECTION & RENAMING
# ==============================================================================
print("\n--- [2] Memilih Fitur & Rename ---")

# Mapping nama kolom dataset asli ke nama yang diinginkan user
# Format: {'NamaAsli': 'NamaBaru'}
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

# Cek apakah kolom ada sebelum rename
available_cols = [c for c in column_mapping.keys() if c in df.columns]
df = df[available_cols].rename(columns=column_mapping)

print(f"Kolom terpilih: {list(df.columns)}")

# 3. DATA CLEANING (DUPLICATES & NULL)
# ==============================================================================
print("\n--- [3] Membersihkan Data (Duplikat & Null) ---")
initial_rows = df.shape[0]

# Hapus Duplikat
df.drop_duplicates(inplace=True)

# Hapus Null (jika ada)
df.dropna(inplace=True)

print(f"Baris dihapus: {initial_rows - df.shape[0]}")
print(f"Ukuran data setelah cleaning: {df.shape}")

# 4. OUTLIER HANDLING (IQR Method)
# ==============================================================================
print("\n--- [4] Menangani Outlier (Metode IQR) ---")
# Kita fokus membuang outlier pada data kontinu yang ekstrem, seperti BMI.
# Untuk MentalHealth/PhysicalHealth (0-30), kita biarkan karena 30 hari sakit adalah valid meski ekstrem.

cols_to_check_outlier = ['BMI']

for col in cols_to_check_outlier:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter data
    old_size = df.shape[0]
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    print(f"Outlier dihapus pada kolom {col}: {old_size - df.shape[0]} baris")

# 5. FEATURE ENGINEERING
# ==============================================================================
print("\n--- [5] Feature Engineering ---")

# Interaksi 1: Risk_Score (Gabungan BMI tinggi dan Tekanan Darah Tinggi)
# Logika: BMI tinggi lebih berbahaya jika disertai darah tinggi
df['BMI_HighBP_Interaction'] = df['BMI'] * df['HighBP']

# Interaksi 2: Age_GenHealth (Kesehatan umum dikalikan faktor Usia)
# Logika: Kesehatan buruk (skala 5) pada usia tua (level 13) memiliki bobot risiko lebih tinggi
df['Age_GenHealth_Interaction'] = df['Age'] * df['GenHealth']

print("Fitur baru ditambahkan: 'BMI_HighBP_Interaction', 'Age_GenHealth_Interaction'")

# 6. STANDARDIZATION (Scaling)
# ==============================================================================
print("\n--- [6] Standarisasi Data ---")
# Kita hanya menstandarisasi fitur numerik/ordinal, bukan binary (0/1).

numeric_features = [
    'BMI', 'GenHealth', 'MentalHealth', 'PhysicalHealth', 'Age', 
    'BMI_HighBP_Interaction', 'Age_GenHealth_Interaction'
]

# Menggunakan StandardScaler (Z-score scaling)
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

print("Fitur numerik telah distandarisasi (Mean=0, Std=1).")

# 7. CEK DISTRIBUSI SEBELUM SMOTEENN
# ==============================================================================
print("\n--- [7] Cek Distribusi Target Awal ---")
target_counts = df['Diabetes'].value_counts()
print(target_counts)
print(f"Rasio Awal (0:1) -> {target_counts[0]}:{target_counts[1]}")

# 8. OVERSAMPLING & CLEANING via SMOTEENN
# ==============================================================================
print("\n--- [8] Balancing Data dengan SMOTEENN ---")

X = df.drop('Diabetes', axis=1)
y = df['Diabetes']

# SMOTEENN menggabungkan oversampling (SMOTE) dan undersampling (Edited Nearest Neighbours)
# untuk membersihkan sampel yang overlapping/berisik setelah di-generate.
smote_enn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smote_enn.fit_resample(X, y)

print(f"Ukuran data setelah SMOTEENN: {X_resampled.shape}")
print(f"Distribusi Target Baru: {Counter(y_resampled)}")

# Menggabungkan kembali menjadi DataFrame
df_clean = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), 
                      pd.Series(y_resampled, name='Diabetes')], axis=1)

# *PENTING*: SMOTE kadang menghasilkan float untuk kolom binary karena interpolasi.
# Kita harus membulatkan kembali kolom binary ke 0 atau 1.
binary_cols = ['HighBP', 'HighChol', 'Smoker', 'Stroke', 'HeartDisease', 
               'PhysActivity', 'Alcohol', 'Sex']

for col in binary_cols:
    df_clean[col] = df_clean[col].round().astype(int)

# 9. MENYIMPAN DATA
# ==============================================================================
print("\n--- [9] Menyimpan Data ---")
output_filename = 'diabetes_clean_smote_dataset.csv'
df_clean.to_csv(output_filename, index=False)
print(f"File berhasil disimpan sebagai: {output_filename}")

# Optional: Visualisasi Distribusi Akhir (Disimpan sebagai gambar)
plt.figure(figsize=(6,4))
sns.countplot(x='Diabetes', data=df_clean)
plt.title('Distribusi Kelas Target Setelah SMOTEENN')
plt.savefig('distribution_after_smoteenn.png')
print("Grafik distribusi disimpan sebagai 'distribution_after_smoteenn.png'")

print("\n=== Preprocessing Selesai ===")