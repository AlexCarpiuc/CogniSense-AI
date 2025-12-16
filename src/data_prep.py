"""
CogniSense AI - Data Preparation Pipeline
Author: Carpiuc Alex
University: Universitatea Babeș-Bolyai
Description:
    Acest script gestioneaza fluxul ETL (Extract, Transform, Load):
    1. Incarca datele brute (CSV).
    2. Curata si standardizeaza etichetele.
    3. Genereaza etichete binare pentru modelul de screening.
    4. Imparte datele in seturi Train/Validation/Test folosind stratificare.
    5. Salveaza fisierele procesate pentru antrenare.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os

# ==========================================
# 1. CONFIGURARE & CONSTANTE
# ==========================================
INPUT_FILE = '../data/Annotated_data.csv'
OUTPUT_DIR = '../data/'
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================
# 2. INCARCAREA SI CURATAREA DATELOR
# ======================================
try:
    df = pd.read_csv(INPUT_FILE)
    print(" Datele au fost incarcate cu succes.")
except FileNotFoundError:
    print(f"Eroare: Nu am gasit fisierul: {INPUT_FILE}")
    exit()

# Pastram doar coloanele relevante, le redenumim si eliminam randuri incomplete
    # 'id': Identificator unic
    # 'text': Declaratia pacientului
    # 'label_multi': Eticheta clinica (tipul distorsiunii)
df = df[['Id_Number', 'Patient Question', 'Dominant Distortion']].copy()
df.columns = ['id', 'text', 'label_multi']
df.dropna(subset=['text', 'label_multi'], inplace=True)

# ====================================
# 3. FEATURE ENGINEERING (Etichete)
# ====================================

# Generare Eticheta Binara pentru Modelul 1
# Logica: 0 = 'No Distortion', 1 = 'Distortion'
df['label_binary'] = df['label_multi'].apply(lambda x: 0 if x == 'No Distortion' else 1)
# =========================================================
# 4. IMPARTIREA DATELOR (Train/Val/Test) - in 2 splituri
# =========================================================
train_df, temp_df = train_test_split(
    df, test_size=0.2, random_state=RANDOM_SEED, stratify=df['label_binary']
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=RANDOM_SEED, stratify=temp_df['label_binary']
)
# ================================================
# 5. PREGATIREA SETURILOR PENTRU AMBELE MODELE
# ================================================

# 5.1. Seturile pentru Modelul I (Binar)
train_df.to_csv(os.path.join(OUTPUT_DIR, 'train_df_binary.csv'), index=False)
val_df.to_csv(os.path.join(OUTPUT_DIR, 'val_df_binary.csv'), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, 'test_df_binary.csv'), index=False)
print("Seturile Binar (Model I) salvate.")


# 5.2. Seturile pentru Modelul II (Multi-Class)

train_multi = train_df[train_df['label_binary'] == 1].copy()
val_multi = val_df[val_df['label_binary'] == 1].copy()
test_multi = test_df[test_df['label_binary'] == 1].copy()

# 5.3. Salvare
train_multi.to_csv(os.path.join(OUTPUT_DIR, 'train_df_multi.csv'), index=False)
val_multi.to_csv(os.path.join(OUTPUT_DIR, 'val_df_multi.csv'), index=False)
test_multi.to_csv(os.path.join(OUTPUT_DIR, 'test_df_multi.csv'), index=False)
print("Seturile Multi-Class (Model II) salvate.")

print("\n--- Preprocesare incheiata cu succes. Poate incepe antrenamentul modulelor. ---")