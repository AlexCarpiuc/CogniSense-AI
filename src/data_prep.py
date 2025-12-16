import pandas as pd
from sklearn.model_selection import train_test_split
import os

# --- 0. CONFIGURARE ---
INPUT_FILE = '../data/Annotated_data.csv'
OUTPUT_DIR = '../data/'
RANDOM_SEED = 42

# Ne asigurăm că directorul de output există
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. ÎNCĂRCAREA ȘI CURĂȚAREA DATELOR ---
try:
    df = pd.read_csv(INPUT_FILE)
    print("✅ Datele au fost încărcate cu succes.")
except FileNotFoundError:
    print(f"❌ Eroare: Nu am găsit fișierul la calea: {INPUT_FILE}")
    exit()

# Reținerea coloanelor esențiale și redenumirea lor
df = df[['Id_Number', 'Patient Question', 'Dominant Distortion']].copy()
df.columns = ['id', 'text', 'label_multi']
df.dropna(subset=['text', 'label_multi'], inplace=True)

# --- 2. CREAREA ETICHETEI BINARE (MODEL I - SCREENING) ---
# 'No Distortion' este clasa negativă (0); orice altceva este Distorsionat (1)
df['label_binary'] = df['label_multi'].apply(lambda x: 0 if x == 'No Distortion' else 1)

# --- 3. ÎMPĂRȚIREA DATELOR (Train/Val/Test) ---

# Split-ul I: Separă 80% Train, 20% Temporar (pentru Validation și Test)
# Folosim 'stratify' pentru a menține proporția de 0/1 (Non/Distorsionat) în fiecare set
train_df, temp_df = train_test_split(
    df, test_size=0.2, random_state=RANDOM_SEED, stratify=df['label_binary']
)

# Split-ul II: Separă setul temporar în 10% Validation și 10% Test
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=RANDOM_SEED, stratify=temp_df['label_binary']
)

# --- 4. PREGĂTIREA SETURILOR PENTRU AMBELE MODELE ȘI SALVAREA ---

# 4.1. Seturile pentru Modelul I (Binar)
train_df.to_csv(os.path.join(OUTPUT_DIR, 'train_df_binary.csv'), index=False)
val_df.to_csv(os.path.join(OUTPUT_DIR, 'val_df_binary.csv'), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, 'test_df_binary.csv'), index=False)
print("✅ Seturile Binar (Model I) salvate.")


# 4.2. Seturile pentru Modelul II (Multi-Class) - Filtrare
# Păstrăm doar rândurile clasificate ca fiind Distorsionate (label_binary == 1)
train_multi = train_df[train_df['label_binary'] == 1].copy()
val_multi = val_df[val_df['label_binary'] == 1].copy()
test_multi = test_df[test_df['label_binary'] == 1].copy()

# Salvăm seturile filtrate
train_multi.to_csv(os.path.join(OUTPUT_DIR, 'train_df_multi.csv'), index=False)
val_multi.to_csv(os.path.join(OUTPUT_DIR, 'val_df_multi.csv'), index=False)
test_multi.to_csv(os.path.join(OUTPUT_DIR, 'test_df_multi.csv'), index=False)
print("✅ Seturile Multi-Class (Model II) salvate.")

print("\n--- Preprocesare încheiată. Vă rugăm să rulați scriptul de antrenare a modelelor. ---")