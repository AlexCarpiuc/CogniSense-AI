import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

# --- 0. CONFIGURARE ---
MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 2
OUTPUT_DIR = "../models/model_binary"
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ⚠️ Atenție: Încărcare din fișierele salvate de data_prep.py
try:
    train_df = pd.read_csv('../data/train_df_binary.csv')
    val_df = pd.read_csv('../data/val_df_binary.csv')
    print("✅ Datele Binar (Model I) încărcate.")
except FileNotFoundError:
    print("❌ Eroare: Nu am găsit fișierele de date binare. Rulează '../src/data_prep.py' mai întâi.")
    exit()

# --- 1. PREGĂTIRE TOKENIZER ȘI DATASET ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_function(examples):
    # 'text' este coloana Patient Question redenumită
    return tokenizer(examples["text"], padding="max_length", truncation=True)


# Conversia DataFrame-urilor în formatul 'Dataset'
train_dataset = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
val_dataset = Dataset.from_pandas(val_df).map(tokenize_function, batched=True)

# Renumim coloana 'label_binary' în 'labels' (cerută de Trainer)
train_dataset = train_dataset.rename_column("label_binary", "labels")
val_dataset = val_dataset.rename_column("label_binary", "labels")

# Eliminăm coloanele inutile
train_dataset = train_dataset.remove_columns(["id", "text", "label_multi"])
val_dataset = val_dataset.remove_columns(["id", "text", "label_multi"])

# Modelul BERT (adaugă un strat de clasificare pentru 2 etichete)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)


# --- 2. METRICI DE EVALUARE ---
def compute_metrics(p):
    predictions = p.predictions.argmax(axis=1)

    # average='binary' este folosit pentru clasificarea binară
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, predictions, average='binary')
    acc = accuracy_score(p.label_ids, predictions)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }


# --- 3. ARGUMENTE ȘI RULARE TRAINER ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    dataloader_num_workers=0,
    weight_decay=0.01,
    logging_steps=50,

    dataloader_pin_memory=False,


    eval_strategy="epoch",
    save_strategy="epoch",


    metric_for_best_model="f1",  # Spunem modelului să urmărească F1-Score
    greater_is_better=True,  # Spunem modelului că scorul F1 mai mare este mai bun
    load_best_model_at_end=True,  # ⬅️ Păstrăm totuși această denumire, dar adăugăm metrici de bază


    seed=RANDOM_SEED,
    learning_rate=2e-5
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

print("\n🔥 Începe Finetuning-ul Modelului I (Binar - Screening)...")
trainer.train()

# --- 4. SALVARE MODEL FINAL ---
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Modelul I a fost salvat în directorul: {OUTPUT_DIR}")