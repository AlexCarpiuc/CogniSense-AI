"""
CogniSense AI - Binary Classification Training Pipeline
Author: Carpiuc Alex
University: Universitatea Babeș-Bolyai
Description:
    Acest script antreneaza Modelul I (Screening Binar).
    Scop: Distinge intre 'Non-Distorsionat' si 'Distorsionat'

    Arhitectura:
    - Metodologie: Fine-tuning pe modelul BERT (bert-base-uncased).
    - Loss Function: Standard Cross-Entropy.
    - Metrici: F1-Score (pentru a balansa Precision/Recall).
"""



import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# =============================
# 1. CONFIGURARE & PARAMETRI
# =============================
MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 2
OUTPUT_DIR = "../models/model_binary"
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. INCARCAREA SETURILOR DE DATE
# ==========================================
try:
    train_df = pd.read_csv('../data/train_df_binary.csv')
    val_df = pd.read_csv('../data/val_df_binary.csv')
    print("Datele Binar (Model I) incarcate.")
except FileNotFoundError:
    print("Eroare: Nu am gasit fisierele de date binare.")
    exit()

# ==========================================
# 3. TOKENIZARE SI PREGATIRE DATASET
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_function(examples):
    """
        Transforma textul in tokeni numerici acceptati de BERT.
    """
    return tokenizer(examples["text"], padding="max_length", truncation=True)


# Conversie din Pandas DataFrame in HuggingFace Dataset
train_dataset = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
val_dataset = Dataset.from_pandas(val_df).map(tokenize_function, batched=True)

# Ajustari structurale pentru compatibilitatea cu biblioteca Transformers:

train_dataset = train_dataset.rename_column("label_binary", "labels")
val_dataset = val_dataset.rename_column("label_binary", "labels")


train_dataset = train_dataset.remove_columns(["id", "text", "label_multi"])
val_dataset = val_dataset.remove_columns(["id", "text", "label_multi"])

# =======================
# 4. INITIALIZARE MODEL
# =======================

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)


# ==========================================
# 5. DEFINIREA METRICILOR DE PERFORMANTA
# ==========================================
def compute_metrics(p):
    """
        Functie apelata la finalul fiecarei epoci pentru evaluare.
        Returneaza: Accuracy, Precision, Recall si F1-Score.
    """
    predictions = p.predictions.argmax(axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, predictions, average='binary')
    acc = accuracy_score(p.label_ids, predictions)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }


# =========================
# 6. CONFIGURARE TRAINER
# =========================
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


    metric_for_best_model="f1",
    greater_is_better=True,
    load_best_model_at_end=True,


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

# =====================================
# 7. EXECUTIE ANTRENAMENT SI SALVARE
# =====================================

print("\n Incepe Finetuning-ul Modelului I (Binar - Screening)...")
trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f" Modelul I a fost salvat in: {OUTPUT_DIR}")