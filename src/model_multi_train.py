"""
CogniSense AI - Multi-Class Classification Training Script
-----------------------------------------------------------
Author: Carpiuc Alex
University: Universitatea Babeș-Bolyai
Description:
    Acest script antreneaza Modelul II (Clasificare Clinica distorsiuni cognitive pe 10 clase).
    Obiectiv: Identificarea tipului specific de distorsiune cognitiva.
    Arhitectura & Tehnici:
    - Weighted Loss Function (pentru gestionarea claselor dezechilibrate).
    - RoBERTa Base (pentru o mai buna intelegere semantica- este mai robust decat BERT).
"""




import pandas as pd
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# ================
# 1. CONFIGURARE
# ================
MODEL_NAME = "roberta-base"
OUTPUT_DIR = "../models/model_multi"
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# 2. INCARCARE SI PREGATIRE DATE
# ===============================
try:
    train_df = pd.read_csv('../data/train_df_multi.csv')
    val_df = pd.read_csv('../data/val_df_multi.csv')
    print("Datele Multi-Class (Model II) incarcate.")
except FileNotFoundError:
    print("Eroare: Nu am gasit fisierele")
    exit()

# MAPARE ETICHETE
label_list = sorted(train_df['label_multi'].unique().tolist())
label_to_id = {label: i for i, label in enumerate(label_list)}

train_df['labels'] = train_df['label_multi'].map(label_to_id)
val_df['labels'] = val_df['label_multi'].map(label_to_id)

NUM_LABELS = len(label_list)

# ---------------------------------------------------------
# CALCUL CLASS WEIGHTS
# Deoarece setul de date este dezechilibrat (unele distorsiuni sunt rare),
# calculam o "greutate" pentru fiecare clasa pentru a ajuta modelul.
# ---------------------------------------------------------
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_df['labels']),
    y=train_df['labels']
)
# Convertim in Tensor pentru PyTorch
class_weights = torch.tensor(class_weights, dtype=torch.float)

# ================
# 3. TOKENIZARE
# ================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


train_dataset = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
val_dataset = Dataset.from_pandas(val_df).map(tokenize_function, batched=True)

train_dataset = train_dataset.remove_columns(['id', 'text', 'label_multi'])
val_dataset = val_dataset.remove_columns(['id', 'text', 'label_multi'])


# ===================================
# 4. CUSTOM TRAINER (WEIGHTED LOSS)
# ===================================
class WeightedTrainer(Trainer):
    """
        Extindem clasa Trainer standard pentru a folosi Weighted Cross Entropy.
        Asta forteaza modelul sa acorde atentie si claselor minoritare.
        """
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss(weight=class_weights.to(model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


# =================================
# 5. CONFIGURARE MODEL SI METRICI
# =================================

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)


def compute_metrics(p):
    predictions = p.predictions.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        p.label_ids, predictions, average='macro', zero_division=0
    )
    acc = accuracy_score(p.label_ids, predictions)
    return {'macro_f1': f1, 'macro_accuracy': acc}


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="macro_f1",
    load_best_model_at_end=True,
    seed=RANDOM_SEED,
    learning_rate=2e-5,
    dataloader_pin_memory=False
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

# =========================
# 6. EXECUTIE SI SALVARE
# =========================

print("\n Incepe Antrenarea cu Weighted Loss...")
trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

with open(f"{OUTPUT_DIR}/labels.txt", "w") as f:
    for label in label_list:
        f.write(f"{label}\n")

print(f"Modelul II salvat in: {OUTPUT_DIR}")