import pandas as pd
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# --- 0. CONFIGURARE ---
MODEL_NAME = "roberta-base"
OUTPUT_DIR = "../models/model_multi"
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Încărcare date
try:
    train_df = pd.read_csv('../data/train_df_multi.csv')
    val_df = pd.read_csv('../data/val_df_multi.csv')
    print("✅ Datele Multi-Class (Model II) încărcate.")
except FileNotFoundError:
    print("❌ Eroare: Nu am găsit fișierele. Rulează data_prep.py.")
    exit()

# --- 1. PREGĂTIREA ETICHETELOR & CALCULAREA GREUTĂȚILOR (WEIGHTS) ---
label_list = sorted(train_df['label_multi'].unique().tolist())
label_to_id = {label: i for i, label in enumerate(label_list)}

train_df['labels'] = train_df['label_multi'].map(label_to_id)
val_df['labels'] = val_df['label_multi'].map(label_to_id)

NUM_LABELS = len(label_list)

# 🔥 CALCUL GREUTĂȚI PENTRU CLASE (BALANCING)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_df['labels']),
    y=train_df['labels']
)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# --- 2. PREGĂTIRE DATASET ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


train_dataset = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
val_dataset = Dataset.from_pandas(val_df).map(tokenize_function, batched=True)

train_dataset = train_dataset.remove_columns(['id', 'text', 'label_multi'])
val_dataset = val_dataset.remove_columns(['id', 'text', 'label_multi'])


# --- 3. DEFINIRE CUSTOM TRAINER (PENTRU WEIGHTED LOSS) ---
class WeightedTrainer(Trainer):
    # ⚠️ FIX: Am adăugat **kwargs pentru a accepta argumente noi din versiunile noi de Transformers
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Trimitem greutățile pe același device cu modelul (GPU/CPU)
        loss_fct = nn.CrossEntropyLoss(weight=class_weights.to(model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


# --- 4. CONFIGURARE MODEL & TRAINER ---
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
    dataloader_pin_memory=False  # Dezactivat pentru a evita warning-ul de CPU
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

print("\n🔥 Începe Antrenarea cu Weighted Loss (Balansare Clase)...")
trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

with open(f"{OUTPUT_DIR}/labels.txt", "w") as f:
    for label in label_list:
        f.write(f"{label}\n")

print(f"✅ Modelul II salvat în: {OUTPUT_DIR}")