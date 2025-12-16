"""
CogniSense AI - Model Evaluation Module
Author: Carpiuc Alex
University: Universitatea Babeș-Bolyai
Description:
    Acest modul evalueaza performanta modelelor antrenate (Binary și Multi-class)
    pe setul de date de testare (nevazut la antrenare - 10%).

    Functionalitati:
    1. Calculeaza metrici standard: Accuracy, Precision, Recall, F1-Score.
    2. Genereaza rapoarte detaliate de clasificare (Classification Report).
    3. Salveaza rezultatele intr-un fisier text pentru includerea in lucrarea de licenta.
"""

import pandas as pd
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# ==========================================
# 1. CONFIGURARE & CONSTANTE
# ==========================================
DATA_DIR = '../data/'
MODEL_DIR_BINARY = '../models/model_binary/'
MODEL_DIR_MULTI = '../models/model_multi/'

# Configurare
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running evaluation on: {device}")


# ==========================================
# 2. FUNCTII UTILITARE (METRICI)
# ==========================================
def compute_metrics_binary(p):
    """
        Calculeaza metricile pentru clasificarea binara (Screening).
        Foloseste 'average=binary' pentru a se concentra pe clasa pozitiva (Distorsionat).
    """
    predictions = p.predictions.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, predictions, average='binary',
                                                               zero_division=0)
    acc = accuracy_score(p.label_ids, predictions)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}


def compute_metrics_multi(p):
    """
        Calculeaza metricile pentru clasificarea multi-class (Tip Distorsiune).
        Foloseste 'average=weighted' pentru a tine cont de dezechilibrul claselor.
    """
    predictions = p.predictions.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, predictions, average='weighted',
                                                               zero_division=0)
    acc = accuracy_score(p.label_ids, predictions)
    return {'accuracy': acc, 'weighted_f1': f1, 'weighted_precision': precision, 'weighted_recall': recall}


# ==========================================
# 3. MOTORUL DE EVALUARE-FUNCTIA PRINCIPALA
# ==========================================
def run_evaluation(model_dir, data_file, metrics_fn, model_type):
    """
        Execută fluxul complet de evaluare pentru un model specific.

    """
    print(f"\n{'=' * 20} Evaluare: {model_type} {'=' * 20}")

    # 1. Incarcare Resurse
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        print("Model incarcat.")
    except Exception as e:
        print(f"Eroare incarcare model {model_dir}: {e}")
        return

    # 2. Incarcare Date Test
    try:
        df_test = pd.read_csv(os.path.join(DATA_DIR, data_file))
    except FileNotFoundError:
        print(f"Nu gasesc fisierul de test: {data_file}")
        return

    # 3. Procesare Etichete & Mapare
    target_names = None
    if model_type == 'BINARY':
        # Binar: 0 = Non-Distorted, 1 = Distorted
        df_test['labels'] = df_test['label_binary']
        target_names = ["Non-Distorted", "Distorted"]
    else:  # MULTI

        try:
            with open(os.path.join(model_dir, "labels.txt"), "r") as f:
                label_list = [line.strip() for line in f if line.strip()]

            label_to_id = {label: i for i, label in enumerate(label_list)}
            df_test['labels'] = df_test['label_multi'].map(label_to_id)
            target_names = label_list


            df_test.dropna(subset=['labels'], inplace=True)
            df_test['labels'] = df_test['labels'].astype(int)

        except FileNotFoundError:
            print("Nu gasesc labels.txt in folderul modelului multi.")
            return

    # 4. Tokenizare
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    test_dataset = Dataset.from_pandas(df_test).map(tokenize_function, batched=True)
    test_dataset = test_dataset.remove_columns(
        [col for col in df_test.columns if col not in ['input_ids', 'attention_mask', 'labels']])

    # 5. Predictie
    trainer = Trainer(model=model, tokenizer=tokenizer)
    raw_preds = trainer.predict(test_dataset)

    preds = np.argmax(raw_preds.predictions, axis=1)
    labels = raw_preds.label_ids

    # 6. Afisare Rezultate
    print("\n --- METRICI GLOBALE ---")
    print(metrics_fn(raw_preds))

    print("\n --- CLASSIFICATION REPORT ---")
    report = classification_report(labels, preds, target_names=target_names, zero_division=0)
    print(report)

    return report


# =============
# 4. EXECUTIE
# =============
if __name__ == '__main__':
    # Evaluare Binar
    run_evaluation(MODEL_DIR_BINARY, 'test_df_binary.csv', compute_metrics_binary, 'BINARY')

    # Evaluare Multi-Class
    run_evaluation(MODEL_DIR_MULTI, 'test_df_multi.csv', compute_metrics_multi, 'MULTI-CLASS')