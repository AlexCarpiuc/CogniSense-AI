# 🧠 CogniSense AI

> **Bachelor's Thesis Application** > **Topic:** Advanced Cognitive Distortion Analysis & Clinical Profiling System  
> **Author:** Alex Carpiuc  
> **University:** Babeș-Bolyai University  

---

## Overview

**CogniSense AI** is a multimodal intelligence system designed to assist clinicians in detecting and analyzing cognitive distortions in patients' speech and text. 

Unlike standard sentiment analysis tools, this system implements a hierarchical NLP pipeline combined with vocal biomarker extraction to provide a comprehensive clinical profile. It distinguishes between healthy negative emotions and pathological cognitive distortions (e.g., *Catastrophizing, All-or-Nothing Thinking*).

### Key Features

* **Multimodal Input:** Accepts textual input, live microphone recording, and audio file uploads (WAV/MP3).
* **Hierarchical NLP Architecture:**
    1.  **Screening Model (BERT):** Binary classification to filter distorted vs. non-distorted statements.
    2.  **Clinical Model (RoBERTa):** Multi-class classification trained with **Weighted Cross-Entropy Loss** to identify specific distortion types.
* **Audio Signal Processing:** Uses `librosa` to extract prosodic features:
    * *Speech Rate (WPM)* & *Silence Ratio* (indicators for psychomotor retardation or agitation).
    * *Pitch Variability (F0)* (indicator for flat affect/monotony).
* **Explainable AI (XAI):** Input perturbation algorithms to highlight "trigger words" that influenced the AI's decision.
* **Dual Interface:** * **Doctor Mode:** Technical dashboards, spectrograms, radar charts, and PDF reporting.
    * **Patient Mode:** Simplified feedback and psycho-education (CBT tips).

---

## 🛠️ Tech Stack

* **Language:** Python 3.9+
* **Frontend:** Streamlit
* **Machine Learning:** HuggingFace Transformers, PyTorch, Scikit-learn
* **Audio Processing:** Librosa, SoundFile, SpeechRecognition (Google API)
* **Visualization:** Plotly, Matplotlib, WordCloud
* **Reporting:** FPDF

---

## 📂 Project Structure

```text
CogniSense-AI/
├── data/                   # Raw and processed datasets
├── models/                 # Saved models (Local storage only, not on GitHub)
│   ├── model_binary/       # Fine-tuned BERT for screening
│   └── model_multi/        # Fine-tuned RoBERTa for specific classification
├── src/
│   ├── app.py              # Main Streamlit application
│   ├── data_prep.py        # ETL pipeline and data splitting
│   ├── model_binary_train.py # Training script for Model I
│   ├── model_multi_train.py  # Training script for Model II
│   └── evaluate.py         # Metrics calculation (F1-Score, Confusion Matrix)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation