"""
CogniSense AI - Bachelor's Thesis Application
Author: Carpiuc Alex
University: Universitatea Babeș-Bolyai, Facultatea de Matematică și Informatică
Description: Multimodal AI system for detecting cognitive distortions and emotional patterns.
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
import csv
import time
from datetime import datetime
import base64
import matplotlib.pyplot as plt

# --- BIBLIOTECI EXTERNE ---
from wordcloud import WordCloud
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from scipy.special import softmax
from fpdf import FPDF
import plotly.express as px
import speech_recognition as sr
import librosa
import librosa.display

# --- NLP SETUP ---
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# ==========================================
# 1. CONFIGURARE & CONSTANTE
# ==========================================

# Info
STUDENT_NAME = "Carpiuc Alex"
COORD_NAME = "Prof. Dr. Dioșan Laura"
UNIV_NAME = "Universitatea Babeș-Bolyai"
FACULTY_NAME = "Facultatea de Matematică și Informatică"

# Configurare Pagina Streamlit
st.set_page_config(
    page_title="CogniSense AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Css interfata
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Header Principal */
    .main-header {
        background: linear-gradient(90deg, #2c3e50 0%, #4ca1af 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    /* Carduri Metrice */
    .metric-card {
        background-color: #ffffff;
        border-left: 6px solid #4ca1af;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    /* Highlight Cuvinte (XAI) */
    .highlight-word {
        background-color: #ffcdd2; 
        color: #b71c1c; 
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: bold;
        border-bottom: 2px solid #b71c1c;
    }

    /* Banner Alerta Risc */
    .safety-banner {
        background-color: #ffebee;
        border: 2px solid #ef5350;
        color: #c62828;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        margin-bottom: 20px;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(239, 83, 80, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(239, 83, 80, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 83, 80, 0); }
    }
    </style>
""", unsafe_allow_html=True)

#Session State (Memorie temporara)
if 'last_text' not in st.session_state: st.session_state['last_text'] = ""
if 'last_pred' not in st.session_state: st.session_state['last_pred'] = ""
if 'session_transcript' not in st.session_state: st.session_state['session_transcript'] = ""
if 'audio_features' not in st.session_state: st.session_state['audio_features'] = None
if 'clinical_scores' not in st.session_state: st.session_state['clinical_scores'] = {'depression': 0.0, 'anxiety': 0.0}
if 'last_audio_file' not in st.session_state: st.session_state['last_audio_file'] = None
if 'emotion_scores' not in st.session_state: st.session_state['emotion_scores'] = None
if 'narrative_text' not in st.session_state: st.session_state['narrative_text'] = ""
if 'accepted_terms' not in st.session_state: st.session_state['accepted_terms'] = False

# Routing Fisiere & Modele
MODEL_DIR_BINARY = "../models/model_binary"
MODEL_DIR_MULTI = "../models/model_multi"
FEEDBACK_FILE = "../data/feedback_data.csv"
TEMP_AUDIO_FILE = "temp_recording.wav"
TEMP_RADAR_FILE = "temp_radar.png"
device = torch.device("cpu")

# Definitii explicative clase distorsiuni cognitive + tips
DISTORTION_DEFINITIONS = {
    "All-or-nothing thinking": "Thinking in absolutes (black & white).",
    "Emotional Reasoning": "Assuming feelings are facts.",
    "Fortune-telling": "Predicting negative future outcomes.",
    "Labeling": "Assigning global negative traits to oneself.",
    "Magnification": "Catastrophizing events.",
    "Mental filter": "Focusing only on negatives.",
    "Mind Reading": "Believing you know others' negative thoughts.",
    "Overgeneralization": "Seeing patterns based on single events.",
    "Personalization": "Taking blame for uncontrollable events.",
    "Should statements": "Using critical pressure words."
}

CBT_STRATEGIES = {
    "All-or-nothing thinking": "💡 **Tip:** Look for 'shades of gray'. Rate 0-100.",
    "Emotional Reasoning": "💡 **Tip:** Feelings are not facts. Check evidence.",
    "Fortune-telling": "💡 **Tip:** Ask: 'What is the most likely outcome?'",
    "Labeling": "💡 **Tip:** Label behavior, not the person.",
    "Magnification": "💡 **Tip:** Will this matter in 1 year?",
    "Mental filter": "💡 **Tip:** Find one neutral thing today.",
    "Mind Reading": "💡 **Tip:** Consider alternative explanations.",
    "Overgeneralization": "💡 **Tip:** Watch for 'always' and 'never'.",
    "Personalization": "💡 **Tip:** Influence vs Control distinction.",
    "Should statements": "💡 **Tip:** Use 'It would be nice if...'."
}

# Ponderi pentru calculul riscurilor de anxietate/depresie
CLINICAL_WEIGHTS = {
    "All-or-nothing thinking": {"dep": 1.2, "anx": 0.5},
    "Emotional Reasoning": {"dep": 0.8, "anx": 0.8},
    "Fortune-telling": {"dep": 0.2, "anx": 1.5},
    "Labeling": {"dep": 1.5, "anx": 0.2},
    "Magnification": {"dep": 0.5, "anx": 1.5},
    "Mental filter": {"dep": 1.2, "anx": 0.3},
    "Mind Reading": {"dep": 0.4, "anx": 1.2},
    "Overgeneralization": {"dep": 1.0, "anx": 0.6},
    "Personalization": {"dep": 1.0, "anx": 1.0},
    "Should statements": {"dep": 1.0, "anx": 0.8}
}


# =======================
# 2. INCARCARE MODELE
# =======================
@st.cache_resource
def load_models():
    """
    Incarca modelele BERT fine-tuned si modelul de emotii RoBERTa.
    Foloseste @st.cache_resource pentru a nu reincarca la fiecare interactiune.
    """
    try:
        if not os.path.exists(MODEL_DIR_BINARY) or not os.path.exists(MODEL_DIR_MULTI):
            return None, None, None, None, None, None, False

        # 1. Binary Classification (Distorted vs Normal)
        tok_bin = AutoTokenizer.from_pretrained(MODEL_DIR_BINARY)
        mod_bin = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR_BINARY).to(device)

        # 2. Multi-class Classification (Tipul Distorsiunii)
        tok_mul = AutoTokenizer.from_pretrained(MODEL_DIR_MULTI)
        mod_mul = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR_MULTI).to(device)

        # 3. Emotion Analysis (Hugging Face Pipeline)
        emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base",
                                      return_all_scores=True)


        label_path = os.path.join(MODEL_DIR_MULTI, "labels.txt")
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                labels = [line.strip() for line in f if line.strip()]
        else:
            return None, None, None, None, None, None, False

        return tok_bin, mod_bin, tok_mul, mod_mul, labels, emotion_classifier, True
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None, None, None, None, False


tokenizer_bin, model_bin, tokenizer_multi, model_multi, label_list_multi, emotion_pipe, LOAD_SUCCESS = load_models()


# ==========================================
# 3. FUNCTII DE PROCESARE NLP & AUDIO
# ==========================================

def predict_full_pipeline(text):
    """Pipeline complet: Binar -> Multi."""
    # Pas 1: Predicție Binară
    inputs = tokenizer_bin(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        output = model_bin(**inputs)
    scores_bin = softmax(output.logits.cpu().numpy()[0])
    pred_bin = np.argmax(scores_bin)
    conf_bin = scores_bin[pred_bin]

    if pred_bin == 0:
        return "Non-Distorted", conf_bin, None, conf_bin, 0

    # Pas 2: Predictie Multi
    inputs_m = tokenizer_multi(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        output_m = model_multi(**inputs_m)
    scores_m = softmax(output_m.logits.cpu().numpy()[0])
    top_idx = np.argmax(scores_m)
    return label_list_multi[top_idx], scores_m[top_idx], scores_m, conf_bin, 1


def analyze_emotions_granular(text):
    """Extrage scorurile pentru 7 emotii de baza."""
    if not text or len(text.split()) < 2: return {}
    preds = emotion_pipe(text[:512])
    return {item['label']: item['score'] for item in preds[0]}


def explain_text(text, target_label):
    """XAI: Calculeaza importanta cuvintelor prin tehnica de perturbare."""
    words = text.split()
    base_type, base_conf, _, _, _ = predict_full_pipeline(text)
    if base_type == "Non-Distorted": return []

    importance_scores = []
    words_to_check = words[:50]
    for i in range(len(words_to_check)):
        if len(words[i]) < 3 and words[i].lower() not in ["no", "not", "bad", "sad"]:
            importance_scores.append((words[i], 0))
            continue

        masked_text = " ".join(words[:i] + words[i + 1:])
        _, new_conf, _, _, _ = predict_full_pipeline(masked_text)
        impact = base_conf - new_conf
        importance_scores.append((words[i], impact))
    return importance_scores


def extract_vocal_features(audio_file, text_transcript):
    """Extrage biomarkeri acustici folosind Librosa."""
    try:
        y, sr_rate = librosa.load(audio_file, sr=None)
        duration = librosa.get_duration(y=y, sr=sr_rate)

        # Detectare liniste vs vorbire
        non_silent_intervals = librosa.effects.split(y, top_db=20)
        speech_time = sum((end - start) / sr_rate for start, end in non_silent_intervals)
        silence_time = duration - speech_time
        silence_ratio = (silence_time / duration) * 100

        # Calcul WPM
        word_count = len(text_transcript.split()) if text_transcript else 0
        wpm = (word_count / duration) * 60 if duration > 0 else 0

        # Calcul Pitch (F0)
        f0 = librosa.yin(y, fmin=60, fmax=300)
        f0 = f0[f0 > 0]
        pitch_std = np.std(f0) if len(f0) > 0 else 0

        return {
            "duration": duration,
            "silence_ratio": silence_ratio,
            "wpm": wpm,
            "pitch_std": pitch_std
        }
    except:
        return None


def transcribe_speech_and_analyze(status_placeholder):
    """Inregistreaza microfon, transcrie (Google API) si analizeaza."""
    r = sr.Recognizer()
    r.dynamic_energy_threshold = True
    r.energy_threshold = 300
    r.pause_threshold = 1.2

    try:
        with sr.Microphone() as source:
            status_placeholder.info("🎙️ Ascult... Vorbește acum!")
            r.adjust_for_ambient_noise(source, duration=1)
            audio_data = r.listen(source, timeout=10, phrase_time_limit=30)
            status_placeholder.warning("⏳ Procesez audio (Transcriere)...")

            with open(TEMP_AUDIO_FILE, "wb") as f:
                f.write(audio_data.get_wav_data())

            try:
                file_source = sr.AudioFile(TEMP_AUDIO_FILE)
                with file_source as source_file:
                    audio_from_file = r.record(source_file)
                    text = r.recognize_google(audio_from_file, language="en-US")
            except sr.UnknownValueError:
                status_placeholder.error("Nu am detectat cuvinte clare.")
                return "", None
            except Exception:
                text = ""

            features = extract_vocal_features(TEMP_AUDIO_FILE, text)
            status_placeholder.success("✅ Gata!")
            time.sleep(1)
            status_placeholder.empty()
            return text, features
    except Exception as e:
        status_placeholder.error(f"Eroare microfon: {e}")
        return None, None


def process_uploaded_audio(uploaded_file, status_placeholder):
    """Proceseaza fisiere audio incarcate (WAV/MP3)."""
    try:
        status_placeholder.info("📂 Processing uploaded file...")
        with open(TEMP_AUDIO_FILE, "wb") as f:
            f.write(uploaded_file.getbuffer())

        r = sr.Recognizer()
        with sr.AudioFile(TEMP_AUDIO_FILE) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data, language="en-US")

        features = extract_vocal_features(TEMP_AUDIO_FILE, text)
        status_placeholder.success("✅ File processed!")
        time.sleep(1)
        status_placeholder.empty()
        return text, features
    except Exception as e:
        status_placeholder.error(f"Error processing file: {e}")
        return None, None


def plot_spectrogram(audio_file):
    """Generează spectrograma """
    try:
        y, sr = librosa.load(audio_file)
        fig, ax = plt.subplots(figsize=(10, 3))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax, cmap='magma')
        ax.set_title('Mel-frequency Spectrogram (Voice Signature)')
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        return fig
    except:
        return None


def generate_narrative_report(scores, audio_features, emotions):
    """Genereaza un text narativ bazat pe toti parametrii."""
    dom_risk = "Anxiety" if scores['anxiety'] > scores['depression'] else "Depression"
    max_score = max(scores['depression'], scores['anxiety'])
    level = "Severe" if max_score > 3.5 else "Moderate" if max_score > 2.0 else "Mild"

    text = f"**Clinical Summary:** The subject exhibits a **{level}** level of cognitive distortions, predominantly consistent with a **{dom_risk}** pattern. "

    if emotions:
        top_emo = max(emotions, key=emotions.get)
        text += f"Textual sentiment analysis indicates a dominant state of **{top_emo.upper()}**. "

    if audio_features:
        if audio_features['wpm'] < 110:
            text += "Vocal biomarkers show reduced speech rate (psychomotor retardation). "
        elif audio_features['wpm'] > 160:
            text += "Rapid speech rate suggests agitation. "

        if audio_features.get('pitch_std', 20) < 15:
            text += "Monotone intonation detected (flat affect). "

    return text


def analyze_session(full_text):
    """Functia principala care leaga toate analizele pentru o sesiune."""
    sentences = smart_sentence_split(full_text)
    results = []
    distortion_counts = {label: 0 for label in label_list_multi}
    scores = {"depression": 0.0, "anxiety": 0.0}
    total_distortions = 0

    for sent in sentences:
        dtype, conf, _, _, bin_class = predict_full_pipeline(sent)
        if bin_class == 1:
            distortion_counts[dtype] += 1
            total_distortions += 1
            if dtype in CLINICAL_WEIGHTS:
                scores["depression"] += CLINICAL_WEIGHTS[dtype]['dep']
                scores["anxiety"] += CLINICAL_WEIGHTS[dtype]['anx']
            results.append({"text": sent, "type": dtype, "confidence": conf})
        else:
            results.append({"text": sent, "type": "Neutral", "confidence": conf})

    emotions = analyze_emotions_granular(full_text)
    return results, distortion_counts, total_distortions, scores, emotions


def smart_sentence_split(text):
    if not text: return []
    try:
        sentences = sent_tokenize(text)
    except:
        sentences = text.split('. ')
    return [s.strip() for s in sentences if len(s.split()) > 2]


def clean_text(text):
    return str(text).encode('latin-1', 'replace').decode('latin-1')


def save_feedback_to_csv(text, predicted, actual, is_correct):
    file_exists = os.path.isfile(FEEDBACK_FILE)
    with open(FEEDBACK_FILE, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Timestamp', 'Text', 'Predicted_Label', 'Actual_Label', 'Is_Correct'])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), text, predicted, actual, is_correct])


# --- GENERARE PDF ---
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.set_text_color(24, 40, 72)
        self.cell(0, 10, 'CogniSense AI - Clinical Report', 0, 1, 'C')
        self.line(10, 20, 200, 20)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')


def create_download_link(results, counts, insights, audio_features=None, emotions=None, narrative=""):
    try:
        pdf = PDFReport()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # 1. Clinical Insights
        pdf.set_font("Arial", 'B', 14)
        pdf.set_fill_color(230, 230, 250)
        pdf.cell(0, 10, clean_text("  1. Clinical Insights"), ln=True, fill=True)
        pdf.ln(5)
        pdf.set_font("Arial", size=11)

        if narrative:
            pdf.set_font("Arial", 'I', 11)
            pdf.multi_cell(0, 7, txt=clean_text(f"Summary: {narrative.replace('**', '')}"))
            pdf.ln(5)
            pdf.set_font("Arial", size=11)

        for msg, _, _ in insights:
            clean_msg = msg.replace("**", "").replace("🔴", "[High Risk]").replace("🟠", "[High Risk]").replace("🟡",
                                                                                                              "[Moderate]").replace(
                "✅", "[Balanced]")
            pdf.multi_cell(0, 7, txt=clean_text(f"- {clean_msg}"))

        if audio_features:
            pdf.ln(5)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, clean_text("  Vocal Biomarkers:"), ln=True)
            pdf.set_font("Arial", size=11)
            pdf.cell(0, 7, txt=f"- Speaking Rate: {audio_features['wpm']:.1f} WPM", ln=True)
            pdf.cell(0, 7, txt=f"- Silence Ratio: {audio_features['silence_ratio']:.1f}%", ln=True)
            pdf.cell(0, 7, txt=f"- Intonation Var: {audio_features.get('pitch_std', 0):.1f} Hz", ln=True)

        if emotions:
            pdf.ln(5)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, clean_text("  Emotional Profile (Text):"), ln=True)
            pdf.set_font("Arial", size=11)
            sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
            for emo, score in sorted_emotions:
                pdf.cell(0, 7, txt=f"- {emo.capitalize()}: {score:.2%}", ln=True)

        # Grafic Radar
        if os.path.exists(TEMP_RADAR_FILE):
            pdf.ln(5)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, clean_text("  Distortion Profile:"), ln=True)
            pdf.image(TEMP_RADAR_FILE, x=10, w=100)
            pdf.ln(5)

        # 2. Statistics
        pdf.ln(10)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, clean_text("  2. Distortion Statistics"), ln=True, fill=True)
        pdf.ln(5)
        pdf.set_font("Arial", size=11)
        for dtype, count in counts.items():
            if count > 0:
                pdf.cell(100, 7, txt=clean_text(f"{dtype}:"), border=0)
                pdf.cell(0, 7, txt=str(count), ln=True)

        # 3. Transcript
        pdf.ln(10)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, clean_text("  3. Transcript Analysis"), ln=True, fill=True)
        pdf.ln(5)
        pdf.set_font("Arial", size=10)
        for item in results:
            label = clean_text(item['type'])
            text = clean_text(item['text'])
            if label == "Neutral":
                pdf.set_text_color(100, 100, 100)
                pdf.multi_cell(0, 6, txt=f"[Neutral] {text}")
            else:
                pdf.set_text_color(200, 0, 0)
                pdf.multi_cell(0, 6, txt=f"[{label}] {text}")

        pdf_bytes = pdf.output(dest='S').encode('latin-1', errors='replace')
        b64 = base64.b64encode(pdf_bytes).decode()
        return f'<div style="text-align:center; margin-top:20px;"><a href="data:application/octet-stream;base64,{b64}" download="Clinical_Report.pdf" class="pdf-button">📄 Download Official PDF Report</a></div>'
    except Exception as e:
        return f"Error: {e}"


# ==========================================
# 4. ECRAN DE START & SIDEBAR
# ==========================================

# ConsentForm
if not st.session_state['accepted_terms']:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("""
        <div style="background-color:white; padding:30px; border-radius:10px; box-shadow:0 0 20px rgba(0,0,0,0.1); text-align:center;">
            <h2>🔒 Research Protocol Consent</h2>
            <p>Welcome to <b>CogniSense AI</b>.</p>
            <p style="text-align:left; font-size:14px; color:#555;">
            This application uses Artificial Intelligence (NLP) to analyze speech and text patterns related to cognitive distortions.<br><br>
            <b>By proceeding, you acknowledge that:</b><br>
            1. This is an academic prototype, not a medical device.<br>
            2. Audio data is processed locally for demonstration purposes.<br>
            3. Results are for informational purposes only and do not constitute a diagnosis.<br>
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("I Understand & Agree", type="primary", use_container_width=True):
            st.session_state['accepted_terms'] = True
            st.rerun()
    st.stop()

# Sidebar - Informatii
with st.sidebar:
    st.markdown("## 🩺 Control Panel")
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=100)

    st.markdown("### 🎓 Academic Info")
    st.markdown(f"**Student:** {STUDENT_NAME}")
    st.markdown(f"**University:** {UNIV_NAME}")
    st.markdown(f"**Faculty:** {FACULTY_NAME}")
    st.markdown(f"**Coordinator:** {COORD_NAME}")

    st.markdown("---")
    user_mode = st.radio("Display Mode", ["Doctor (Clinical)", "Patient (Simplified)"])

    st.markdown("---")
    # Reset Button
    if st.button("🗑️ Reset Session", use_container_width=True):
        for key in st.session_state.keys():
            if key != 'accepted_terms':  # Păstrăm consent-ul
                del st.session_state[key]
        st.rerun()

    st.markdown("---")
    with st.expander("ℹ️ About & Methodology"):
        st.info("""
        **System Architecture:**
        1. **NLP Core:** BERT fine-tuned on C2D2 dataset for Cognitive Distortions.
        2. **Emotion AI:** DistilRoBERTa (7 classes: Joy, Sadness, etc.).
        3. **Audio Processing:** Librosa for spectral analysis, Pitch (F0), and Rhythm (WPM).
        4. **Explainability (XAI):** Perturbation-based feature importance.
        """)

    if LOAD_SUCCESS:
        st.success("🟢 Models Online")
    else:
        st.error("🔴 Models Offline")

st.markdown(
    """<div class="main-header"><h1>🧠 CogniSense AI</h1><p>Advanced Cognitive Distortion Analysis & Clinical Profiling System</p></div>""",
    unsafe_allow_html=True)

if not LOAD_SUCCESS: st.stop()

tab1, tab2, tab3 = st.tabs(["⚡ Real-Time Analysis", "📂 Session Profiling", "🚀 Bulk Analysis"])

# ==========================================
# TAB 1: SINGLE THOUGHT ANALYSIS
# ==========================================
with tab1:
    st.markdown("### 🔍 Analyze a single thought")
    col_input, col_viz = st.columns([1, 1])
    with col_input:
        user_input = st.text_area("Patient Statement:", "I failed the test, so I am a complete loser.", height=150)
        analyze_btn = st.button("🔍 Analyze Thought", type="primary", use_container_width=True)

    if analyze_btn:
        with st.spinner("Processing semantics..."):
            dtype, conf, _, bin_conf, bin_class = predict_full_pipeline(user_input)
            st.session_state['last_text'] = user_input
            st.session_state['last_pred'] = dtype

        with col_viz:
            if bin_class == 0:
                st.markdown(
                    f"""<div class="metric-card" style="border-left-color: #2ecc71;"><h3 style="color:#2ecc71">✅ Balanced Thought</h3><small>Confidence: {bin_conf:.2%}</small></div>""",
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f"""<div class="metric-card" style="border-left-color: #e74c3c;"><h3 style="color:#e74c3c">⚠️ Detected: {dtype}</h3><p>Confidence: <b>{conf:.2%}</b></p></div>""",
                    unsafe_allow_html=True)
                if dtype in CBT_STRATEGIES: st.info(CBT_STRATEGIES[dtype])

                with st.spinner("Calculating explainability (XAI)..."):
                    importance = explain_text(user_input, dtype)
                    max_imp = max([s for w, s in importance]) if importance else 0.001
                    html_text = "<p style='line-height:1.6; font-size:16px;'>"
                    for word, impact in importance:
                        if impact > (max_imp * 0.20) and impact > 0.01:
                            html_text += f"<span class='highlight-word' title='Contribution: {impact:.3f}'>{word}</span> "
                        else:
                            html_text += f"{word} "
                    html_text += "</p>"
                    with st.expander("👁️ Explainability Map (Key Triggers)", expanded=True):
                        st.markdown(html_text, unsafe_allow_html=True)
                        st.caption("*Words highlighted in red strongly influenced the AI prediction.*")

    # Feedback
    if st.session_state['last_text']:
        st.markdown("---")
        with st.expander("👩‍⚕️ Expert Feedback Loop (Improve the Model)"):
            c1, c2 = st.columns([1, 3])
            with c1:
                if st.button("👍 Correct Prediction", key="fb_yes"):
                    save_feedback_to_csv(st.session_state['last_text'], st.session_state['last_pred'],
                                         st.session_state['last_pred'], True)
                    st.toast("Feedback saved!", icon="✅")
            with c2:
                actual = st.selectbox("If incorrect, select true label:", ["Non-Distorted"] + label_list_multi,
                                      key="fb_select")
                if st.button("💾 Save Correction", key="fb_no"):
                    save_feedback_to_csv(st.session_state['last_text'], st.session_state['last_pred'], actual, False)
                    st.toast("Correction logged for retraining!", icon="📥")

# ==========================================
# TAB 2: PROFILING (Multimodal)
# ==========================================
with tab2:
    c_left, c_right = st.columns([2, 1])
    with c_left:
        st.markdown("### 📝 Session Transcript")
        status_box = st.empty()

        tab_mic, tab_up = st.tabs(["🎙️ Microphone", "📂 Audio Upload"])

        with tab_mic:
            if st.button("🎙️ Record & Analyze"):
                txt, feats = transcribe_speech_and_analyze(status_box)
                if txt:
                    st.session_state['session_transcript'] = (
                                st.session_state['session_transcript'] + " " + txt).strip()
                    st.session_state['last_audio_file'] = TEMP_AUDIO_FILE
                if feats: st.session_state['audio_features'] = feats
                st.rerun()

        with tab_up:
            up_file = st.file_uploader("Upload Audio (WAV/MP3)", type=["wav", "mp3"])
            if up_file is not None:
                if st.button("Processing File"):
                    txt, feats = process_uploaded_audio(up_file, status_box)
                    if txt:
                        st.session_state['session_transcript'] = (
                                    st.session_state['session_transcript'] + " " + txt).strip()
                        st.session_state['last_audio_file'] = TEMP_AUDIO_FILE
                    if feats: st.session_state['audio_features'] = feats
                    st.rerun()

        session_text = st.text_area("Transcript text:", height=200, key="session_transcript")

        # Audio Player & Spectrogram
        if user_mode.startswith("Doctor") and st.session_state.get('last_audio_file') and os.path.exists(
                st.session_state['last_audio_file']):
            with st.expander("🎧 Audio Playback & Signal Analysis (Clinical)", expanded=False):
                st.audio(st.session_state['last_audio_file'])
                fig = plot_spectrogram(st.session_state['last_audio_file'])
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)

        if st.button("📊 Generate Clinical Report", type="primary", use_container_width=True):
            res, counts, total, scores, emotions = analyze_session(st.session_state['session_transcript'])

            # Calcul Risk Insights
            insights = []
            if scores['depression'] > 3.5 and scores['depression'] > scores['anxiety']:
                insights.append(
                    (f"🔴 **High Risk:** Depressive Pattern (Score: {scores['depression']:.1f})", "#e53935", "white"))
            elif scores['anxiety'] > 3.5 and scores['anxiety'] > scores['depression']:
                insights.append(
                    (f"🟠 **High Risk:** Anxiety Pattern (Score: {scores['anxiety']:.1f})", "#fb8c00", "white"))
            elif scores['depression'] > 2.0 and scores['anxiety'] > 2.0:
                insights.append(("🟡 **Mixed Profile:** Complex distortions detected.", "#fdd835", "black"))
            elif total > 0:
                insights.append(("🔵 **Mild Distortions:** Monitor progress.", "#29b6f6", "white"))
            else:
                insights.append(("✅ **Balanced State:** No significant patterns detected.", "#43a047", "white"))


            narrative = generate_narrative_report(scores, st.session_state.get('audio_features'), emotions)
            st.session_state['narrative_text'] = narrative

            st.session_state['pdf_data'] = (res, counts, insights)
            st.session_state['clinical_scores'] = scores
            st.session_state['emotion_scores'] = emotions

    with c_right:
        st.markdown("### 🏥 Analytics")
        if 'pdf_data' in st.session_state:
            res, counts, insights = st.session_state['pdf_data']

            # Safety Alert
            clin_scores = st.session_state.get('clinical_scores', {'depression': 0, 'anxiety': 0})
            if clin_scores['depression'] > 4.0:
                st.markdown("""
                <div class="safety-banner">
                    ⚠️ CRITICAL RISK DETECTED <br>
                    Contact: 112 / Suicide Prevention
                </div>""", unsafe_allow_html=True)

            if 'narrative_text' in st.session_state and user_mode.startswith("Doctor"):
                st.info(st.session_state['narrative_text'])

            for msg, bg, txt in insights:
                st.markdown(
                    f"<div style='background:{bg}; color:{txt}; padding:15px; border-radius:5px; margin-bottom:10px; text-align:center;'><b>{msg}</b></div>",
                    unsafe_allow_html=True)

            st.markdown("#### ❤️ Emotional Spectrum")
            emotions = st.session_state.get('emotion_scores', {})
            if emotions:
                if user_mode.startswith("Doctor"):
                    df_emo = pd.DataFrame(list(emotions.items()), columns=['Emotion', 'Score'])
                    fig_emo = px.bar(df_emo, x='Score', y='Emotion', orientation='h', range_x=[0, 1], color='Score',
                                     color_continuous_scale='Bluered')
                    fig_emo.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(fig_emo, use_container_width=True)
                else:
                    top_emo = max(emotions, key=emotions.get)
                    st.success(f"Dominant Mood: **{top_emo.upper()}** ({emotions[top_emo]:.0%})")

            st.markdown("---")
            if user_mode.startswith("Doctor"):
                m1, m2 = st.columns(2)
                m1.metric("Depression Risk", f"{clin_scores['depression']:.1f}",
                          delta="High" if clin_scores['depression'] > 3.5 else "Stable", delta_color="inverse")
                m2.metric("Anxiety Risk", f"{clin_scores['anxiety']:.1f}",
                          delta="High" if clin_scores['anxiety'] > 3.5 else "Stable", delta_color="inverse")

                if 'audio_features' in st.session_state and st.session_state['audio_features']:
                    f = st.session_state['audio_features']
                    st.markdown("---")
                    a1, a2, a3 = st.columns(3)
                    wpm_val = int(f['wpm'])
                    w_label = "Low" if wpm_val < 110 else "High" if wpm_val > 160 else "Normal"
                    w_col = "inverse" if wpm_val < 110 or wpm_val > 160 else "normal"
                    a1.metric("Rate", f"{wpm_val} WPM", delta=w_label, delta_color=w_col)
                    a2.metric("Silence", f"{int(f['silence_ratio'])}%",
                              delta="High" if f['silence_ratio'] > 35 else "Ok", delta_color="inverse")
                    a3.metric("Pitch Var", f"{f.get('pitch_std', 0):.1f}",
                              delta="Flat" if f.get('pitch_std', 0) < 15 else "Ok")

                st.markdown("<div class='centered-content'>", unsafe_allow_html=True)
                if sum(counts.values()) > 0:
                    df_r = pd.DataFrame(dict(r=list(counts.values()), theta=list(counts.keys())))
                    fig = px.line_polar(df_r, r='r', theta='theta', line_close=True, template="plotly_white")
                    fig.update_traces(fill='toself', line_color='#4ca1af')
                    fig.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20))

                    try:
                        fig.write_image(TEMP_RADAR_FILE)
                    except:
                        pass

                    st.plotly_chart(fig, use_container_width=True)

            st.markdown(create_download_link(
                res,
                counts,
                insights,
                st.session_state.get('audio_features'),
                emotions,
                st.session_state.get('narrative_text', "")
            ), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            if user_mode.startswith("Patient"):
                st.markdown("#### 💡 Self-Help Tips")
            else:
                st.markdown("#### 💡 Interventions")

            found_tips = False
            for dtype, count in counts.items():
                if count > 0 and dtype in CBT_STRATEGIES:
                    st.success(f"**{dtype}:** {CBT_STRATEGIES[dtype]}")
                    found_tips = True
            if not found_tips: st.info("No specific distortions detected.")

    if 'pdf_data' in st.session_state and user_mode.startswith("Doctor"):
        st.markdown("---")
        st.markdown("### 🔍 Sentence Breakdown")
        res_data, _, _ = st.session_state['pdf_data']
        for item in res_data:
            if item['type'] == "Neutral":
                st.markdown(f"⚪ {item['text']}")
            else:
                st.markdown(f"🔴 **[{item['type']}]** ({item['confidence']:.2f}): {item['text']}")

    if user_mode.startswith("Doctor"):
        with st.expander("📘 Clinical Guide: Models & Metrics"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Cognitive Distortions")
                for k, v in DISTORTION_DEFINITIONS.items():
                    st.markdown(f"**{k}:** {v}")
            with c2:
                st.markdown("#### Decision Logic & Biomarkers")
                st.markdown("""
                * **Emotion AI:** RoBERTa model classifying 7 basic emotions (Ekman's Model).
                * **Speech Rate:** <110 WPM (Psychomotor Retardation) vs >160 WPM (Agitation).
                * **Pitch Std:** <15 Hz indicates flat affect (Monotony), a negative symptom of depression.
                """)

# ========================
# TAB 3: BULK PROCESSING
# ========================
with tab3:
    st.markdown("### 🚀 Batch Processing")
    f = st.file_uploader("Upload CSV", type=["csv"])
    if f:
        df = pd.read_csv(f)
        if 'text' in df.columns and st.button("Start Processing"):
            res = []
            distorted_text_corpus = ""
            bar = st.progress(0)

            for i, r in df.iterrows():
                dtype, conf, _, bin_conf, bin_class = predict_full_pipeline(str(r['text']))
                screen_res = "Distorted" if bin_class == 1 else "Non-Distorted"
                clinical_res = dtype if bin_class == 1 else "-"

                if bin_class == 1:
                    distorted_text_corpus += " " + str(r['text'])

                res.append({
                    "Text": r['text'],
                    "Screening Result": screen_res,
                    "Screening Confidence": f"{bin_conf:.2f}",
                    "Clinical Type": clinical_res,
                    "Type Confidence": f"{conf:.2f}" if bin_class == 1 else "-"
                })
                bar.progress((i + 1) / len(df))

            df_res = pd.DataFrame(res)
            st.dataframe(df_res)

            if distorted_text_corpus.strip():
                st.markdown("### ☁️ Common Distortion Themes (Word Cloud)")
                wc = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(
                    distorted_text_corpus)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("No distortions found to generate Word Cloud.")

            csv_data = df_res.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Download Results CSV", data=csv_data, file_name="bulk_analysis_results.csv",
                               mime="text/csv")

st.markdown("""<div class="footer-disclaimer">⚠️ ACADEMIC PROTOTYPE. NOT A MEDICAL DEVICE.</div>""",
            unsafe_allow_html=True)