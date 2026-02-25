import random
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from googletrans import Translator

st.set_page_config(page_title="Medibot", page_icon="🤖", layout="wide")

# ── Load data & models ─────────────────────────────────────────────────────────
df = pd.read_csv(r'C:\Users\yaswi\OneDrive\Pictures\Desktop\Documents\PROJECTS\MAJOR projects\AI-Diseases Prediction\AI-Diseases Prediction\pages\dataset - Sheet1.csv')
model = SentenceTransformer('all-MiniLM-L6-v2')
translator = Translator()

# ── Medical keywords fallback ──────────────────────────────────────────────────
medical_keywords = {
    "fever":    "It sounds like you may have a fever. Stay hydrated and consider seeing a doctor if symptoms persist.",
    "cough":    "A persistent cough might be due to an infection or allergy. Try warm fluids and rest.",
    "headache": "Headaches can have many causes, including stress and dehydration. Consider resting and drinking water.",
    "cold":     "Common colds usually go away on their own. Stay warm, drink fluids, and get rest.",
}

# ── Health tips ────────────────────────────────────────────────────────────────
health_tips = {
    "sleep":   ["Try to get at least 7-8 hours of sleep each night.",
                "Establish a regular sleep routine to improve sleep quality.",
                "Avoid screens before bed to help your mind relax."],
    "energy":  ["Make sure you're eating a balanced diet to maintain energy.",
                "Exercise regularly to boost your energy levels.",
                "Stay hydrated throughout the day to avoid fatigue."],
    "stress":  ["Take short breaks throughout the day to reduce stress.",
                "Practice mindfulness or meditation to help manage stress.",
                "Engage in physical activity to reduce anxiety and stress."],
    "general": ["Drink plenty of water throughout the day.",
                "Get at least 30 minutes of exercise every day.",
                "Eat a balanced diet rich in fruits and vegetables."],
}

def get_personalized_health_tip(user_input):
    u = user_input.lower()
    if "tired" in u or "fatigue" in u:
        return random.choice(health_tips["energy"])
    elif "sleep" in u or "rest" in u:
        return random.choice(health_tips["sleep"])
    elif "stress" in u or "anxious" in u:
        return random.choice(health_tips["stress"])
    return random.choice(health_tips["general"])

def find_best_cure(user_input):
    user_emb = model.encode(user_input, convert_to_tensor=True)
    disease_emb = model.encode(df['disease'].tolist(), convert_to_tensor=True)
    sims = util.pytorch_cos_sim(user_emb, disease_emb)[0]
    idx  = sims.argmax().item()
    score = sims[idx].item()
    if score < 0.5:
        for kw, resp in medical_keywords.items():
            if kw in user_input.lower():
                return resp
        return "I'm sorry, I don't have enough information on this. Please consult a healthcare professional."
    return df.iloc[idx]['cure']

def translate_text(text, dest_language='en'):
    return translator.translate(text, dest=dest_language).text

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🤖 Medibot — AI Medical Assistant")
st.write("Ask any health-related question and get AI-powered suggestions in your language.")
st.markdown("---")

# Input row
col1, col2 = st.columns([3, 1])

with col1:
    user_input = st.text_input(
        "💬 Ask a health question",
        placeholder="Eg: I have a fever and sore throat, what should I do?"
    )

with col2:
    language_choice = st.selectbox("🌐 Response Language", [
        "English", "Hindi", "Gujarati", "Korean", "Turkish",
        "German", "French", "Arabic", "Urdu", "Tamil", "Telugu",
        "Chinese", "Japanese"
    ])

language_codes = {
    "English": "en", "Hindi": "hi", "Gujarati": "gu", "Korean": "ko",
    "Turkish": "tr", "German": "de", "French": "fr", "Arabic": "ar",
    "Urdu": "ur", "Tamil": "ta", "Telugu": "te",
    "Chinese": "zh-CN", "Japanese": "ja",
}

st.markdown("")

# Action buttons
btn_col1, btn_col2, _ = st.columns([1, 1, 2])

with btn_col1:
    get_response = st.button("🩺 Get Medical Suggestion", use_container_width=True)

with btn_col2:
    get_tip = st.button("💡 Get Health Tip", use_container_width=True)

st.markdown("---")

# ── Responses ──────────────────────────────────────────────────────────────────
if get_response:
    if not user_input.strip():
        st.warning("⚠️ Please type a health question first.")
    else:
        with st.spinner("🔍 Analyzing your symptoms..."):
            response = find_best_cure(user_input)
            dest_lang = language_codes[language_choice]
            translated = translate_text(response, dest_language=dest_lang)

        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
                border-left: 5px solid #43a047;
                border-radius: 12px;
                padding: 20px 24px;
                margin-top: 8px;
            ">
                <h4 style="color:#2e7d32; margin-bottom:8px;">🩺 Medical Suggestion</h4>
                <p style="font-size:16px; color:#1b5e20; margin:0;">{translated}</p>
            </div>
            <p style="color:#888; font-size:12px; margin-top:8px;">
                ⚕️ AI-generated suggestion. Always consult a certified healthcare professional for medical advice.
            </p>
            """,
            unsafe_allow_html=True
        )

if get_tip:
    if not user_input.strip():
        st.warning("⚠️ Please type a question or keyword first (e.g. 'sleep', 'stress', 'tired').")
    else:
        with st.spinner("💡 Generating your personalized tip..."):
            tip = get_personalized_health_tip(user_input)
            dest_lang = language_codes[language_choice]
            translated_tip = translate_text(tip, dest_language=dest_lang)

        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #e3f2fd, #ede7f6);
                border-left: 5px solid #4285F4;
                border-radius: 12px;
                padding: 20px 24px;
                margin-top: 8px;
            ">
                <h4 style="color:#1565c0; margin-bottom:8px;">💡 Personalized Health Tip</h4>
                <p style="font-size:16px; color:#0d47a1; margin:0;">{translated_tip}</p>
            </div>
            <p style="color:#888; font-size:12px; margin-top:8px;">
                🌐 Translation powered by AI — may not be 100% accurate.
            </p>
            """,
            unsafe_allow_html=True
        )

# ── Quick Health Info Cards ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🏥 Quick Health Reminders")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.info("💧 **Stay Hydrated**\nDrink 8+ glasses of water daily.")
with c2:
    st.info("🏃 **Stay Active**\n30 min of exercise every day.")
with c3:
    st.info("😴 **Sleep Well**\n7–8 hours of quality sleep.")
with c4:
    st.info("🥦 **Eat Healthy**\nFruits, vegetables & whole grains.")
