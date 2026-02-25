import pandas as pd
import streamlit as st
import numpy as np
import pickle
import joblib
from PIL import Image

st.set_page_config(page_title="Drug Recommendation", page_icon="💊", layout="wide")

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("<h2 style='color: #ffffff;'>📌 Description</h2>", unsafe_allow_html=True)
st.sidebar.image("utils/ph4.png", use_container_width=True)
st.sidebar.markdown("<p class='sidebar-text'>Our AI-powered Drug Recommendation System uses NLP and cosine similarity to analyze medicines and recommend the most relevant alternatives, ensuring accurate, data-driven, and personalized treatment options.</p>", unsafe_allow_html=True)

# ── Load models ────────────────────────────────────────────────────────────────
@st.cache_resource()
def load_models():
    with open('models/second_feature_models/medicine_dict.pkl', 'rb') as file:
        medicine_dict = pickle.load(file)
    similarity = joblib.load('models/second_feature_models/similarity.joblib')
    return pd.DataFrame(medicine_dict), similarity

@st.cache_resource()
def load_description_data():
    return pd.read_csv('data/Drug reccomendation/medicine.csv')

medicines, similarity = load_models()
description_data = load_description_data()

@st.cache_data()
def recommend(medicine):
    try:
        medicine_index = medicines[medicines['Drug_Name'] == medicine].index[0]
    except IndexError:
        return []
    distances = similarity[medicine_index]
    medicines_list = sorted(enumerate(distances), reverse=True, key=lambda x: x[1])[1:6]
    return [medicines.iloc[i[0]].Drug_Name for i in medicines_list]

# ══════════════════════════════════════════════════════════════════════════════
# HEADER BANNER
# ══════════════════════════════════════════════════════════════════════════════
hdr_img, hdr_txt = st.columns([1, 3])
with hdr_img:
    st.image("utils/medss.png", use_container_width=True)
with hdr_txt:
    st.markdown("""
    <div style='padding:18px 0 0 14px;'>
        <h1 style='color:#1565c0; margin-bottom:6px;'>💊 Drug Recommendation System</h1>
        <p style='font-size:17px; color:#555;'>
            AI-powered alternative medicine finder using NLP &amp; Cosine Similarity.
        </p>
        <div style='display:flex; gap:10px; flex-wrap:wrap; margin-top:12px;'>
            <span style='background:#e3f2fd;color:#1565c0;padding:5px 14px;border-radius:20px;font-size:13px;font-weight:600;border:1px solid #bbdefb;'>🧠 NLP-Powered</span>
            <span style='background:#e8f5e9;color:#2e7d32;padding:5px 14px;border-radius:20px;font-size:13px;font-weight:600;border:1px solid #c8e6c9;'>📐 Cosine Similarity</span>
            <span style='background:#fff3e0;color:#e65100;padding:5px 14px;border-radius:20px;font-size:13px;font-weight:600;border:1px solid #ffe0b2;'>💊 10,000+ Medicines</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SEARCH SECTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<h3 style='color:#1565c0;'>🔍 Find Similar Drugs</h3>", unsafe_allow_html=True)

src_col, btn_col = st.columns([4, 1])
with src_col:
    selected_medicine_name = st.selectbox(
        "💊 Select a medicine:",
        sorted(medicines['Drug_Name'].values),
        label_visibility="collapsed"
    )
with btn_col:
    recommend_btn = st.button("🔍 Recommend Drug", use_container_width=True)

# ── Drug Description Card ──────────────────────────────────────────────────────
desc = description_data.loc[description_data['Drug_Name'] == selected_medicine_name, 'Description']
if not desc.empty:
    st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, #e3f2fd, #f3e5f5);
        border-left: 5px solid #1565c0;
        border-radius: 12px;
        padding: 18px 22px;
        margin: 14px 0;
    '>
        <span style='font-size:13px; color:#1565c0; font-weight:700; letter-spacing:1px; text-transform:uppercase;'>📋 About this Medicine</span>
        <p style='font-size:15px; color:#1a237e; margin:8px 0 0 0;'>{desc.values[0]}</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
if recommend_btn:
    with st.spinner("🔬 Finding best alternatives..."):
        recommendations = recommend(selected_medicine_name)

    if recommendations:
        st.markdown("<h3 style='color:#1565c0;'>📌 Top 5 Recommended Alternatives</h3>", unsafe_allow_html=True)

        # Color palette for the 5 cards
        card_colors = [
            ("#e3f2fd", "#1565c0", "#bbdefb"),   # blue
            ("#e8f5e9", "#2e7d32", "#c8e6c9"),   # green
            ("#fff3e0", "#e65100", "#ffe0b2"),   # orange
            ("#f3e5f5", "#6a1b9a", "#e1bee7"),   # purple
            ("#fce4ec", "#c62828", "#f48fb1"),   # pink
        ]
        rank_emojis = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]

        for i, drug in enumerate(recommendations):
            bg, accent, border = card_colors[i % len(card_colors)]
            buy_link = f"https://pharmeasy.in/search/all?name={drug}"

            st.markdown(f"""
            <div style='
                background: {bg};
                border: 1.5px solid {border};
                border-left: 6px solid {accent};
                border-radius: 14px;
                padding: 16px 22px;
                margin-bottom: 12px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                box-shadow: 0 2px 10px rgba(0,0,0,0.06);
            '>
                <div style='display:flex; align-items:center; gap:14px;'>
                    <span style='font-size:28px;'>{rank_emojis[i]}</span>
                    <div>
                        <div style='font-size:11px; color:{accent}; font-weight:700; letter-spacing:1px; text-transform:uppercase;'>Alternative #{i+1}</div>
                        <div style='font-size:19px; font-weight:700; color:{accent};'>{drug}</div>
                    </div>
                </div>
                <a href='{buy_link}' target='_blank' style='
                    background:{accent};
                    color:white;
                    padding:9px 20px;
                    border-radius:10px;
                    text-decoration:none;
                    font-size:14px;
                    font-weight:600;
                    white-space:nowrap;
                '>🛒 Buy Now</a>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.error("❌ No recommendations found. Please select another drug.")

# ── Quick Tips ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 💡 Medicine Safety Tips")
t1, t2, t3, t4 = st.columns(4)
with t1: st.info("👨‍⚕️ **Consult First**\nAlways consult your doctor before switching medicines.")
with t2: st.info("📋 **Read Labels**\nCheck dosage instructions and expiry dates carefully.")
with t3: st.info("⚠️ **Check Allergies**\nInform your doctor of any known drug allergies.")
with t4: st.info("🏥 **Buy Authentic**\nPurchase only from licensed pharmacies or trusted platforms.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style='text-align:center; color:#aaa; font-size:13px;'>
    Made with ❤️ by <span style='color:#e65100; font-weight:600;'>Abhay</span> &nbsp;|&nbsp;
    <a href='https://github.com/AbhaySingh71' style='color:#1565c0; text-decoration:none;'>GitHub</a>
    &nbsp;© 2025
</p>
""", unsafe_allow_html=True)
