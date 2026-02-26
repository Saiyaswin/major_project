import pandas as pd
import streamlit as st
import numpy as np
import pickle
import joblib
from thefuzz import process

st.set_page_config(page_title="Drug Recommendation", page_icon="💊", layout="wide")

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("<h2 style='color: #ffffff;'>📌 Description</h2>", unsafe_allow_html=True)
st.sidebar.image("utils/ph4.png", use_container_width=True)
st.sidebar.markdown("<p class='sidebar-text'>Our AI-powered Drug Recommendation System uses NLP and cosine similarity to analyze medicines and recommend the most relevant drugs for your disease, symptoms, or as alternatives to an existing medication.</p>", unsafe_allow_html=True)

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

# ── Reason→Disease mapping ─────────────────────────────────────────────────────
# Maps common lay-terms/disease names to the 'Reason' values in medicine.csv
CONDITION_ALIASES = {
    # Reason value      : [aliases user might type]
    'Acne'             : ['acne', 'pimple', 'pimples', 'spots', 'blackheads'],
    'Adhd'             : ['adhd', 'attention deficit', 'hyperactivity'],
    'Allergies'        : ['allergy', 'allergies', 'allergic', 'sneezing', 'hives', 'itching', 'rash'],
    'Alzheimer'        : ['alzheimer', 'dementia', 'memory loss'],
    'Amoebiasis'       : ['amoebiasis', 'amoeba', 'diarrhoea', 'diarrhea', 'stomach infection'],
    'Anaemia'          : ['anaemia', 'anemia', 'low haemoglobin', 'iron deficiency'],
    'Angina'           : ['angina', 'chest pain', 'chest tightness'],
    'Anxiety'          : ['anxiety', 'stress', 'panic', 'nervousness'],
    'Appetite'         : ['appetite', 'weight gain', 'underweight', 'loss of appetite'],
    'Arrhythmiasis'    : ['arrhythmia', 'irregular heartbeat', 'palpitations'],
    'Arthritis'        : ['arthritis', 'joint pain', 'joint swelling', 'rheumatoid', 'osteoarthritis'],
    'Cleanser'         : ['cleanser', 'skin cleanser', 'face wash'],
    'Constipation'     : ['constipation', 'bowel', 'laxative'],
    'Contraception'    : ['contraception', 'birth control', 'contraceptive'],
    'Dandruff'         : ['dandruff', 'scalp', 'flaky scalp'],
    'Depression'       : ['depression', 'sadness', 'low mood', 'depressed'],
    'Diabetes'         : ['diabetes', 'blood sugar', 'high sugar', 'diabetic', 'hyperglycemia'],
    'Diarrhoea'        : ['diarrhoea', 'diarrhea', 'loose stools', 'loose motion'],
    'Digestion'        : ['digestion', 'indigestion', 'bloating', 'gas', 'gastric', 'acidity', 'gerd', 'acid reflux'],
    'Fever'            : ['fever', 'high fever', 'temperature', 'pyrexia', 'cold', 'common cold', 'flu', 'influenza'],
    'Fungal'           : ['fungal', 'fungal infection', 'ringworm', 'candida', 'athletes foot', 'tinea'],
    'General'          : ['general', 'multivitamin', 'vitamin', 'supplement health'],
    'Glaucoma'         : ['glaucoma', 'eye pressure', 'intraocular pressure'],
    'Gout'             : ['gout', 'uric acid', 'gout attack'],
    'Haematopoiesis'   : ['haematopoiesis', 'blood cell', 'bone marrow'],
    'Haemorrhoid'      : ['haemorrhoid', 'hemorrhoid', 'piles'],
    'Hyperpigmentation': ['hyperpigmentation', 'dark spots', 'skin darkening', 'melasma', 'pigmentation'],
    'Hypertension'     : ['hypertension', 'high blood pressure', 'hbp', 'blood pressure'],
    'Hyperthyroidism'  : ['hyperthyroidism', 'overactive thyroid', 'high thyroid'],
    'Hypnosis'         : ['insomnia', 'sleep', 'sleeplessness', 'hypnotic', 'sedative'],
    'Hypotension'      : ['hypotension', 'low blood pressure'],
    'Hypothyroidism'   : ['hypothyroidism', 'underactive thyroid', 'low thyroid'],
    'Infection'        : ['infection', 'bacterial', 'antibiotic', 'pneumonia', 'typhoid', 'dengue', 'malaria', 'uti', 'urinary tract infection'],
    'Malarial'         : ['malaria', 'malarial', 'plasmodium', 'antimalarial'],
    'Migraine'         : ['migraine', 'headache', 'severe headache', 'cluster headache'],
    'Mydriasis'        : ['mydriasis', 'pupil dilation'],
    'Osteoporosis'     : ['osteoporosis', 'bone loss', 'weak bones', 'bone density'],
    'Pain'             : ['pain', 'painkiller', 'pain relief', 'analgesic', 'muscle pain', 'back pain', 'neck pain'],
    'Parkinson'        : ['parkinson', 'tremor', 'parkinsons'],
    'Psychosis'        : ['psychosis', 'schizophrenia', 'hallucination', 'antipsychotic'],
    'Pyrexia'          : ['pyrexia', 'fever reducer', 'antipyretic', 'paracetamol'],
    'Scabies'          : ['scabies', 'mites', 'skin mites', 'itchy skin'],
    'Schizophrenia'    : ['schizophrenia', 'psychosis', 'antipsychotic'],
    'Smoking'          : ['smoking', 'nicotine', 'quit smoking', 'tobacco'],
    'Supplement'       : ['supplement', 'vitamin', 'mineral', 'multivitamin', 'omega', 'calcium'],
    'Thrombolysis'     : ['thrombolysis', 'blood clot', 'anticoagulant', 'clot'],
    'Vaccines'         : ['vaccine', 'vaccination', 'immunization', 'immunisation'],
    'Vertigo'          : ['vertigo', 'dizziness', 'spinning', 'balance'],
    'Viral'            : ['viral', 'virus', 'antiviral', 'cold', 'flu', 'influenza', 'covid'],
    'Wound'            : ['wound', 'cut', 'burn', 'antiseptic', 'injury'],
}

# Build reverse map: alias -> Reason
ALIAS_TO_REASON = {}
for reason, aliases in CONDITION_ALIASES.items():
    for alias in aliases:
        ALIAS_TO_REASON[alias.lower()] = reason

ALL_REASONS = sorted(description_data['Reason'].dropna().unique().tolist())

def find_condition_from_input(user_text):
    """Map user text to the best matching Reason category."""
    text_lower = user_text.lower().strip()
    # Exact alias match
    if text_lower in ALIAS_TO_REASON:
        return ALIAS_TO_REASON[text_lower]
    # Direct reason match (case-insensitive)
    for r in ALL_REASONS:
        if r.lower() == text_lower or r.lower() in text_lower or text_lower in r.lower():
            return r
    # Fuzzy match against all aliases + reasons
    all_terms = list(ALIAS_TO_REASON.keys()) + [r.lower() for r in ALL_REASONS]
    match, score = process.extractOne(text_lower, all_terms)
    if score >= 65:
        # Resolve alias to reason
        if match in ALIAS_TO_REASON:
            return ALIAS_TO_REASON[match]
        # It matched a reason directly
        for r in ALL_REASONS:
            if r.lower() == match:
                return r
    return None

def recommend_by_condition(condition_reason, top_n=8):
    """Return top N drugs for the given condition, sorted by cosine similarity within the group."""
    subset = description_data[description_data['Reason'] == condition_reason]
    if subset.empty:
        return []
    # Get their indices in medicines df
    drug_names_in_condition = subset['Drug_Name'].tolist()
    # Find which rows in medicines match
    cond_indices = medicines[medicines['Drug_Name'].isin(drug_names_in_condition)].index.tolist()
    if not cond_indices:
        return drug_names_in_condition[:top_n]
    # Use centroid of the condition drugs to rank all condition drugs
    # Get average similarity among themselves: pick top_n by highest avg similarity
    sim_matrix_subset = similarity[np.ix_(cond_indices, cond_indices)]
    avg_sim = sim_matrix_subset.mean(axis=1)
    ranked = sorted(zip(cond_indices, avg_sim), key=lambda x: x[1], reverse=True)
    results = []
    for idx, score in ranked[:top_n]:
        drug_name = medicines.iloc[idx]['Drug_Name']
        results.append(drug_name)
    return results

@st.cache_data()
def recommend_similar(medicine, top_n=5):
    """Find similar alternative drugs for a given medicine."""
    try:
        medicine_index = medicines[medicines['Drug_Name'] == medicine].index[0]
    except IndexError:
        return []
    distances = similarity[medicine_index]
    medicines_list = sorted(enumerate(distances), reverse=True, key=lambda x: x[1])[1:top_n+1]
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
            AI-powered drug finder — search by disease/symptom or find alternatives to an existing medicine.
        </p>
        <div style='display:flex; gap:10px; flex-wrap:wrap; margin-top:12px;'>
            <span style='background:#e3f2fd;color:#1565c0;padding:5px 14px;border-radius:20px;font-size:13px;font-weight:600;border:1px solid #bbdefb;'>🧠 NLP-Powered</span>
            <span style='background:#e8f5e9;color:#2e7d32;padding:5px 14px;border-radius:20px;font-size:13px;font-weight:600;border:1px solid #c8e6c9;'>📐 Cosine Similarity</span>
            <span style='background:#fff3e0;color:#e65100;padding:5px 14px;border-radius:20px;font-size:13px;font-weight:600;border:1px solid #ffe0b2;'>💊 10,000+ Medicines</span>
            <span style='background:#f3e5f5;color:#6a1b9a;padding:5px 14px;border-radius:20px;font-size:13px;font-weight:600;border:1px solid #e1bee7;'>🏥 50 Disease Categories</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# TAB LAYOUT: Disease/Symptom Search | Medicine Alternatives
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["🏥 Recommend Drugs by Disease / Symptom", "🔄 Find Alternative Medicines"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: DISEASE / SYMPTOM → DRUG
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("<h3 style='color:#1565c0;'>🔍 Find Drugs for Your Disease or Symptom</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#e3f2fd; border-left:5px solid #1e88e5; border-radius:10px; padding:12px 18px; margin-bottom:14px;'>
        <span style='color:#1565c0; font-size:14px;'>💡 <b>Tip:</b> Type your disease (e.g. <i>Diabetes, Fever, Arthritis</i>) or symptom
        (e.g. <i>headache, itching, pain, dizziness</i>) and click <b>Get Drugs</b>.</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Custom CSS for buttons ─────────────────────────────────────────────────
    st.markdown("""
    <style>
    /* ── Get Drugs button ── */
    div[data-testid="column"]:has(button[kind="secondary"]) button[kind="secondary"] {
        all: unset;
    }
    button[kind="secondary"] { border-radius: 50px !important; }

    /* Main Get-Drugs button */
    div[data-testid="stButton"] > button[data-testid="baseButton-secondary"]:has-text {
        border-radius: 50px !important;
    }
    /* Style the Get Drugs button by key */
    [data-testid="stButton"] button {
        border-radius: 30px !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stButton"] button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 18px rgba(0,0,0,0.15) !important;
    }
    /* Quick-condition buttons: each gets a unique gradient via nth-child */
    /* Row 1 */
    [data-testid="stButton"]:has(button[aria-label="Fever"]) button,
    button[key="quick_Fever"] { background: linear-gradient(135deg,#ff6f61,#ff3d00) !important; color:white !important; border:none !important; }
    [data-testid="stButton"]:has(button[aria-label="Diabetes"]) button,
    button[key="quick_Diabetes"] { background: linear-gradient(135deg,#29b6f6,#0277bd) !important; color:white !important; border:none !important; }
    [data-testid="stButton"]:has(button[aria-label="Hypertension"]) button,
    button[key="quick_Hypertension"] { background: linear-gradient(135deg,#ef5350,#b71c1c) !important; color:white !important; border:none !important; }
    [data-testid="stButton"]:has(button[aria-label="Pain"]) button,
    button[key="quick_Pain"] { background: linear-gradient(135deg,#ab47bc,#6a1b9a) !important; color:white !important; border:none !important; }
    [data-testid="stButton"]:has(button[aria-label="Infection"]) button,
    button[key="quick_Infection"] { background: linear-gradient(135deg,#26a69a,#00695c) !important; color:white !important; border:none !important; }
    /* Row 2 */
    [data-testid="stButton"]:has(button[aria-label="Allergies"]) button,
    button[key="quick_Allergies"] { background: linear-gradient(135deg,#ffb74d,#e65100) !important; color:white !important; border:none !important; }
    [data-testid="stButton"]:has(button[aria-label="Arthritis"]) button,
    button[key="quick_Arthritis"] { background: linear-gradient(135deg,#7986cb,#283593) !important; color:white !important; border:none !important; }
    [data-testid="stButton"]:has(button[aria-label="Migraine"]) button,
    button[key="quick_Migraine"] { background: linear-gradient(135deg,#f06292,#880e4f) !important; color:white !important; border:none !important; }
    [data-testid="stButton"]:has(button[aria-label="Fungal"]) button,
    button[key="quick_Fungal"] { background: linear-gradient(135deg,#66bb6a,#1b5e20) !important; color:white !important; border:none !important; }
    [data-testid="stButton"]:has(button[aria-label="Anxiety"]) button,
    button[key="quick_Anxiety"] { background: linear-gradient(135deg,#ffa726,#e65100) !important; color:white !important; border:none !important; }
    /* Row 3 */
    [data-testid="stButton"]:has(button[aria-label="Digestion"]) button,
    button[key="quick_Digestion"] { background: linear-gradient(135deg,#4db6ac,#004d40) !important; color:white !important; border:none !important; }
    [data-testid="stButton"]:has(button[aria-label="Constipation"]) button,
    button[key="quick_Constipation"] { background: linear-gradient(135deg,#a1887f,#4e342e) !important; color:white !important; border:none !important; }
    [data-testid="stButton"]:has(button[aria-label="Anaemia"]) button,
    button[key="quick_Anaemia"] { background: linear-gradient(135deg,#e57373,#c62828) !important; color:white !important; border:none !important; }
    [data-testid="stButton"]:has(button[aria-label="Hypothyroidism"]) button,
    button[key="quick_Hypothyroidism"] { background: linear-gradient(135deg,#4fc3f7,#01579b) !important; color:white !important; border:none !important; }
    [data-testid="stButton"]:has(button[aria-label="Acne"]) button,
    button[key="quick_Acne"] { background: linear-gradient(135deg,#ffcc80,#bf360c) !important; color:white !important; border:none !important; }
    </style>
    """, unsafe_allow_html=True)

    inp_col, btn_col = st.columns([4, 1])
    with inp_col:
        disease_input = st.text_input(
            "Disease/Symptom input",
            placeholder="e.g. Fever, Diabetes, headache, joint pain, fungal infection ...",
            label_visibility="collapsed",
            key="disease_text_input"
        )
    with btn_col:
        get_drugs_btn = st.button("💊 Get Drugs", use_container_width=True)

    # Also show quick condition buttons
    st.markdown("""
    <div style='font-size:13px; font-weight:600; color:#555; margin:12px 0 8px 0;
                letter-spacing:0.5px; text-transform:uppercase;'>
        ⚡ Quick Select a Condition
    </div>
    """, unsafe_allow_html=True)

    # Emoji map per condition
    COND_EMOJI = {
        'Fever': '🌡️', 'Diabetes': '🩸', 'Hypertension': '❤️', 'Pain': '💢',
        'Infection': '🦠', 'Allergies': '🤧', 'Arthritis': '🦴', 'Migraine': '🧠',
        'Fungal': '🍄', 'Anxiety': '😟', 'Digestion': '🫀', 'Constipation': '🌿',
        'Anaemia': '🩸', 'Hypothyroidism': '🔬', 'Acne': '✨'
    }

    quick_conditions = ['Fever', 'Diabetes', 'Hypertension', 'Pain', 'Infection',
                        'Allergies', 'Arthritis', 'Migraine', 'Fungal', 'Anxiety',
                        'Digestion', 'Constipation', 'Anaemia', 'Hypothyroidism', 'Acne']
    q_cols = st.columns(5)
    quick_selected = None
    for i, cond in enumerate(quick_conditions):
        emoji = COND_EMOJI.get(cond, '💊')
        with q_cols[i % 5]:
            if st.button(f"{emoji} {cond}", key=f"quick_{cond}", use_container_width=True):
                quick_selected = cond

    # Determine active input:
    # Quick button click always takes priority over the text box.
    # Get Drugs button uses the text box.
    if quick_selected:
        active_input = quick_selected
    elif get_drugs_btn:
        active_input = disease_input.strip()
    else:
        active_input = ""

    if active_input:
        matched_reason = find_condition_from_input(active_input)
        if matched_reason:
            st.markdown(f"""
            <div style='background:#e8f5e9; border-left:5px solid #43a047; border-radius:10px;
                        padding:12px 18px; margin:12px 0;'>
                <span style='color:#2e7d32; font-size:14px;'>
                    ✅ <b>"{active_input}"</b> matched condition: <b>{matched_reason}</b>
                </span>
            </div>
            """, unsafe_allow_html=True)

            with st.spinner("🔬 Finding best drugs for this condition..."):
                drug_results = recommend_by_condition(matched_reason, top_n=8)

            if drug_results:
                st.markdown(f"<h4 style='color:#1565c0; margin-top:16px;'>💊 Top Recommended Drugs for <span style='color:#2e7d32;'>{matched_reason}</span></h4>", unsafe_allow_html=True)

                card_colors = [
                    ("#e3f2fd", "#1565c0", "#bbdefb"),
                    ("#e8f5e9", "#2e7d32", "#c8e6c9"),
                    ("#fff3e0", "#e65100", "#ffe0b2"),
                    ("#f3e5f5", "#6a1b9a", "#e1bee7"),
                    ("#fce4ec", "#c62828", "#f48fb1"),
                    ("#e0f2f1", "#00695c", "#b2dfdb"),
                    ("#fff8e1", "#f57f17", "#ffecb3"),
                    ("#ede7f6", "#4527a0", "#d1c4e9"),
                ]
                rank_labels = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"]
                rank_emojis = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣"]

                c1, c2 = st.columns(2)
                for i, drug in enumerate(drug_results):
                    bg, accent, border = card_colors[i % len(card_colors)]
                    buy_link = f"https://pharmeasy.in/search/all?name={drug}"
                    # Get description
                    desc_row = description_data[description_data['Drug_Name'] == drug]
                    desc_txt = desc_row['Description'].values[0] if not desc_row.empty else ""
                    card_html = f"""
                    <div style='background:{bg}; border:1.5px solid {border}; border-left:6px solid {accent};
                                border-radius:14px; padding:14px 18px; margin-bottom:12px;
                                box-shadow:0 2px 10px rgba(0,0,0,0.06);'>
                        <div style='display:flex; justify-content:space-between; align-items:flex-start;'>
                            <div style='display:flex; align-items:center; gap:10px;'>
                                <span style='font-size:24px;'>{rank_emojis[i]}</span>
                                <div>
                                    <div style='font-size:10px; color:{accent}; font-weight:700;
                                                letter-spacing:1px; text-transform:uppercase;'>
                                        Rank {rank_labels[i]} &nbsp;•&nbsp; {matched_reason}
                                    </div>
                                    <div style='font-size:17px; font-weight:700; color:{accent}; margin-top:2px;'>{drug}</div>
                                </div>
                            </div>
                            <a href='{buy_link}' target='_blank' style='background:{accent}; color:white;
                                padding:7px 14px; border-radius:10px; text-decoration:none;
                                font-size:13px; font-weight:600; white-space:nowrap; margin-left:8px;'>
                                🛒 Buy
                            </a>
                        </div>
                        {f"<div style='font-size:12px; color:#546e7a; margin-top:8px; padding-left:34px;'>{desc_txt[:130]}{'...' if len(desc_txt)>130 else ''}</div>" if desc_txt else ""}
                    </div>"""
                    if i % 2 == 0:
                        with c1:
                            st.markdown(card_html, unsafe_allow_html=True)
                    else:
                        with c2:
                            st.markdown(card_html, unsafe_allow_html=True)
            else:
                st.warning(f"⚠️ No drugs found for '{matched_reason}'. Try a different condition.")
        else:
            st.error(f"""
            ❌ Could not match **"{active_input}"** to any known disease/condition category.

            **Try one of these:** {', '.join(ALL_REASONS[:20])}...

            Or use the **Quick Select** buttons above.
            """)
    elif get_drugs_btn and not disease_input.strip():
        st.warning("⚠️ Please enter a disease or symptom name.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: MEDICINE → SIMILAR ALTERNATIVES
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("<h3 style='color:#1565c0;'>🔄 Find Alternative Medicines</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#e3f2fd; border-left:5px solid #1e88e5; border-radius:10px; padding:12px 18px; margin-bottom:14px;'>
        <span style='color:#1565c0; font-size:14px;'>💡 <b>Tip:</b> Select a specific medicine to find the 5 most similar alternatives using AI.</span>
    </div>
    """, unsafe_allow_html=True)

    src_col, btn_col = st.columns([4, 1])
    with src_col:
        selected_medicine_name = st.selectbox(
            "Select a medicine",
            sorted(medicines['Drug_Name'].values),
            label_visibility="collapsed"
        )
    with btn_col:
        recommend_btn = st.button("🔍 Find Alternatives", use_container_width=True)

    # Drug Description Card
    desc_row = description_data.loc[description_data['Drug_Name'] == selected_medicine_name]
    if not desc_row.empty:
        reason_val = desc_row['Reason'].values[0] if 'Reason' in desc_row.columns else ''
        desc_val = desc_row['Description'].values[0] if not desc_row.empty else ''
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#e3f2fd,#f3e5f5); border-left:5px solid #1565c0;
                    border-radius:12px; padding:18px 22px; margin:14px 0;'>
            <div style='display:flex; justify-content:space-between; align-items:flex-start;'>
                <span style='font-size:13px; color:#1565c0; font-weight:700; letter-spacing:1px; text-transform:uppercase;'>📋 About this Medicine</span>
                {f"<span style='background:#e8f5e9; color:#2e7d32; padding:4px 12px; border-radius:16px; font-size:12px; font-weight:700;'>🏥 {reason_val}</span>" if reason_val else ""}
            </div>
            <p style='font-size:15px; color:#1a237e; margin:8px 0 0 0;'>{desc_val}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    if recommend_btn:
        with st.spinner("🔬 Finding best alternatives..."):
            recommendations = recommend_similar(selected_medicine_name)

        if recommendations:
            st.markdown("<h4 style='color:#1565c0;'>📌 Top 5 Similar Alternative Medicines</h4>", unsafe_allow_html=True)

            # Get the reason of the selected drug to show context
            selected_reason = desc_row['Reason'].values[0] if not desc_row.empty and 'Reason' in desc_row.columns else ''
            if selected_reason:
                st.markdown(f"""
                <div style='background:#fff8e1; border-left:5px solid #ffa000; border-radius:10px;
                            padding:10px 16px; margin-bottom:12px;'>
                    <span style='color:#e65100; font-size:13px;'>
                        💡 Showing alternatives in the <b>{selected_reason}</b> category. These medicines have similar uses.
                    </span>
                </div>
                """, unsafe_allow_html=True)

            card_colors = [
                ("#e3f2fd", "#1565c0", "#bbdefb"),
                ("#e8f5e9", "#2e7d32", "#c8e6c9"),
                ("#fff3e0", "#e65100", "#ffe0b2"),
                ("#f3e5f5", "#6a1b9a", "#e1bee7"),
                ("#fce4ec", "#c62828", "#f48fb1"),
            ]
            rank_emojis = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]

            for i, drug in enumerate(recommendations):
                bg, accent, border = card_colors[i % len(card_colors)]
                buy_link = f"https://pharmeasy.in/search/all?name={drug}"
                # Check reason of the recommended drug
                alt_row = description_data[description_data['Drug_Name'] == drug]
                alt_reason = alt_row['Reason'].values[0] if not alt_row.empty and 'Reason' in alt_row.columns else ''
                alt_desc = alt_row['Description'].values[0] if not alt_row.empty else ''

                st.markdown(f"""
                <div style='background:{bg}; border:1.5px solid {border}; border-left:6px solid {accent};
                            border-radius:14px; padding:16px 22px; margin-bottom:12px;
                            box-shadow:0 2px 10px rgba(0,0,0,0.06);'>
                    <div style='display:flex; justify-content:space-between; align-items:center;'>
                        <div style='display:flex; align-items:center; gap:14px;'>
                            <span style='font-size:28px;'>{rank_emojis[i]}</span>
                            <div>
                                <div style='font-size:11px; color:{accent}; font-weight:700; letter-spacing:1px; text-transform:uppercase;'>
                                    Alternative #{i+1}{f" &nbsp;•&nbsp; {alt_reason}" if alt_reason else ""}
                                </div>
                                <div style='font-size:19px; font-weight:700; color:{accent};'>{drug}</div>
                                {f"<div style='font-size:12px; color:#546e7a; margin-top:4px;'>{alt_desc[:120]}{'...' if len(alt_desc)>120 else ''}</div>" if alt_desc else ""}
                            </div>
                        </div>
                        <a href='{buy_link}' target='_blank' style='background:{accent}; color:white;
                            padding:9px 20px; border-radius:10px; text-decoration:none;
                            font-size:14px; font-weight:600; white-space:nowrap;'>🛒 Buy Now</a>
                    </div>
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
    Made with ❤️ by <span style='color:#e65100; font-weight:600;'>Sai</span> &nbsp;|&nbsp;
    <a href='https://github.com/Saiyaswin' style='color:#1565c0; text-decoration:none;'>GitHub</a>
    &nbsp;© 2025
</p>
""", unsafe_allow_html=True)
