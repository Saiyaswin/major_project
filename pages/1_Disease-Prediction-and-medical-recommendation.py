import streamlit as st
import pickle
import pandas as pd
import numpy as np
from thefuzz import process
import ast

st.set_page_config(page_title="AI-Powered Healthcare Intelligence Network", page_icon="🩺", layout='wide')

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("<h2 style='color: #ffffff;'>📌 Description</h2>", unsafe_allow_html=True)
st.sidebar.image("utils/ph3.png", use_container_width=True)
st.sidebar.markdown("<p class='sidebar-text'>The Disease Prediction & Medical Recommendation system uses AI to analyze symptoms, predict diseases, assess health risks, and suggest personalized treatments—enhancing early diagnosis and improving healthcare decisions for better patient outcomes.</p>", unsafe_allow_html=True)

# ── Load Data ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_data():
    try:
        sym_des     = pd.read_csv("data/Disease-Prediction-and-Medical dataset/symptoms_df.csv")
        precautions = pd.read_csv("data/Disease-Prediction-and-Medical dataset/precautions_df.csv")
        workout     = pd.read_csv("data/Disease-Prediction-and-Medical dataset/workout_df.csv")
        description = pd.read_csv("data/Disease-Prediction-and-Medical dataset/description.csv")
        medications = pd.read_csv("data/Disease-Prediction-and-Medical dataset/medications.csv")
        diets       = pd.read_csv("data/Disease-Prediction-and-Medical dataset/diets.csv")
        model       = pickle.load(open('models/first_feature_models/RandomForest.pkl', 'rb'))
        return sym_des, precautions, workout, description, medications, diets, model
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None, None, None

sym_des, precautions, workout, description, medications, diets, model = load_data()
disease_names = list(description['Disease'].unique()) if description is not None else []

symptoms_list = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}
symptoms_list_processed = {symptom.replace('_', ' ').lower(): value for symptom, value in symptoms_list.items()}

def information(predicted_dis):
    try:
        disease_desciption = description.loc[description['Disease'] == predicted_dis, 'Description'].values[0]
        disease_precautions = precautions.loc[precautions['Disease'] == predicted_dis, ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values.flatten().tolist()
        disease_medications = ast.literal_eval(medications.loc[medications['Disease'] == predicted_dis, 'Medication'].values[0])
        disease_diet = ast.literal_eval(diets.loc[diets['Disease'] == predicted_dis, 'Diet'].values[0])
        disease_workout = workout.loc[workout['disease'] == predicted_dis, 'workout'].values.tolist()
        return disease_desciption, disease_precautions, disease_medications, disease_diet, disease_workout
    except Exception:
        return "Description not available", [], [], [], []

def predicted_value(patient_symptoms):
    try:
        i_vector = np.zeros(len(symptoms_list_processed))
        for symptom in patient_symptoms:
            i_vector[symptoms_list_processed[symptom]] = 1
        return diseases_list.get(model.predict([i_vector])[0], "Unknown Disease")
    except Exception:
        return "Prediction Error"

def correct_spelling(symptom):
    closest_match, score = process.extractOne(symptom, symptoms_list_processed.keys())
    return closest_match if score >= 80 else None

def render_result_cards(dis_des, precautions_list, medications_list, rec_diet, workout_list):
    """Render the 5 info cards for any disease result."""
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#e8f5e9,#f1f8e9); border-left:5px solid #43a047;
                border-radius:12px; padding:18px 22px; margin:14px 0;'>
        <div style='font-size:12px; color:#2e7d32; font-weight:700; letter-spacing:1px; text-transform:uppercase;'>📋 About this Disease</div>
        <p style='font-size:15px; color:#1b5e20; margin:8px 0 0 0;'>{dis_des}</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        # Precautions
        items = [f"<li style='margin-bottom:5px;'>⚠️ {p}</li>" for p in precautions_list if p]
        st.markdown(f"""
        <div style='background:#fff8e1; border-left:5px solid #ffa000; border-radius:12px; padding:16px 20px; margin-bottom:12px; height:100%;'>
            <b style='color:#e65100;'>🛡️ Precautions</b>
            <ul style='margin:10px 0 0 0; padding-left:18px; color:#5d4037;'>{''.join(items) if items else '<li>Not available</li>'}</ul>
        </div>""", unsafe_allow_html=True)
        # Workout
        items = [f"<li style='margin-bottom:5px;'>🏃 {w}</li>" for w in workout_list if w]
        st.markdown(f"""
        <div style='background:#e8eaf6; border-left:5px solid #3949ab; border-radius:12px; padding:16px 20px; height:100%;'>
            <b style='color:#1a237e;'>🏋️ Recommended Workout</b>
            <ul style='margin:10px 0 0 0; padding-left:18px; color:#283593;'>{''.join(items) if items else '<li>Not available</li>'}</ul>
        </div>""", unsafe_allow_html=True)

    with c2:
        # Medications
        items = [f"<li style='margin-bottom:5px;'>💊 {m}</li>" for m in medications_list if m]
        st.markdown(f"""
        <div style='background:#fce4ec; border-left:5px solid #e91e63; border-radius:12px; padding:16px 20px; margin-bottom:12px; height:100%;'>
            <b style='color:#880e4f;'>💊 Medications</b>
            <ul style='margin:10px 0 0 0; padding-left:18px; color:#880e4f;'>{''.join(items) if items else '<li>Not available</li>'}</ul>
        </div>""", unsafe_allow_html=True)
        # Diet
        items = [f"<li style='margin-bottom:5px;'>🥗 {d}</li>" for d in rec_diet if d]
        st.markdown(f"""
        <div style='background:#e0f2f1; border-left:5px solid #00897b; border-radius:12px; padding:16px 20px; height:100%;'>
            <b style='color:#004d40;'>🥦 Recommended Diet</b>
            <ul style='margin:10px 0 0 0; padding-left:18px; color:#00695c;'>{''.join(items) if items else '<li>Not available</li>'}</ul>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HEADER — Full-width front page style
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='text-align:center; padding:10px 0 14px 0;'>
    <div style='font-size:44px; font-weight:800; color:#00695c;'>🩺 Disease Prediction &amp; Medical Recommendation</div>
    <div style='font-size:18px; color:#555; margin-top:8px;'>AI-powered symptom analysis to predict diseases and provide personalised medical guidance.</div>
    <div style='display:flex; gap:10px; flex-wrap:wrap; justify-content:center; margin-top:14px;'>
        <span style='background:#e8f5e9;color:#2e7d32;padding:5px 14px;border-radius:20px;font-size:13px;font-weight:600;border:1px solid #c8e6c9;'>🌲 Random Forest Model</span>
        <span style='background:#e3f2fd;color:#1565c0;padding:5px 14px;border-radius:20px;font-size:13px;font-weight:600;border:1px solid #bbdefb;'>💊 Medication Guide</span>
        <span style='background:#fff3e0;color:#e65100;padding:5px 14px;border-radius:20px;font-size:13px;font-weight:600;border:1px solid #ffe0b2;'>🥗 Diet &amp; Workout Plans</span>
        <span style='background:#f3e5f5;color:#6a1b9a;padding:5px 14px;border-radius:20px;font-size:13px;font-weight:600;border:1px solid #e1bee7;'>🔬 41 Diseases</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.image("utils/ph1.png", use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DISEASE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<h3 style='color:#00695c;'>🔬 Disease Prediction Based on Symptoms</h3>", unsafe_allow_html=True)
st.markdown("""
<div style='background:#f1f8e9; border-left:5px solid #7cb342; border-radius:10px; padding:12px 18px; margin-bottom:14px;'>
    <span style='color:#33691e; font-size:14px;'>💡 <b>Tip:</b> Enter as many symptoms as possible (comma-separated) for the most accurate prediction.</span>
</div>
""", unsafe_allow_html=True)

# ── Predict button style ───────────────────────────────────────────────────────
st.markdown("""
<style>
div[data-testid="stButton"] > button[kind="secondary"] {
    background: linear-gradient(135deg, #2e7d32, #43a047, #66bb6a) !important;
    color: white !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 10px !important;
    width: 100% !important;
    height: 108px !important;
    box-shadow: 0 4px 14px rgba(46,125,50,0.35) !important;
    transition: all 0.25s ease !important;
    letter-spacing: 0.5px !important;
}
div[data-testid="stButton"] > button[kind="secondary"]:hover {
    background: linear-gradient(135deg, #1b5e20, #2e7d32, #43a047) !important;
    box-shadow: 0 6px 22px rgba(46,125,50,0.55) !important;
    transform: translateY(-2px) !important;
}
</style>
""", unsafe_allow_html=True)

inp_col, btn_col = st.columns([4, 1])
with inp_col:
    user_input = st.text_area(
        "🤒 Enter your symptoms:",
        placeholder="e.g., headache, constipation, nausea, vomiting, fatigue",
        height=108,
        label_visibility="collapsed"
    )
with btn_col:
    predict_btn = st.button("🔬 Predict Disease", use_container_width=True)


def keyword_to_symptoms(keyword):
    """Return all symptom keys whose name CONTAINS the keyword as a substring."""
    kw = keyword.strip().lower().replace(" ", "_")
    kw_plain = keyword.strip().lower()
    matched = []
    for sym_key in symptoms_list_processed.keys():   # keys are already human-readable (spaces)
        sym_key_nospace = sym_key.replace(" ", "_")
        if kw in sym_key or kw_plain in sym_key or kw in sym_key_nospace:
            matched.append(sym_key)
    return matched

def diseases_with_symptom(symptom_name):
    """Return every disease in symptoms_df that has this symptom."""
    col_syms = ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']
    cols = [c for c in col_syms if c in sym_des.columns]
    mask = False
    for c in cols:
        mask = mask | sym_des[c].str.strip().str.lower().eq(symptom_name.replace(' ', '_'))
    return sym_des.loc[mask, 'Disease'].unique().tolist() if hasattr(mask, '__iter__') else []

def diseases_matching_name(keyword):
    """Return diseases whose name CONTAINS the keyword as a substring."""
    kw = keyword.strip().lower()
    return [d for d in disease_names if kw in d.lower()]

if predict_btn:
    if user_input:
        raw_keywords = [s.strip() for s in user_input.split(',')]
        final_symptoms = []   # confirmed symptom keys
        info_lines = []       # messages to show user

        for kw in raw_keywords:
            # ── Tier 1: Disease name substring match (checked FIRST) ────────────
            dis_matches = diseases_matching_name(kw)
            if dis_matches:
                sel = dis_matches[0]
                info_lines.append(f"💡 **'{kw}'** matched disease name → **{sel}**")
                with st.spinner("🧠 Fetching disease info..."):
                    dis_des_v, prec_v, meds_v, diet_v, wrkt_v = information(sel)
                st.markdown(f"""
                <div style='background:linear-gradient(135deg,#e3f2fd,#ede7f6); border:2px solid #1e88e5;
                            border-radius:16px; padding:22px 28px; text-align:center; margin:16px 0;
                            box-shadow:0 4px 16px rgba(0,0,0,0.08);'>
                    <div style='font-size:40px; margin-bottom:6px;'>🔍</div>
                    <div style='font-size:13px; color:#1565c0; font-weight:700;
                                letter-spacing:1.5px; text-transform:uppercase;'>Matched by Disease Name</div>
                    <div style='font-size:34px; font-weight:800; color:#0d47a1; margin:6px 0;'>{sel}</div>
                    <span style='background:#1e88e5; color:white; padding:5px 18px;
                                 border-radius:16px; font-size:14px; font-weight:600;'>Name Match ✅</span>
                </div>
                """, unsafe_allow_html=True)
                for line in info_lines:
                    st.info(line)
                render_result_cards(dis_des_v, prec_v, meds_v, diet_v, wrkt_v)
                st.stop()

            # ── Tier 2: Symptom substring match ─────────────────────────────────
            sub_matches = keyword_to_symptoms(kw)
            if sub_matches:
                final_symptoms.extend(sub_matches)
                info_lines.append(f"🔎 **'{kw}'** matched symptoms: " + ", ".join(
                    [f"`{s}`" for s in sub_matches]))
                continue

            # ── Tier 3: Fuzzy spelling correction ───────────────────────────────
            fuzzy_match = correct_spelling(kw)
            if fuzzy_match:
                final_symptoms.append(fuzzy_match)
                info_lines.append(f"🔤 **'{kw}'** fuzzy-corrected → `{fuzzy_match}`")
            else:
                info_lines.append(f"⚠️ **'{kw}'** — no match found, skipped.")

        # Show what was resolved
        if info_lines:
            for line in info_lines:
                st.info(line)

        if final_symptoms:
            # ── Single symptom: bypass RF model, look up diseases directly ───────
            if len(final_symptoms) == 1:
                sym = final_symptoms[0]
                possible_diseases = diseases_with_symptom(sym)
                if possible_diseases:
                    # Use first matching disease directly from dataset
                    direct_disease = possible_diseases[0]
                    dis_des_r, prec_r, meds_r, rec_diet_r, wrkt_r = information(direct_disease)

                    # Show possible diseases list
                    all_dis_str = ", ".join(possible_diseases)
                    st.markdown(f"""
                    <div style='background:#fff8e1; border-left:5px solid #ffa000;
                                border-radius:12px; padding:16px 20px; margin:12px 0;'>
                        <b style='color:#e65100;'>⚠️ Only 1 symptom matched: <code>{sym}</code></b><br>
                        <span style='color:#5d4037; font-size:14px;'>
                            This symptom appears in: <b>{all_dis_str}</b>.<br>
                            Showing info for the most likely: <b>{direct_disease}</b>.
                            Enter more symptoms for a precise AI prediction.
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div style='background:linear-gradient(135deg,#e8f5e9,#e0f2f1); border:2px solid #43a047;
                                border-radius:16px; padding:22px 28px; text-align:center; margin:16px 0;
                                box-shadow:0 4px 16px rgba(0,0,0,0.08);'>
                        <div style='font-size:40px; margin-bottom:6px;'>🩺</div>
                        <div style='font-size:13px; color:#2e7d32; font-weight:700;
                                    letter-spacing:1.5px; text-transform:uppercase;'>Most Likely Disease</div>
                        <div style='font-size:36px; font-weight:800; color:#1b5e20; margin:6px 0;'>{direct_disease}</div>
                        <span style='background:#43a047; color:white; padding:5px 18px;
                                     border-radius:16px; font-size:14px; font-weight:600;'>Dataset Match 🗂️</span>
                    </div>
                    """, unsafe_allow_html=True)
                    render_result_cards(dis_des_r, prec_r, meds_r, rec_diet_r, wrkt_r)
                else:
                    st.warning("⚠️ Symptom found but no matching disease in dataset.")

            else:
                # ── Multiple symptoms: use RF model ──────────────────────────────
                with st.spinner("🧠 Analyzing your symptoms..."):
                    predicted_disease = predicted_value(final_symptoms)
                    dis_des_r, prec_r, meds_r, rec_diet_r, wrkt_r = information(predicted_disease)

                st.markdown(f"""
                <div style='background:linear-gradient(135deg,#e8f5e9,#e0f2f1); border:2px solid #43a047;
                            border-radius:16px; padding:22px 28px; text-align:center; margin:16px 0;
                            box-shadow:0 4px 16px rgba(0,0,0,0.08);'>
                    <div style='font-size:40px; margin-bottom:6px;'>🩺</div>
                    <div style='font-size:13px; color:#2e7d32; font-weight:700;
                                letter-spacing:1.5px; text-transform:uppercase;'>Predicted Disease</div>
                    <div style='font-size:36px; font-weight:800; color:#1b5e20; margin:6px 0;'>{predicted_disease}</div>
                    <span style='background:#43a047; color:white; padding:5px 18px;
                                 border-radius:16px; font-size:14px; font-weight:600;'>AI Prediction ✅</span>
                </div>
                """, unsafe_allow_html=True)
                render_result_cards(dis_des_r, prec_r, meds_r, rec_diet_r, wrkt_r)
        else:
            st.error("❌ None of the keywords matched any known symptom or disease. Please check your spelling.")
    else:
        st.warning("⚠️ Please enter at least one symptom or keyword.")


st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DISEASE SEARCH
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<h3 style='color:#1565c0;'>🔍 Search Disease Description &amp; Recommendations</h3>", unsafe_allow_html=True)
st.markdown("""
<div style='background:#e3f2fd; border-left:5px solid #1e88e5; border-radius:10px; padding:12px 18px; margin-bottom:14px;'>
    <span style='color:#1565c0; font-size:14px;'>💡 <b>Tip:</b> Start typing a disease name to get its full description, medications, diet and workout plan.</span>
</div>
""", unsafe_allow_html=True)

disease_query = st.text_input(
    "🔍 Search disease:",
    placeholder="Start typing a disease name e.g. Diabetes, Malaria, Dengue...",
    label_visibility="collapsed"
)

if disease_query:
    matches = [d for d in disease_names if d.lower().startswith(disease_query.lower())]
    if matches:
        selected_disease = matches[0]
        dis_des, prec, meds, rec_diet, wrkt = information(selected_disease)

        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#e3f2fd,#ede7f6); border:2px solid #1e88e5;
                    border-radius:16px; padding:22px 28px; text-align:center; margin:16px 0;
                    box-shadow:0 4px 16px rgba(0,0,0,0.08);'>
            <div style='font-size:40px; margin-bottom:6px;'>🔍</div>
            <div style='font-size:13px; color:#1565c0; font-weight:700; letter-spacing:1.5px; text-transform:uppercase;'>Disease Information</div>
            <div style='font-size:34px; font-weight:800; color:#0d47a1; margin:6px 0;'>{selected_disease}</div>
        </div>
        """, unsafe_allow_html=True)

        render_result_cards(dis_des, prec, meds, rec_diet, wrkt)
    else:
        st.warning("⚠️ No matching disease found. Try a different name.")

# ── Quick Tips strip ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🌿 General Health Tips")
tc1, tc2, tc3, tc4 = st.columns(4)
with tc1: st.info("🩺 **See a Doctor**\nNever self-medicate. Always consult a healthcare professional.")
with tc2: st.info("💧 **Stay Hydrated**\nDrink 8+ glasses of water every day.")
with tc3: st.info("😴 **Rest Well**\nAdequate sleep boosts your immune system.")
with tc4: st.info("🥗 **Eat Right**\nA balanced diet is your first line of defense.")