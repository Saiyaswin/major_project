# 🏥 AI-Diseases Prediction App

An AI-powered multi-feature healthcare web application built with **Streamlit**, combining machine learning models, NLP, and rich UI design to assist with disease prediction, drug recommendation, heart disease risk assessment, and AI medical chat.

---

## 🚀 How to Run

```bash
# Make sure you are in the project root
cd "AI-Diseases Prediction"

# Activate virtual environment (Windows)
venv\Scripts\activate

# Run the app
streamlit run home.py
```

App will open at: **http://localhost:8501**

---

## 📁 Project Structure

```
AI-Diseases Prediction/
├── home.py                          # Main landing page
├── pages/
│   ├── 1_Disease-Prediction-and-medical-recommendation.py
│   ├── 2_drug_recommendation.py
│   ├── 3_heart_Disease_Risk_Assesment.py
│   ├── 4_Medibot.py
│   └── 5_Hospital_Recommendation.py
├── models/
│   ├── first_feature_models/
│   │   └── RandomForest.pkl          # Disease prediction model
│   ├── second_feature_models/
│   │   ├── medicine_dict.pkl         # Drug name dictionary
│   │   └── similarity.joblib         # Cosine similarity matrix
│   └── third_feature_models/
│       ├── best_model.pkl            # Heart disease EasyEnsemble+LightGBM
│       └── cbe_encoder.pkl           # CatBoost encoder for heart inputs
├── data/
│   ├── Disease-Prediction-and-Medical dataset/
│   │   ├── Training.csv              # Model training data (4,920 rows, 41 diseases)
│   │   ├── symptoms_df.csv           # Symptom combinations per disease
│   │   ├── description.csv           # Disease descriptions
│   │   ├── precautions_df.csv        # Precautions per disease
│   │   ├── medications.csv           # Medications per disease
│   │   ├── diets.csv                 # Diet recommendations
│   │   └── workout_df.csv            # Workout recommendations
│   └── Drug reccomendation/
│       └── medicine.csv              # 10,000+ medicine names + descriptions
├── utils/
│   ├── style_v1.css                  # Global stylesheet
│   ├── heart_disease.jpg             # Heart disease page banner image
│   ├── medss.png                     # Drug recommendation banner image
│   └── ph3.png / ph4.png             # Sidebar images
└── requirements.txt
```

---

## 🔬 Feature 1 — Disease Prediction & Medical Recommendation

**Page:** `1_Disease-Prediction-and-medical-recommendation.py`
**Model:** `RandomForest.pkl` trained on 4,920 symptom-disease records across **41 diseases**

### How It Works
1. User enters symptoms (comma-separated keywords)
2. **3-tier smart search** resolves each keyword:
   - **Tier 1 — Substring match:** Scans all 132 symptom names for the keyword as a substring
   - **Tier 2 — Disease name match:** If no symptom found, checks if keyword matches a disease name directly (e.g. typing `cold` → fetches **Common Cold** info)
   - **Tier 3 — Fuzzy fallback:** Corrects spelling mistakes using `thefuzz`
3. A 132-length binary vector is built and passed to the RandomForest model
4. Predicted disease is shown with full information

### Output Cards
After prediction, displays 5 color-coded cards:
| Card | Color | Content |
|------|-------|---------|
| About Disease | Green | Disease description |
| Precautions | Yellow | 4 precautions with icons |
| Medications | Pink | Medication list |
| Diet | Teal | Recommended foods |
| Workout | Indigo | Exercise recommendations |

### Supported Diseases (41 total)
Fungal infection, Allergy, GERD, Chronic cholestasis, Drug Reaction, Peptic ulcer, AIDS, Diabetes, Gastroenteritis, Bronchial Asthma, Hypertension, Migraine, Cervical spondylosis, Paralysis, Jaundice, Malaria, Chicken pox, Dengue, Typhoid, Hepatitis A/B/C/D/E, Alcoholic hepatitis, Tuberculosis, Common Cold, Pneumonia, Piles, Heart attack, Varicose veins, Hypothyroidism, Hyperthyroidism, Hypoglycemia, Osteoarthritis, Arthritis, Vertigo, Acne, UTI, Psoriasis, Impetigo

---

## 💊 Feature 2 — Drug Recommendation

**Page:** `2_drug_recommendation.py`
**Model:** Pre-computed cosine similarity matrix (`similarity.joblib`) on 10,000+ medicines

### How It Works
1. User selects a medicine from the dropdown
2. System finds the medicine index in `medicine_dict.pkl`
3. Top 5 most similar medicines are retrieved from the similarity matrix
4. Results shown with rank medals, color-coded cards, and **Buy Now** links to PharmEasy

### UI Highlights
- Ranked result cards (🥇🥈🥉4️⃣5️⃣) in 5 distinct colors
- Drug description fetched from `medicine.csv`
- 4 Medicine Safety Tips at the bottom

---

## ❤️ Feature 3 — Heart Disease Risk Assessment

**Page:** `3_heart_Disease_Risk_Assesment.py`
**Model:** `EasyEnsembleClassifier` with `LightGBM` base estimator + `CatBoostEncoder`
**Dataset:** BRFSS 2022 Survey (400,000+ patient records)

### Input Sections
Inputs are organized into 3 sections with 22 total features:

| Section | Emoji | Fields |
|---------|-------|--------|
| Demographics | 👤 | Gender, Race/Ethnicity, Age Group |
| Medical History | 🏥 | General Health, Heart Attack, Kidney Disease, Asthma, BMI, Diabetes, Stroke, Depression, Physical/Mental Health days, Walking difficulty, Healthcare provider, Checkup history, Doctor affordability |
| Lifestyle | 🌿 | Smoking status, Sleep hours, Alcohol drinks/week, Binge drinking, Exercise |

### Output — Dynamic Risk Score Card
Color-coded card based on predicted risk:
| Risk % | Color | Label |
|--------|-------|-------|
| ≤ 25% | 🟢 Green | Low Risk ✅ |
| 26–40% | 🟡 Yellow | Moderate Risk |
| 41–70% | 🟠 Orange | High Risk ⚠️ |
| > 70% | 🔴 Red | Very High Risk ⚠️ |

Also includes:
- Large bold risk % display
- Visual progress bar
- SHAP explainability via donut chart showing top risk factors

---

## 🤖 Feature 4 — Medibot (AI Medical Assistant)

**Page:** `4_Medibot.py`
**Model:** `all-MiniLM-L6-v2` (SentenceTransformer) + `googletrans`

### How It Works
1. User asks a medical question in any language
2. Question is encoded and compared via cosine similarity to known disease-cure pairs in `dataset - Sheet1.csv`
3. Best matching cure is returned and optionally translated

---

## 🏥 Feature 5 — Hospital Recommendation

**Page:** `5_Hospital_Recommendation.py`

- 3-column input: Location + Hospital Type + Specialty
- Styled Google Maps button to open search
- Embedded live Google Maps iframe
- Hospital selection tips section

---

## 🎨 UI Enhancements Made

All pages were redesigned with:
- **Header banners** with relevant images + feature badge pills
- **Colored section headers** with emoji icons
- **Styled action buttons** with gradient backgrounds and hover effects
- **Color-coded result cards** for all outputs
- **Tip banners** with helpful instructions
- **Footer** with attribution

---

## 📦 Dependencies

```
streamlit
pandas
numpy
scikit-learn
lightgbm
imbalanced-learn
category_encoders
shap
plotly
sentence-transformers
googletrans==4.0.0-rc1
thefuzz
pillow
joblib
```

Install all:
```bash
pip install -r requirements.txt
```

---

## ⚠️ Known Issues

1. **`medications.csv` data mismatch** — Some diseases have incorrect medication lists due to misaligned rows in the CSV (e.g. Heart Attack row shows Varicose Veins medications). Requires manual CSV correction.
2. **Single-symptom predictions** — Entering only one symptom gives lower accuracy. Always provide 3–5 symptoms for the disease prediction feature.
3. **Hospital map iframe** — Google Maps may block iframe embedding in some browsers due to X-Frame-Options restrictions.

---

## 👤 Author

Made with ❤️ by **Sai**
- [GitHub](https://github.com/Saiyaswin)
- © 2026
