import streamlit as st

st.set_page_config(
    page_title="AI-Powered Healthcare Intelligence Network",
    page_icon="🩺",
    layout="wide"
)

# ── Custom Sidebar Nav CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebarNav"] {
    background: linear-gradient(160deg, #e8eaf6 0%, #ede7f6 100%);
    border-radius: 12px;
    padding: 8px 4px 10px 4px;
    margin-bottom: 8px;
    border: 1.5px solid #c5cae9;
}
[data-testid="stSidebarNav"] a {
    display: flex;
    align-items: center;
    padding: 9px 14px;
    border-radius: 10px;
    margin: 3px 4px;
    font-size: 13.5px;
    font-weight: 600;
    color: #283593 !important;
    text-decoration: none !important;
    transition: background 0.2s, transform 0.15s;
}
[data-testid="stSidebarNav"] a:hover {
    background: rgba(63,81,181,0.13);
    transform: translateX(4px);
    color: #1a237e !important;
}
[data-testid="stSidebarNav"] a[aria-current="page"] {
    background: #dde9ff !important;
    color: #1a237e !important;
    font-weight: 800;
    border-left: 4px solid #3949ab;
    box-shadow: 0 2px 8px rgba(57,73,171,0.15);
}
[data-testid="stSidebarNav"] a[aria-current="page"]:hover {
    background: #c5d8ff !important;
    transform: none;
}
[data-testid="stSidebarNav"] span {
    font-family: 'Segoe UI', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("""
<div style='text-align:center; padding:12px 0 6px 0;'>
    <div style='font-size:28px; font-weight:900; color:#00695c;'>🩺 HealthAI</div>
    <div style='font-size:12px; color:#888; margin-top:2px;'>AI-Powered Healthcare Network</div>
</div>
<hr style='border:none; border-top:1.5px solid #e0f2f1; margin:8px 0;'>
""", unsafe_allow_html=True)

st.sidebar.image("utils/ph3.png", use_container_width=True)

st.sidebar.markdown("""
<hr style='border:none; border-top:1.5px solid #e0f2f1; margin:10px 0;'>
<div style='font-size:13px; font-weight:700; color:#00695c; margin-bottom:8px; letter-spacing:0.5px;'>📌 Quick Navigation</div>
<div style='font-size:13px; color:#444; line-height:2.2;'>
    🩺 &nbsp;Disease Prediction<br>
    💊 &nbsp;Drug Recommendation<br>
    ❤️ &nbsp;Heart Risk Assessment<br>
    🤖 &nbsp;AI Medibot<br>
    🏥 &nbsp;Hospital Finder
</div>
<hr style='border:none; border-top:1.5px solid #e0f2f1; margin:12px 0;'>
<div style='font-size:12px; color:#aaa; text-align:center; line-height:1.8;'>
    Built with ❤️ using <b style='color:#00897b;'>Streamlit</b><br>
    © 2025 AI Healthcare Network
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HERO SECTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='text-align:center; padding:28px 0 12px 0;'>
    <div style='font-size:52px; font-weight:900; color:#00695c; letter-spacing:-1px; line-height:1.1;'>
        🩺 AI-Powered Healthcare
    </div>
    <div style='font-size:42px; font-weight:900; color:#00897b; letter-spacing:-1px;'>
        Intelligence Network
    </div>
    <div style='font-size:19px; color:#666; margin-top:12px; font-weight:400;'>
        Transforming Healthcare with
        <span style='color:#00897b; font-weight:700;'>AI-driven Predictions</span> &amp; Insights
    </div>
    <div style='display:flex; gap:10px; flex-wrap:wrap; justify-content:center; margin-top:18px;'>
        <span style='background:#e8f5e9;color:#2e7d32;padding:6px 16px;border-radius:20px;font-size:13px;font-weight:700;border:1.5px solid #c8e6c9;'>🔬 41 Diseases</span>
        <span style='background:#e3f2fd;color:#1565c0;padding:6px 16px;border-radius:20px;font-size:13px;font-weight:700;border:1.5px solid #bbdefb;'>💊 10,000+ Medicines</span>
        <span style='background:#fce4ec;color:#c62828;padding:6px 16px;border-radius:20px;font-size:13px;font-weight:700;border:1.5px solid #f8bbd0;'>❤️ Heart Risk AI</span>
        <span style='background:#f3e5f5;color:#6a1b9a;padding:6px 16px;border-radius:20px;font-size:13px;font-weight:700;border:1.5px solid #e1bee7;'>🤖 AI Chatbot</span>
        <span style='background:#e0f2f1;color:#00695c;padding:6px 16px;border-radius:20px;font-size:13px;font-weight:700;border:1.5px solid #b2dfdb;'>🏥 Hospital Finder</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Hero Image ──────────────────────────────────────────────────────────────
st.image("utils/front page.jpg", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# FEATURES
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='text-align:center; margin-bottom:20px;'>
    <div style='font-size:32px; font-weight:900; color:#1b5e20;'>✨ Features</div>
    <div style='font-size:14px; color:#888; margin-top:4px;'>Everything you need for AI-powered healthcare decisions</div>
</div>
""", unsafe_allow_html=True)

features = [
    ("🩺", "Disease Prediction", "#e8f5e9", "#2e7d32", "#c8e6c9",
     "Predict diseases from symptoms using a trained Random Forest model across 41 conditions."),
    ("💊", "Drug Recommendation", "#e3f2fd", "#1565c0", "#bbdefb",
     "AI-powered medicine suggestions using NLP cosine similarity on 10,000+ drugs."),
    ("❤️", "Heart Risk Assessment", "#fce4ec", "#c62828", "#f8bbd0",
     "Predict heart disease risk % using BRFSS 2022 data and a LightGBM ensemble."),
    ("🤖", "AI Medibot", "#f3e5f5", "#6a1b9a", "#e1bee7",
     "Chat with an AI medical assistant for health queries in any language."),
    ("🏥", "Hospital Finder", "#fff3e0", "#e65100", "#ffe0b2",
     "Find nearby hospitals by type & specialty with live Google Maps integration."),
]

feat_cols = st.columns(5)
for col, (icon, title, bg, color, border, desc) in zip(feat_cols, features):
    with col:
        st.markdown(f"""
        <div style='background:{bg}; border:1.5px solid {border}; border-radius:16px;
                    padding:22px 14px; text-align:center; min-height:220px;
                    box-shadow:0 4px 14px rgba(0,0,0,0.07);'>
            <div style='font-size:38px; margin-bottom:8px;'>{icon}</div>
            <div style='font-size:14px; font-weight:800; color:{color}; margin-bottom:8px; line-height:1.3;'>{title}</div>
            <div style='font-size:12px; color:#555; line-height:1.6;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# TECHNOLOGIES
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='text-align:center; margin:16px 0 20px 0;'>
    <div style='font-size:30px; font-weight:900; color:#1b5e20;'>⚙️ Technologies Used</div>
    <div style='font-size:14px; color:#888; margin-top:4px;'>Powered by cutting-edge AI &amp; data science tools</div>
</div>
""", unsafe_allow_html=True)

techs = [
    ("🤖", "Machine Learning", "#e8f5e9", "#2e7d32", "#c8e6c9",
     "RandomForest · LightGBM · EasyEnsemble · scikit-learn"),
    ("💬", "NLP & AI", "#e3f2fd", "#1565c0", "#bbdefb",
     "SentenceTransformer · Cosine Similarity · googletrans"),
    ("📊", "Data Science", "#fff3e0", "#e65100", "#ffe0b2",
     "Pandas · NumPy · SHAP · Plotly · joblib"),
    ("🌐", "Web & Deployment", "#f3e5f5", "#6a1b9a", "#e1bee7",
     "Streamlit · Python · Google Maps API"),
]

tech_cols = st.columns(4)
for col, (icon, title, bg, color, border, desc) in zip(tech_cols, techs):
    with col:
        st.markdown(f"""
        <div style='background:{bg}; border:1.5px solid {border}; border-radius:14px;
                    padding:20px 14px; text-align:center;
                    box-shadow:0 3px 10px rgba(0,0,0,0.06);'>
            <div style='font-size:32px; margin-bottom:6px;'>{icon}</div>
            <div style='font-size:14px; font-weight:800; color:{color}; margin-bottom:6px;'>{title}</div>
            <div style='font-size:12px; color:#666; line-height:1.6;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# WHY USE THIS APP
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='text-align:center; margin:16px 0 20px 0;'>
    <div style='font-size:30px; font-weight:900; color:#1b5e20;'>🔍 Why Use This App?</div>
</div>
""", unsafe_allow_html=True)

whys = [
    ("✅", "Accurate Predictions", "#e8f5e9", "#2e7d32", "#c8e6c9",
     "AI models trained on large real-world healthcare datasets for reliable results."),
    ("⚡", "Real-Time Assistance", "#e3f2fd", "#1565c0", "#bbdefb",
     "Get instant health insights and personalised recommendations in seconds."),
    ("🎯", "User-Friendly", "#fff3e0", "#e65100", "#ffe0b2",
     "Designed for both healthcare professionals and everyday users."),
    ("🔒", "Secure & Reliable", "#f3e5f5", "#6a1b9a", "#e1bee7",
     "Your health data is handled responsibly and is never stored."),
]

why_cols = st.columns(4)
for col, (icon, title, bg, color, border, desc) in zip(why_cols, whys):
    with col:
        st.markdown(f"""
        <div style='background:{bg}; border:1.5px solid {border}; border-radius:14px;
                    padding:20px 14px; text-align:center;
                    box-shadow:0 3px 10px rgba(0,0,0,0.06);'>
            <div style='font-size:30px; margin-bottom:6px;'>{icon}</div>
            <div style='font-size:14px; font-weight:800; color:{color}; margin-bottom:6px;'>{title}</div>
            <div style='font-size:12px; color:#666; line-height:1.6;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
