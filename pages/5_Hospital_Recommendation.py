import streamlit as st
import urllib.parse

st.set_page_config(page_title="Hospital Recommendation", page_icon="🏥", layout="wide")

st.title("🏥 Hospital Recommendation System")
st.write("Find the best nearby hospitals and clinics based on your location and needs.")

st.markdown("---")

# ── Input Section ──────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    location = st.text_input(
        "📍 Enter your city or area",
        placeholder="Eg: Chennai, Coimbatore, Madurai"
    )

with col2:
    hospital_type = st.selectbox(
        "🏨 Hospital Type",
        [
            "Hospital",
            "Government Hospital",
            "Private Hospital",
            "Multispeciality Hospital",
            "Clinic",
            "Medical Centre",
            "Nursing Home",
        ]
    )

with col3:
    speciality = st.selectbox(
        "🩺 Speciality (optional)",
        [
            "General",
            "Cardiology",
            "Neurology",
            "Orthopedics",
            "Oncology",
            "Pediatrics",
            "Gynaecology",
            "Dermatology",
            "Ophthalmology",
            "ENT",
            "Psychiatry",
            "Dental",
            "Emergency & Trauma",
        ]
    )

st.markdown("")

# ── Search Button ──────────────────────────────────────────────────────────────
search_clicked = st.button("🔍 Find Hospitals", use_container_width=True)

if search_clicked:
    if location.strip() == "":
        st.warning("⚠️ Please enter a location to search.")
    else:
        # Build query
        spec_part = f" {speciality}" if speciality != "General" else ""
        query = f"{hospital_type}{spec_part} near {location}"
        encoded_query = urllib.parse.quote(query)
        map_url = f"https://www.google.com/maps/search/{encoded_query}"
        embed_url = f"https://maps.google.com/maps?q={encoded_query}&output=embed"

        st.success(f"🔎 Showing results for: **{query}**")

        st.markdown("---")

        # ── Google Maps Button (prominent) ─────────────────────────────────────
        st.markdown(
            f"""
            <div style="text-align:center; margin: 16px 0 24px 0;">
                <a href="{map_url}" target="_blank"
                   style="
                       display: inline-block;
                       background-color: #4285F4;
                       color: white;
                       font-size: 18px;
                       font-weight: 600;
                       padding: 14px 36px;
                       border-radius: 10px;
                       text-decoration: none;
                       box-shadow: 0 4px 12px rgba(66,133,244,0.4);
                       transition: background 0.2s;
                   "
                   onmouseover="this.style.backgroundColor='#1a73e8'"
                   onmouseout="this.style.backgroundColor='#4285F4'"
                >
                    📍 Open in Google Maps
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ── Embedded Map Preview ───────────────────────────────────────────────
        st.markdown("##### 🗺️ Map Preview")

        st.components.v1.iframe(
            embed_url,
            width=None,
            height=520,
            scrolling=True
        )

        st.markdown("---")

        # ── Quick Tips ─────────────────────────────────────────────────────────
        st.markdown("### � Tips for Choosing a Hospital")
        tip_col1, tip_col2, tip_col3 = st.columns(3)
        with tip_col1:
            st.info("⭐ **Check Reviews**\nRead patient reviews and ratings before visiting.")
        with tip_col2:
            st.info("📞 **Call Ahead**\nConfirm availability and appointment timings.")
        with tip_col3:
            st.info("🚑 **Check Emergency**\nVerify 24/7 emergency services for critical cases.")
