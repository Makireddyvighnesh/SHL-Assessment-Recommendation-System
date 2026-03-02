import os
import sys
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import requests
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")

TEST_TYPE_COLORS = {
    "Ability & Aptitude": "#4A90D9",
    "Biodata & Situational Judgement": "#7B68EE",
    "Competencies": "#20B2AA",
    "Development & 360": "#DAA520",
    "Assessment Exercises": "#FF7F50",
    "Knowledge & Skills": "#3CB371",
    "Personality & Behavior": "#CD5C5C",
    "Simulations": "#9370DB",
}

# ── Page Setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🎯",
    layout="wide",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #555;
        margin-bottom: 2rem;
        font-size: 1rem;
    }
    .assessment-card {
        background: #f8f9fa;
        border-left: 4px solid #4A90D9;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 4px;
        color: white;
    }
    .meta-pill {
        background: #e9ecef;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.78rem;
        margin-right: 6px;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🎯 SHL Assessment Recommender</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Enter a job description or hiring query to get intelligent assessment recommendations</div>', unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
query = st.text_area(
    "Job Description / Query",
    height=160,
    placeholder="Paste a job description, hiring requirement, or role description here...",
    label_visibility="collapsed",
)

submit = st.button("🔍 Get Recommendations", type="primary")

# ── Submit Logic ──────────────────────────────────────────────────────────────
if submit:
    if not query or not query.strip():
        st.warning("Please enter a job description or query.")
    else:
        with st.spinner("Analyzing query and retrieving assessments..."):
            try:
                response = requests.post(
                    f"{API_BASE}/recommend",
                    json={"query": query.strip()},
                    timeout=60,
                )

                if response.status_code == 200:
                    data = response.json()
                    assessments = data.get("recommended_assessments", [])

                    st.success(f"Found **{len(assessments)}** relevant assessments")
                    st.markdown("---")

                    # Download CSV
                    if assessments:
                        df = pd.DataFrame([
                            {
                                "Name": a["name"],
                                "URL": a["url"],
                                "Test Types": ", ".join(a.get("test_type", [])),
                                "Duration (min)": a.get("duration") or "N/A",
                                "Remote": a.get("remote_support"),
                                "Adaptive": a.get("adaptive_support"),
                            }
                            for a in assessments
                        ])

                        st.download_button(
                            "📥 Download Results as CSV",
                            df.to_csv(index=False),
                            file_name="shl_recommendations.csv",
                            mime="text/csv",
                        )

                    # Assessment Cards
                    for i, a in enumerate(assessments, 1):
                        badges_html = ""
                        for tt in a.get("test_type", []):
                            color = TEST_TYPE_COLORS.get(tt, "#888")
                            badges_html += f'<span class="badge" style="background:{color}">{tt}</span>'

                        duration_display = f"⏱ {a.get('duration')} min" if a.get("duration") else "⏱ N/A"
                        remote_display = "🌐 Remote" if a.get("remote_support") == "Yes" else "🏢 On-site"
                        adaptive_display = "🔄 Adaptive" if a.get("adaptive_support") == "Yes" else ""
                        desc = a.get("description", "No description available.")
                        desc_short = desc[:250] + "..." if len(desc) > 250 else desc

                        st.markdown(f"""
<div class="assessment-card">
    <div style="display:flex; justify-content:space-between; align-items:flex-start;">
        <div>
            <strong style="font-size:1.05rem">#{i} — <a href="{a['url']}" target="_blank" style="color:#1a1a2e; text-decoration:none">{a['name']}</a></strong>
        </div>
        <div>
            <span class="meta-pill">{duration_display}</span>
            <span class="meta-pill">{remote_display}</span>
            {"<span class='meta-pill'>" + adaptive_display + "</span>" if adaptive_display else ""}
        </div>
    </div>
    <div style="margin-top:6px">{badges_html}</div>
    <p style="color:#555; font-size:0.88rem; margin-top:8px">{desc_short}</p>
    <a href="{a['url']}" target="_blank" style="font-size:0.82rem; color:#4A90D9">View on SHL →</a>
</div>
""", unsafe_allow_html=True)

                else:
                    st.error(f"API Error {response.status_code}: {response.json().get('detail', 'Unknown error')}")

            except requests.exceptions.ConnectionError:
                st.error(f"Cannot connect to API at `{API_BASE}`. Make sure the API server is running.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown(
        "This tool recommends **SHL Individual Test Solutions** based on hiring needs "
        "using a RAG pipeline powered by LlamaIndex + Groq."
    )

    st.markdown("### 📊 Test Type Legend")
    for label, color in TEST_TYPE_COLORS.items():
        st.markdown(
            f'<span style="background:{color}; color:white; padding:2px 8px; border-radius:10px; font-size:0.8rem">{label}</span>',
            unsafe_allow_html=True,
        )
        st.markdown("")

    st.markdown("### 🔧 API Status")
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        if r.status_code == 200:
            st.success("API is healthy ✅")
        else:
            st.error("API returned non-200 status")
    except Exception:
        st.error("API not reachable ❌")