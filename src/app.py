import streamlit as st
from src.pipeline import (
    initialize_pipeline,
    run_pipeline
)

# -------------------------------------------------
# Cache expensive initialization
# -------------------------------------------------

@st.cache_resource
def load_pipeline():
    return initialize_pipeline()


# This runs only once (unless the code changes)
retriever, ranker, job = load_pipeline()

# -------------------------------------------------
# Streamlit UI starts here
# -------------------------------------------------


st.markdown("""
<style>

html, body, [class*="css"]  {
    font-size:18px;
}

h1 {
    font-size:42px !important;
}

h2 {
    font-size:34px !important;
}

h3 {
    font-size:28px !important;
}

p {
    font-size:18px !important;
}

div[data-testid="stMetricValue"] {
    font-size:28px !important;
}

div[data-testid="stMetricLabel"] {
    font-size:18px !important;
}

button[kind="primary"] {
    font-size:22px !important;
    padding:12px 30px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style="text-align:center;">
🤖 AI Recruiter
</h1>

<h3 style="text-align:center;color:gray;">
Hybrid Retrieval + Explainable AI Candidate Ranking
</h3>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="AI Recruiter",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Recruiter")
st.caption("Hybrid Retrieval + Explainable AI Candidate Ranking")

st.divider()


st.markdown("""
### Pipeline

📄 Job Description  
⬇️  
🔍 Hybrid Retrieval (BM25 + Experience Embeddings + RRF)  
⬇️  
📊 Multi-Factor Ranking  
⬇️  
🤖 LLM Candidate Evaluation  
⬇️  
🏆 Top Ranked Candidates
""")

st.divider()

st.subheader("📋 System Overview")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **📄 Job Description**

    Software Engineer (Backend)

    **👥 Candidate Pool**

    100 Candidates
    """)

with col2:
    st.markdown("""
    **🧠 Retrieval**

    • BM25

    • Experience Embeddings

    • Reciprocal Rank Fusion (RRF)

    **🤖 LLM**

    Qwen2.5 (Ollama)
    """)

st.divider()


if st.button("🚀 Run Ranking", type="primary"):

    with st.spinner("Ranking candidates..."):

        rankings = run_pipeline(
            retriever,
            ranker,
            job
        )

    st.success("Ranking completed!")

    st.divider()

    st.header("Top Candidates")

    for i, result in enumerate(rankings[:10], start=1):

        candidate = result["candidate"]

        with st.expander(
            f"#{i} • {candidate.candidate_id} • Score: {result['final_score']:.3f}"
        ):

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Final Score", f"{result['final_score']:.3f}")
                st.metric("Skill Score", f"{result['skill_score']:.3f}")
                st.metric("Experience", f"{result['experience_score']:.3f}")

            with col2:
                st.metric("Evidence", f"{result['evidence_score']:.3f}")
                st.metric("Behavior", f"{result['behavior_score']:.3f}")

            analysis = result["analysis"]

            if analysis:

                st.subheader("Summary")
                st.write(analysis.summary)

                st.subheader("Strengths")

                for strength in analysis.strengths:
                    st.write(f"• {strength.title}")

                if analysis.missing_skills:

                    st.subheader("Missing Skills")

                    for skill in analysis.missing_skills:
                        st.write(f"• {skill}")