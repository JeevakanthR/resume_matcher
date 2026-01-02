import streamlit as st
from collections import defaultdict
from typing import List, Tuple

from langchain_community.vectorstores import Chroma

from simple_embeddings import SimpleHFEmbeddings


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="TalentIQ: AI Resume Matcher",
    layout="wide",
    page_icon="💼",
)


# ---------------- DARK THEME CSS ----------------
st.markdown(
    """
    <style>
    body, .stApp {
        background-color: #050509;
        color: #ffffff;
        font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
    }

    header, .stApp > header {
        background-color: #6c63ff !important;
    }

    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 3rem;
        padding-left: 4rem;
        padding-right: 4rem;
    }

    h1 {
        font-size: 3.0rem;
        font-weight: 800;
        letter-spacing: 0.02em;
        color: #ffffff;
        margin-bottom: 0.2rem;
    }

    .tiq-subtitle {
        font-size: 0.98rem;
        font-weight: 400;
        color: #e5e5ea;
        max-width: 820px;
        line-height: 1.6;
        margin-top: 0.2rem;
        margin-bottom: 2.4rem;
    }

    .tiq-label {
        font-size: 0.98rem;
        font-weight: 600;
        color: #f5f5f7;
        margin-bottom: 0.35rem;
        margin-top: 0.6rem;
    }

    textarea {
        background-color: #ffffff !important;
        color: #000000 !important;
        font-size: 0.95rem !important;
        border-radius: 14px !important;
        border: 1px solid #d0d0d5 !important;
        padding: 14px 14px !important;
    }

    .stNumberInput > label {
        font-size: 0.98rem;
        font-weight: 600;
        color: #f5f5f7;
        margin-bottom: 0.35rem;
    }

    .stNumberInput input {
        background-color: #ffffff !important;
        color: #000000 !important;
        border-radius: 10px !important;
        border: 1px solid #d0d0d5 !important;
        font-size: 0.95rem !important;
    }

    .stButton > button {
        background-color: #6c63ff !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border-radius: 999px !important;
        border: none !important;
        height: 48px !important;
        padding: 0 40px !important;
        font-size: 1.0rem !important;
        box-shadow: 0 8px 22px rgba(108, 99, 255, 0.45) !important;
    }
    .stButton > button:hover {
        filter: brightness(1.06);
    }

    .tiq-results-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 0.8rem;
    }
    .tiq-resume-card {
        background-color: #101013;
        padding: 22px;
        border-radius: 14px;
        margin-bottom: 22px;
        font-size: 0.94rem;
        white-space: pre-wrap;
        border: 1px solid #27272f;
    }
    .tiq-source {
        color: #9fa0a5;
        font-size: 0.86rem;
        margin-bottom: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- HEADER ----------------
st.markdown(
    "<h1>TalentIQ: AI Resume Matcher</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="tiq-subtitle">
    An AI-powered resume matching system that compares resumes with job descriptions using semantic search and RAG. 
    It analyzes skills and experience to identify the most relevant candidates. The platform presents ranked results with clear explanations to support better hiring decisions.
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------- LAYOUT ----------------
main_col, _ = st.columns([2.4, 1])

with main_col:
    st.markdown("<div class='tiq-label'>Paste Job Description</div>", unsafe_allow_html=True)
    job_desc = st.text_area(
        label="",
        height=190,
        placeholder="Example: Seeking a Software Engineer with Python, SQL, and cloud experience.",
    )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    top_k = st.number_input(
        "Number of resumes to match",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
    )

    st.markdown("<div style='height: 26px;'></div>", unsafe_allow_html=True)

    left, center, right = st.columns([1, 1, 1])
    with center:
        match_clicked = st.button("Match Resumes", use_container_width=False)


# ---------------- BACKEND SETUP ----------------
embeddings = SimpleHFEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
db = Chroma(persist_directory="vector_db", embedding_function=embeddings)


# --------- RANKING: GROUP BY RESUME FILE ---------
def get_ranked_resumes(query: str, k_resumes: int):
    """
    Return list of (source_file, [chunks]) sorted by best similarity score.
    """
    raw = db.similarity_search_with_score(query, k=k_resumes * 8)

    best_score = defaultdict(lambda: float("inf"))
    chunks_by_source = defaultdict(list)

    for doc, score in raw:
        source = doc.metadata.get("source", "Unknown")
        chunks_by_source[source].append(doc.page_content)
        if score < best_score[source]:
            best_score[source] = score

    ranked_sources = sorted(best_score.items(), key=lambda x: x[1])[:k_resumes]

    result = []
    for source, _ in ranked_sources:
        result.append((source, chunks_by_source[source]))
    return result


# --------- SIMPLE RULE‑BASED EXPLANATION ---------
def build_explanation(job_desc: str, resume_text: str) -> str:
    job_lower = job_desc.lower()
    resume_lower = resume_text.lower()

    skills = [
        "python", "java", "sql", "excel", "power bi", "tableau",
        "machine learning", "django", "react", "communication",
        "intern", "backend", "frontend", "data analyst",
        "data science", "cloud", "aws", "azure"
    ]

    matched = [s for s in skills if s in job_lower and s in resume_lower]

    lines: List[str] = []
    lines.append("- The resume content is semantically similar to the job description.")

    if matched:
        lines.append(f"- Shared skills/keywords: {', '.join(sorted(set(matched)))}.")

    if "intern" in job_lower and ("intern" in resume_lower or "student" in resume_lower or "college" in resume_lower):
        lines.append("- The candidate profile matches the internship / student requirement.")

    if "backend" in job_lower and "backend" in resume_lower:
        lines.append("- Both the job description and resume emphasize backend development.")

    if "frontend" in job_lower and "frontend" in resume_lower:
        lines.append("- Both focus on frontend / UI development responsibilities.")

    if "data" in job_lower and "data" in resume_lower:
        lines.append("- The resume shows experience with data‑related tasks, matching the role’s focus on data.")

    if len(lines) == 1:
        lines.append("- Overall wording and topics in the resume are close to the job description, leading to a high match score.")

    return "\n".join(lines)


# ---------------- APP RUN ----------------
if match_clicked:
    if not job_desc.strip():
        st.warning("Please enter a job description.")
    else:
        ranked = get_ranked_resumes(job_desc, int(top_k))

        if not ranked:
            st.warning("No resumes found in the database. Please ingest some PDFs first.")
        else:
            st.markdown("<div class='tiq-results-title'>Matching Results</div>", unsafe_allow_html=True)

            for rank, (source, chunks) in enumerate(ranked, start=1):
                combined_text = "\n\n".join(chunks)

                st.markdown(f"### {rank}. Resume Match")
                st.markdown(
                    f"<div class='tiq-source'>Source File: {source}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='tiq-resume-card'>{combined_text}</div>",
                    unsafe_allow_html=True,
                )

                st.markdown("**AI Explanation:**")
                st.write(build_explanation(job_desc, combined_text))
