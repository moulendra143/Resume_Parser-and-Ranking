"""
Streamlit application for the Resume Screening & Ranking System.
Simplified clean interface showing only Job Description input, Resume upload, and Analysis Trigger.
"""

import streamlit as st
import tempfile
import os
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.parser.pdf_parser import extract_text as extract_pdf
from src.parser.docx_parser import extract_text as extract_docx
from src.parser.text_cleaner import clean_text
from src.features.skills_extractor import extract_skills
from src.features.experience_extractor import extract_experience
from src.features.education_extractor import extract_education
from src.features.contact_extractor import extract_contact_info
from src.similarity.embedding_matcher import compute_similarity
from src.ranking.scorer import compute_score
from src.main import load_skills_taxonomy

# Page settings
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load taxonomy
@st.cache_data
def get_taxonomy():
    try:
        return load_skills_taxonomy("config/skills_taxonomy.json")
    except Exception:
        return ["Python", "SQL", "AWS", "Machine Learning", "Pandas", "Scikit-learn", "TensorFlow", "Git", "Java", "Django", "Linux", "PostgreSQL", "C++", "C#", ".NET"]

taxonomy = get_taxonomy()

# Custom CSS for gorgeous design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    /* Hide sidebar and collapsed controls */
    [data-testid="stSidebar"] {
        display: none !important;
    }
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    
    /* Set main font and background */
    html, body, [class*="st-"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #fafafa;
        color: #1f2937;
    }
    .stApp {
        background-color: #fafafa;
    }
    
    /* Main container styling */
    .main-container {
        max-width: 700px;
        margin: 0 auto;
        padding: 2rem 1rem;
    }
    
    /* Header section */
    .header-title {
        font-size: 2.25rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .header-subtitle {
        font-size: 1rem;
        text-align: center;
        color: #6b7280;
        margin-bottom: 2.5rem;
    }
    
    /* Style the Streamlit bordered containers as clean white cards */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 16px !important;
        padding: 2rem !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important;
        margin-bottom: 2rem !important;
    }
    
    /* Input Labels */
    .input-label {
        font-size: 1rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* File uploader styling targeting dropzone to prevent button/label alignment issues */
    [data-testid="stFileUploaderDropzone"] {
        border: 2px dashed #6366f1 !important;
        background-color: #f8fafc !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        transition: all 0.2s ease-in-out !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #4f46e5 !important;
        background-color: #eef2ff !important;
    }
    
    /* Translation-proof browse files button styling */
    [data-testid="stFileUploaderDropzone"] button {
        border: 1px solid #d1d5db !important;
        background-color: #ffffff !important;
        padding: 0.5rem 1rem !important;
        border-radius: 8px !important;
        transition: all 0.2s ease-in-out !important;
        height: auto !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover {
        border-color: #6366f1 !important;
        background-color: #f8fafc !important;
    }
    [data-testid="stFileUploaderDropzone"] button * {
        display: none !important;
    }
    [data-testid="stFileUploaderDropzone"] button::after {
        content: "Browse files" !important;
        display: inline-block !important;
        color: #4f46e5 !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* Textarea clean style */
    div[data-element-type="textarea"] textarea {
        border: 1px solid #d1d5db !important;
        border-radius: 10px !important;
        background-color: #ffffff !important;
        font-size: 0.95rem !important;
        padding: 12px !important;
        transition: border-color 0.2s ease;
    }
    div[data-element-type="textarea"] textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
    }

    /* Primary button style */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.8rem 1.6rem !important;
        width: 100% !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.3) !important;
        transition: all 0.2s ease-in-out !important;
        font-size: 1.05rem !important;
    }
    .stButton > button:hover {
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.45) !important;
        transform: translateY(-2px) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
    }
    
    /* Results formatting */
    .result-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 0.5rem;
        text-align: center;
    }
    .score-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        font-size: 2rem;
        font-weight: 800;
        width: 100px;
        height: 100px;
        border-radius: 50%;
        margin: 0 auto 1.5rem auto;
        box-shadow: 0 4px 10px rgba(16, 185, 129, 0.3);
    }
    
    .detail-row {
        display: flex;
        justify-content: space-between;
        padding: 0.75rem 0;
        border-bottom: 1px solid #f3f4f6;
    }
    .detail-label {
        font-weight: 600;
        color: #4b5563;
    }
    .detail-value {
        color: #111827;
        font-weight: 500;
    }
    
    .skill-pill {
        display: inline-block;
        background-color: #eef2ff;
        color: #4f46e5;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 4px 4px 4px 0;
        border: 1px solid #e0e7ff;
    }
    .skill-pill-missing {
        display: inline-block;
        background-color: #fff1f2;
        color: #e11d48;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 4px 4px 4px 0;
        border: 1px solid #ffe4e6;
    }
</style>
""", unsafe_allow_html=True)

# Main UI container
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown('<div class="header-title">🎯 AI Resume Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="header-subtitle">Instantly screen and match candidate profiles against your job requirements</div>', unsafe_allow_html=True)

# Wrap inputs inside a native Streamlit border container (styled as a beautiful card via CSS)
with st.container(border=True):
    # 1. Job Description Field
    st.markdown('<div class="input-label">📝 Step 1: Job Description</div>', unsafe_allow_html=True)
    jd_input = st.text_area(
        "Job Description",
        height=200,
        placeholder="Paste the job description here...",
        label_visibility="collapsed"
    )

    st.markdown('<div style="height: 1.5rem;"></div>', unsafe_allow_html=True)

    # 2. Upload Resume Field
    st.markdown('<div class="input-label">📁 Step 2: Upload Candidate Resume</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "",
        type=["pdf", "docx", "txt"],
        label_visibility="collapsed"
    )

    st.markdown('<div style="height: 1.5rem;"></div>', unsafe_allow_html=True)

    # Submit button
    submit_clicked = st.button("Analyse Resume", type="primary", use_container_width=True)


def parse_uploaded_file(file_obj):
    ext = os.path.splitext(file_obj.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file_obj.read())
        tmp_path = tmp.name
    try:
        if ext == ".pdf":
            return extract_pdf(tmp_path)
        elif ext == ".docx":
            return extract_docx(tmp_path)
        else:
            with open(tmp_path, "r", encoding="utf-8") as f:
                return f.read()
    finally:
        os.remove(tmp_path)

if submit_clicked:
    if not jd_input.strip():
        st.error("⚠️ Please provide a Job Description.")
    elif not uploaded_file:
        st.error("⚠️ Please upload a candidate resume file.")
    else:
        with st.spinner("Analyzing candidate resume..."):
            try:
                # Parse resume
                resume_raw = parse_uploaded_file(uploaded_file)
                
                # Extraction
                resume_skills = extract_skills(resume_raw, taxonomy)
                resume_exp = extract_experience(resume_raw)
                resume_edu = extract_education(resume_raw)
                contact = extract_contact_info(resume_raw)

                jd_skills = extract_skills(jd_input, taxonomy)
                jd_exp = extract_experience(jd_input)

                resume_clean = clean_text(resume_raw)
                jd_clean = clean_text(jd_input)

                # Similarity
                emb_sim = compute_similarity(resume_clean, jd_clean)
                
                # Scoring
                final_score = compute_score(
                    similarity=emb_sim,
                    resume_skills=resume_skills,
                    jd_skills=jd_skills,
                    resume_exp=resume_exp,
                    jd_exp=jd_exp,
                    resume_edu=resume_edu,
                )
                
                # Display Results Card
                with st.container(border=True):
                    st.markdown('<div class="result-title">Analysis Results</div>', unsafe_allow_html=True)
                    
                    # Score Circle Centering Container
                    score_percentage = int(final_score)
                    badge_bg = "linear-gradient(135deg, #10b981 0%, #059669 100%)" # Green
                    if score_percentage < 50:
                        badge_bg = "linear-gradient(135deg, #ef4444 0%, #dc2626 100%)" # Red
                    elif score_percentage < 75:
                        badge_bg = "linear-gradient(135deg, #f59e0b 0%, #d97706 100%)" # Orange
                    
                    st.markdown(f'<div style="text-align: center;"><div class="score-badge" style="background: {badge_bg};">{score_percentage}%</div></div>', unsafe_allow_html=True)
                    st.markdown(f'<h4 style="text-align: center;">Match Rating: {"Excellent" if score_percentage >= 75 else "Good" if score_percentage >= 50 else "Weak"}</h4>', unsafe_allow_html=True)
                    
                    # Extracted Info Card
                    st.markdown('<div style="text-align: left; margin-top: 2rem;">', unsafe_allow_html=True)
                    st.markdown('<h4 style="font-weight:700; margin-bottom:1rem; color:#374151;">📋 Candidate Overview</h4>', unsafe_allow_html=True)
                    
                    # Contacts
                    emails = contact.get("emails", [])
                    phones = contact.get("phones", [])
                    linkedin = contact.get("linkedin", [])
                    
                    email_str = emails[0] if emails else "Not detected"
                    phone_str = phones[0] if phones else "Not detected"
                    linkedin_str = linkedin[0] if linkedin else "Not detected"
                    
                    st.markdown(f"""
                    <div class="detail-row"><span class="detail-label">📧 Email</span><span class="detail-value">{email_str}</span></div>
                    <div class="detail-row"><span class="detail-label">📞 Phone</span><span class="detail-value">{phone_str}</span></div>
                    <div class="detail-row"><span class="detail-label">🔗 LinkedIn</span><span class="detail-value">{linkedin_str}</span></div>
                    """, unsafe_allow_html=True)
                    
                    # Experience
                    st.markdown(f"""
                    <div class="detail-row"><span class="detail-label">⏱️ Total Experience</span><span class="detail-value">{resume_exp} Years (Required: {jd_exp} Years)</span></div>
                    """, unsafe_allow_html=True)
                    
                    # Education
                    edu_str = resume_edu[0] if resume_edu else "Not detected"
                    st.markdown(f"""
                    <div class="detail-row"><span class="detail-label">🎓 Education</span><span class="detail-value">{edu_str}</span></div>
                    """, unsafe_allow_html=True)
                    
                    # Skills
                    matched_skills = set(resume_skills) & set(jd_skills)
                    missing_skills = set(jd_skills) - set(resume_skills)
                    
                    st.markdown('<h4 style="font-weight:700; margin-top:1.5rem; margin-bottom:0.5rem; color:#374151;">🛠️ Skills Analysis</h4>', unsafe_allow_html=True)
                    
                    st.markdown('<div><b>Matched Skills:</b></div>', unsafe_allow_html=True)
                    if matched_skills:
                        skills_html = "".join(f'<span class="skill-pill">{s}</span>' for s in matched_skills)
                        st.markdown(f'<div>{skills_html}</div>', unsafe_allow_html=True)
                    else:
                        st.write("No matching taxonomy skills detected.")
                        
                    st.markdown('<div style="height: 0.5rem;"></div>', unsafe_allow_html=True)
                    st.markdown('<div><b>Missing required skills:</b></div>', unsafe_allow_html=True)
                    if missing_skills:
                        missing_html = "".join(f'<span class="skill-pill-missing">{s}</span>' for s in missing_skills)
                        st.markdown(f'<div>{missing_html}</div>', unsafe_allow_html=True)
                    else:
                        st.write("None")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"An error occurred during parsing: {e}")

st.markdown('</div>', unsafe_allow_html=True) # End of main-container
