import streamlit as st
import tempfile
import os
import sys

# Add the parent directory to sys.path so 'src' module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Local imports
from src.parser.pdf_parser import extract_text as extract_pdf
from src.parser.docx_parser import extract_text as extract_docx
from src.parser.text_cleaner import clean_text
from src.features.skills_extractor import extract_skills
from src.features.experience_extractor import extract_experience
from src.features.education_extractor import extract_education
from src.similarity.embedding_matcher import compute_similarity
from src.ranking.scorer import compute_score
from src.main import load_skills_taxonomy

# Page configuration
st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")

st.title("📄 AI Resume Screener & Ranker")
st.markdown("Upload a resume and paste a job description to get an AI-powered candidate fit score.")

# Load taxonomy
@st.cache_data
def get_taxonomy():
    return load_skills_taxonomy("config/skills_taxonomy.json")

taxonomy = get_taxonomy()

# UI Layout: Two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Job Description")
    job_description = st.text_area("Paste the Job Description here", height=300)

with col2:
    st.subheader("2. Candidate Resume")
    uploaded_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])

if st.button("Analyze Candidate", type="primary"):
    if not job_description.strip():
        st.error("Please provide a Job Description.")
    elif not uploaded_file:
        st.error("Please upload a resume file.")
    else:
        with st.spinner("Analyzing Resume & Calculating Fit..."):
            try:
                # Save uploaded file to a temporary location
                ext = os.path.splitext(uploaded_file.name)[1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_path = temp_file.name

                # 1. Parse Resume Text
                resume_text = ""
                if ext == ".pdf":
                    resume_text = extract_pdf(temp_path)
                elif ext == ".docx":
                    resume_text = extract_docx(temp_path)
                
                # Cleanup temp file
                os.remove(temp_path)

                # 2. Extract Features
                resume_skills = extract_skills(resume_text, taxonomy)
                resume_exp = extract_experience(resume_text)
                resume_edu = extract_education(resume_text)
                
                jd_skills = extract_skills(job_description, taxonomy)
                jd_exp = extract_experience(job_description)

                # 3. Clean Text & Compute Similarity
                resume_clean = clean_text(resume_text)
                jd_clean = clean_text(job_description)
                sim_score = compute_similarity(resume_clean, jd_clean)

                # 4. ML Scoring
                final_score = compute_score(
                    similarity=sim_score,
                    resume_skills=resume_skills,
                    jd_skills=jd_skills,
                    resume_exp=resume_exp,
                    jd_exp=jd_exp,
                    resume_edu=resume_edu
                )

                st.success("Analysis Complete!")

                # Display Results
                st.markdown("---")
                st.header(f"🏆 Final Candidate Score: {final_score} / 100")
                
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.metric("Semantic Similarity", f"{sim_score:.2f}")
                    st.metric("Experience Found", f"{resume_exp} Years")
                
                with res_col2:
                    st.subheader("Matched Skills")
                    if resume_skills:
                        st.write(", ".join(resume_skills))
                    else:
                        st.write("None detected.")

                    st.subheader("Education")
                    if resume_edu:
                        for edu in resume_edu:
                            st.write(f"- {edu}")
                    else:
                        st.write("None detected.")

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
