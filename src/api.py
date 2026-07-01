import os
import shutil
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

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

app = FastAPI(title="Resume Screening API", description="API for parsing and ranking resumes against a job description.")

# Load taxonomy once on startup
TAXONOMY_PATH = "config/skills_taxonomy.json"
taxonomy = load_skills_taxonomy(TAXONOMY_PATH)

def extract_text_from_upload(upload_file: UploadFile) -> str:
    """Helper to save uploaded file temporarily and extract text."""
    temp_file_path = f"temp_{upload_file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
        
    ext = os.path.splitext(temp_file_path)[1].lower()
    text = ""
    try:
        if ext == ".pdf":
            text = extract_pdf(temp_file_path)
        elif ext == ".docx":
            text = extract_docx(temp_file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
    return text

@app.get("/")
def read_root():
    return {"message": "Welcome to the Resume Screening API"}

@app.post("/api/v1/rank")
async def rank_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):
    """
    Endpoint to upload a resume file and a job description text,
    and get back the extracted features and ML-predicted fit score.
    """
    if not resume.filename.endswith(('.pdf', '.docx')):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")
        
    try:
        # Extract text
        resume_text = extract_text_from_upload(resume)
        jd_text = job_description
        
        # Extract features
        resume_skills = extract_skills(resume_text, taxonomy)
        resume_exp = extract_experience(resume_text)
        resume_edu = extract_education(resume_text)
        
        jd_skills = extract_skills(jd_text, taxonomy)
        jd_exp = extract_experience(jd_text)
        
        # Clean text
        resume_clean = clean_text(resume_text)
        jd_clean = clean_text(jd_text)
        
        # Compute embeddings similarity
        sim_score = compute_similarity(resume_clean, jd_clean)
        
        # Score candidate using ML ranker
        final_score = compute_score(
            similarity=sim_score,
            resume_skills=resume_skills,
            jd_skills=jd_skills,
            resume_exp=resume_exp,
            jd_exp=jd_exp,
            resume_edu=resume_edu
        )
        
        return JSONResponse(content={
            "candidate_name": resume.filename,
            "final_score_ml": final_score,
            "features_extracted": {
                "skills": resume_skills,
                "experience_years": resume_exp,
                "education": resume_edu
            },
            "similarity_score": round(sim_score, 4)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
