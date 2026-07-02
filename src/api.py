"""
FastAPI REST server for the Resume Screening & Ranking System.

Endpoints:
  GET  /                           → Health check
  POST /api/v1/rank                → Score a single resume
  POST /api/v1/rank-batch          → Score and rank multiple resumes
"""

import os
import shutil
from typing import List

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse

from src.parser.pdf_parser import extract_text as extract_pdf
from src.parser.docx_parser import extract_text as extract_docx
from src.parser.text_cleaner import clean_text
from src.features.skills_extractor import extract_skills
from src.features.experience_extractor import extract_experience
from src.features.education_extractor import extract_education
from src.features.contact_extractor import extract_contact_info
from src.similarity.embedding_matcher import compute_similarity
from src.similarity.tfidf_matcher import compute_tfidf_similarity
from src.ranking.scorer import compute_score
from src.ranking.ranker import rank_candidates
from src.main import load_skills_taxonomy

app = FastAPI(
    title="Resume Screening & Ranking API",
    description="API for parsing resumes, extracting features, and ranking candidates against a job description.",
    version="1.0.0",
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load taxonomy once on startup
TAXONOMY_PATH = "config/skills_taxonomy.json"
taxonomy = load_skills_taxonomy(TAXONOMY_PATH)


def extract_text_from_upload(upload_file: UploadFile) -> str:
    """Save uploaded file temporarily and extract text."""
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
        elif ext == ".txt":
            with open(temp_file_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    return text


def _score_resume(resume_text: str, jd_text: str) -> dict:
    """Score a single resume against a job description. Returns feature dict."""
    resume_skills = extract_skills(resume_text, taxonomy)
    resume_exp = extract_experience(resume_text)
    resume_edu = extract_education(resume_text)
    contact = extract_contact_info(resume_text)

    jd_skills = extract_skills(jd_text, taxonomy)
    jd_exp = extract_experience(jd_text)

    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(jd_text)

    emb_sim = compute_similarity(resume_clean, jd_clean)
    tfidf_sim = compute_tfidf_similarity(resume_clean, jd_clean)

    final_score = compute_score(
        similarity=emb_sim,
        resume_skills=resume_skills,
        jd_skills=jd_skills,
        resume_exp=resume_exp,
        jd_exp=jd_exp,
        resume_edu=resume_edu,
    )

    return {
        "skills": resume_skills,
        "experience_years": resume_exp,
        "education": resume_edu,
        "contact": contact,
        "embedding_similarity": round(emb_sim, 4),
        "tfidf_similarity": round(tfidf_sim, 4),
        "ml_score": final_score,
    }


@app.get("/")
def read_root():
    return {"message": "Resume Screening & Ranking API is running.", "version": "1.0.0"}


@app.post("/api/v1/rank")
async def rank_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
):
    """Upload a single resume and get back extracted features and ML-predicted fit score."""
    if not resume.filename.endswith((".pdf", ".docx", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF, DOCX, and TXT files are supported.")

    try:
        resume_text = extract_text_from_upload(resume)
        result = _score_resume(resume_text, job_description)
        result["candidate_name"] = resume.filename
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/rank-batch")
async def rank_resumes_batch(
    resumes: List[UploadFile] = File(...),
    job_description: str = Form(...),
):
    """Upload multiple resumes and get a ranked list of candidates."""
    candidates = []
    jd_skills = extract_skills(job_description, taxonomy)
    jd_exp = extract_experience(job_description)

    for resume_file in resumes:
        if not resume_file.filename.endswith((".pdf", ".docx", ".txt")):
            continue

        try:
            resume_text = extract_text_from_upload(resume_file)
            resume_clean = clean_text(resume_text)
            jd_clean = clean_text(job_description)
            emb_sim = compute_similarity(resume_clean, jd_clean)

            candidates.append({
                "name": resume_file.filename,
                "similarity": emb_sim,
                "skills": extract_skills(resume_text, taxonomy),
                "experience": extract_experience(resume_text),
                "education": extract_education(resume_text),
            })
        except Exception:
            continue

    if not candidates:
        raise HTTPException(status_code=400, detail="No valid resumes could be processed.")

    ranked = rank_candidates(candidates, jd_skills=jd_skills, jd_exp=jd_exp)

    return JSONResponse(content={
        "total_candidates": len(ranked),
        "ranking": [
            {
                "rank": r.rank,
                "name": r.name,
                "score": r.score,
                "skills_matched": len(r.skills),
                "experience_years": r.experience,
            }
            for r in ranked
        ],
    })


@app.post("/api/v1/sandbox/skills")
async def sandbox_skills(text: str = Form(...)):
    skills = extract_skills(text, taxonomy)
    return {"skills": skills}


@app.post("/api/v1/sandbox/experience")
async def sandbox_experience(text: str = Form(...)):
    years = extract_experience(text)
    return {"experience_years": years}


@app.post("/api/v1/sandbox/education")
async def sandbox_education(text: str = Form(...)):
    education = extract_education(text)
    return {"education": education}


@app.post("/api/v1/sandbox/contact")
async def sandbox_contact(text: str = Form(...)):
    contact = extract_contact_info(text)
    return {"contact": contact}


# Serve Frontend static assets if directory exists
static_dir = "frontend/dist" if os.path.exists("frontend/dist") else "frontend"
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
else:
    @app.get("/")
    async def index():
        return {
            "status": "ok",
            "message": "FastAPI Resume Screening & Ranking Server is running. (Frontend folder is excluded as out-of-scope)."
        }

