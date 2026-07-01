import os
import argparse
import json
import pandas as pd
from pathlib import Path

# Local imports
from src.parser.pdf_parser import extract_text as extract_pdf
from src.parser.docx_parser import extract_text as extract_docx
from src.parser.text_cleaner import clean_text
from src.features.skills_extractor import extract_skills
from src.features.experience_extractor import extract_experience
from src.features.education_extractor import extract_education
from src.similarity.embedding_matcher import compute_similarity
from src.ranking.scorer import compute_score

def load_skills_taxonomy(path: str) -> list:
    try:
        with open(path, "r") as f:
            data = json.load(f)
            return data.get("skills", [])
    except Exception as e:
        print(f"Error loading taxonomy: {e}")
        return []

def read_file(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        return extract_pdf(filepath)
    elif ext == ".docx":
        return extract_docx(filepath)
    elif ext == ".txt":
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    else:
        print(f"Unsupported file type: {ext}")
        return ""

def main():
    parser = argparse.ArgumentParser(description="Resume Screening System")
    parser.add_argument("--resume", type=str, required=True, help="Path to resume file (PDF/DOCX)")
    parser.add_argument("--jd", type=str, required=True, help="Path to job description (TXT)")
    parser.add_argument("--taxonomy", type=str, default="config/skills_taxonomy.json", help="Path to skills taxonomy JSON")
    args = parser.parse_args()
    
    print(f"Processing Resume: {args.resume}")
    print(f"Using Job Description: {args.jd}")
    
    taxonomy = load_skills_taxonomy(args.taxonomy)
    
    # 1. Read files
    resume_raw = read_file(args.resume)
    jd_raw = read_file(args.jd)
    
    if not resume_raw or not jd_raw:
        print("Failed to read input files.")
        return
        
    # 2. Extract features
    resume_skills = extract_skills(resume_raw, taxonomy)
    resume_exp = extract_experience(resume_raw)
    resume_edu = extract_education(resume_raw)
    
    # For JD, let's extract required skills and experience
    jd_skills = extract_skills(jd_raw, taxonomy)
    jd_exp = extract_experience(jd_raw)
    
    # 3. Clean text for similarity matching
    resume_clean = clean_text(resume_raw)
    jd_clean = clean_text(jd_raw)
    
    # 4. Compute similarity
    print("Computing embeddings and similarity...")
    sim_score = compute_similarity(resume_clean, jd_clean)
    
    # 5. Compute final heuristic score
    final_score = compute_score(
        similarity=sim_score,
        resume_skills=resume_skills,
        jd_skills=jd_skills,
        resume_exp=resume_exp,
        jd_exp=jd_exp,
        resume_edu=resume_edu
    )
    
    print("\n" + "="*40)
    print("ANALYSIS RESULTS")
    print("="*40)
    print(f"Similarity Score: {sim_score:.4f}")
    print(f"Extracted Skills: {', '.join(resume_skills)}")
    print(f"Experience Found: {resume_exp} years")
    print(f"Education Found: {', '.join(resume_edu) if resume_edu else 'None'}")
    print("-" * 40)
    print(f"FINAL CANDIDATE SCORE: {final_score} / 100")
    print("="*40)

if __name__ == "__main__":
    main()
