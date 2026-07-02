"""
CLI entry point for the Resume Screening & Ranking System.

Supports two modes:
  1. Single-resume analysis:  --resume <file> --jd <file>
  2. Batch ranking:           --resume-dir <dir> --jd <file>
"""

import os
import argparse
import json

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
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_skills_taxonomy(path: str) -> list:
    """Load the skills taxonomy from a JSON file."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
            return data.get("skills", [])
    except Exception as e:
        logger.warning(f"Error loading taxonomy: {e}")
        return []


def read_file(filepath: str) -> str:
    """Read text content from a PDF, DOCX, or TXT file."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        return extract_pdf(filepath)
    elif ext == ".docx":
        return extract_docx(filepath)
    elif ext == ".txt":
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    else:
        logger.warning(f"Unsupported file type: {ext}")
        return ""


def process_single_resume(resume_path: str, jd_raw: str, taxonomy: list) -> dict:
    """Process a single resume against a job description.

    Returns a dict with all extracted features and scores.
    """
    resume_raw = read_file(resume_path)
    if not resume_raw:
        return {}

    # Extract features
    resume_skills = extract_skills(resume_raw, taxonomy)
    resume_exp = extract_experience(resume_raw)
    resume_edu = extract_education(resume_raw)
    contact_info = extract_contact_info(resume_raw)

    jd_skills = extract_skills(jd_raw, taxonomy)
    jd_exp = extract_experience(jd_raw)

    # Clean text for similarity
    resume_clean = clean_text(resume_raw)
    jd_clean = clean_text(jd_raw)

    # Compute similarities
    embedding_sim = compute_similarity(resume_clean, jd_clean)
    tfidf_sim = compute_tfidf_similarity(resume_clean, jd_clean)

    # ML score
    final_score = compute_score(
        similarity=embedding_sim,
        resume_skills=resume_skills,
        jd_skills=jd_skills,
        resume_exp=resume_exp,
        jd_exp=jd_exp,
        resume_edu=resume_edu,
    )

    return {
        "name": os.path.basename(resume_path),
        "similarity": embedding_sim,
        "tfidf_similarity": tfidf_sim,
        "skills": resume_skills,
        "experience": resume_exp,
        "education": resume_edu,
        "contact": contact_info,
        "score": final_score,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Resume Screening & Ranking System — Parse, score, and rank candidates."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--resume", type=str, help="Path to a single resume file (PDF/DOCX/TXT)")
    group.add_argument("--resume-dir", type=str, help="Path to a directory of resume files for batch ranking")
    parser.add_argument("--jd", type=str, required=True, help="Path to job description file (TXT)")
    parser.add_argument("--taxonomy", type=str, default="config/skills_taxonomy.json", help="Path to skills taxonomy JSON")
    args = parser.parse_args()

    taxonomy = load_skills_taxonomy(args.taxonomy)
    jd_raw = read_file(args.jd)
    if not jd_raw:
        logger.error("Failed to read job description file.")
        return

    # ---- Single resume mode ----
    if args.resume:
        logger.info(f"Processing resume: {args.resume}")
        result = process_single_resume(args.resume, jd_raw, taxonomy)
        if not result:
            logger.error("Failed to process resume.")
            return

        print("\n" + "=" * 50)
        print("  SINGLE RESUME ANALYSIS")
        print("=" * 50)
        print(f"  File:               {result['name']}")
        print(f"  Embedding Sim:      {result['similarity']:.4f}")
        print(f"  TF-IDF Sim:         {result['tfidf_similarity']:.4f}")
        print(f"  Skills Matched:     {', '.join(result['skills']) or 'None'}")
        print(f"  Experience:         {result['experience']} years")
        print(f"  Education:          {', '.join(result['education']) or 'None'}")
        print("-" * 50)
        print(f"  ML CANDIDATE SCORE: {result['score']} / 100")
        print("=" * 50)

    # ---- Batch ranking mode ----
    elif args.resume_dir:
        if not os.path.isdir(args.resume_dir):
            logger.error(f"Directory not found: {args.resume_dir}")
            return

        files = [
            os.path.join(args.resume_dir, f)
            for f in sorted(os.listdir(args.resume_dir))
            if os.path.splitext(f)[1].lower() in (".pdf", ".docx", ".txt")
        ]

        if not files:
            logger.error("No resume files found in the directory.")
            return

        logger.info(f"Batch ranking {len(files)} resumes...")

        # Process each resume
        candidates = []
        for fpath in files:
            result = process_single_resume(fpath, jd_raw, taxonomy)
            if result:
                candidates.append(result)

        if not candidates:
            logger.error("No resumes could be processed.")
            return

        # Extract JD features for ranking
        jd_skills = extract_skills(jd_raw, taxonomy)
        jd_exp = extract_experience(jd_raw)

        # Rank all candidates
        ranked = rank_candidates(candidates, jd_skills=jd_skills, jd_exp=jd_exp)

        print("\n" + "=" * 70)
        print("  CANDIDATE RANKING RESULTS")
        print("=" * 70)
        print(f"  {'Rank':<6} {'Candidate':<30} {'Score':<10} {'Skills':<8} {'Exp':<6}")
        print("-" * 70)
        for r in ranked:
            print(f"  #{r.rank:<5} {r.name:<30} {r.score:<10} {len(r.skills):<8} {r.experience:<6}")
        print("=" * 70)
        print(f"\n  Top Candidate: #{ranked[0].rank} {ranked[0].name} (Score: {ranked[0].score}/100)")
        print()


if __name__ == "__main__":
    main()
