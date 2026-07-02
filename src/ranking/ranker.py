"""
Batch candidate ranker.

Accepts a list of parsed resume data and a job description, scores every
candidate using the ML pipeline, and returns a sorted ranking.
"""

from dataclasses import dataclass, field
from typing import List

from src.ranking.scorer import compute_score


@dataclass
class CandidateResult:
    """Holds the scoring result for a single candidate."""
    name: str
    score: float
    skills: list = field(default_factory=list)
    experience: float = 0.0
    education: list = field(default_factory=list)
    similarity: float = 0.0
    rank: int = 0


def rank_candidates(
    candidates: List[dict],
    jd_skills: list,
    jd_exp: float,
) -> List[CandidateResult]:
    """Score and rank a batch of candidates against a job description.

    Each dict in *candidates* must contain::

        {
            "name":       str,
            "similarity": float,       # pre-computed embedding similarity
            "skills":     list[str],
            "experience": float,
            "education":  list[str],
        }

    Args:
        candidates: List of candidate feature dicts.
        jd_skills: Skills extracted from the job description.
        jd_exp: Years of experience required by the job description.

    Returns:
        List of :class:`CandidateResult` sorted by score descending.
    """
    results: List[CandidateResult] = []

    for cand in candidates:
        score = compute_score(
            similarity=cand.get("similarity", 0.0),
            resume_skills=cand.get("skills", []),
            jd_skills=jd_skills,
            resume_exp=cand.get("experience", 0.0),
            jd_exp=jd_exp,
            resume_edu=cand.get("education", []),
        )
        results.append(
            CandidateResult(
                name=cand.get("name", "Unknown"),
                score=score,
                skills=cand.get("skills", []),
                experience=cand.get("experience", 0.0),
                education=cand.get("education", []),
                similarity=cand.get("similarity", 0.0),
            )
        )

    # Sort descending by score
    results.sort(key=lambda r: r.score, reverse=True)

    # Assign rank
    for idx, result in enumerate(results, start=1):
        result.rank = idx

    return results
