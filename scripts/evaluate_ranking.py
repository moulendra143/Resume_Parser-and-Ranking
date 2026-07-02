"""
Ranking Evaluation Script
=========================

Generates a synthetic ground-truth dataset of candidates with known ideal
rankings, runs them through the ML ranking pipeline, then measures accuracy
using standard Information Retrieval metrics:

    - NDCG@k  (Normalised Discounted Cumulative Gain)
    - Precision@k
    - Spearman Rank Correlation
    - Mean Average Precision (MAP)

Usage
-----
    python scripts/evaluate_ranking.py

    # With a specific k cutoff
    python scripts/evaluate_ranking.py --k 3

    # Save results to JSON
    python scripts/evaluate_ranking.py --output results/eval_report.json

The script is fully self-contained — no real resume files required.
It models a realistic ML Engineer job description and 8 candidates with
varying levels of fit, covering common edge cases (over-qualified, under-
qualified, skill-gap, wrong domain).
"""

import argparse
import json
import os
import sys

# Ensure project root is on path when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ranking.ranker import rank_candidates
from src.evaluation.metrics import evaluate_ranking


# ---------------------------------------------------------------------------
# Ground-truth dataset
# ---------------------------------------------------------------------------

JD_SKILLS = ["Python", "Machine Learning", "TensorFlow", "AWS", "SQL", "Docker", "Git"]
JD_EXP = 3.0
JD_DESCRIPTION = (
    "Senior ML Engineer with 3+ years of experience in Python, TensorFlow, "
    "AWS, SQL, Docker, and Git.  Strong background in machine learning and "
    "model deployment."
)

# Each candidate has pre-computed similarity scores and features.
# 'ideal_rank' is the expert-assigned ground-truth ranking (1 = best fit).
CANDIDATES = [
    {
        "name": "Priya Patel",
        "ideal_rank": 1,
        "similarity": 0.92,
        "skills": ["Python", "Machine Learning", "TensorFlow", "AWS", "SQL", "Docker", "Git", "Kubernetes"],
        "experience": 5.0,
        "education": ["Master of Technology in Computer Science"],
    },
    {
        "name": "Jane Smith",
        "ideal_rank": 2,
        "similarity": 0.85,
        "skills": ["Python", "Machine Learning", "Scikit-learn", "SQL", "Git", "Pandas", "NumPy"],
        "experience": 6.0,
        "education": ["Bachelor of Science in Data Science"],
    },
    {
        "name": "David Lee",
        "ideal_rank": 3,
        "similarity": 0.78,
        "skills": ["Python", "TensorFlow", "Docker", "Git", "AWS"],
        "experience": 3.0,
        "education": ["Bachelor of Engineering in Computer Science"],
    },
    {
        "name": "Aisha Khan",
        "ideal_rank": 4,
        "similarity": 0.70,
        "skills": ["Python", "SQL", "Machine Learning", "Git"],
        "experience": 2.0,
        "education": ["Bachelor of Science in Mathematics"],
    },
    {
        "name": "Sam Rivera",
        "ideal_rank": 5,
        "similarity": 0.60,
        "skills": ["Python", "SQL", "Pandas", "Git"],
        "experience": 2.0,
        "education": ["Bachelor of Science in Statistics"],
    },
    {
        "name": "Alex Johnson",
        "ideal_rank": 6,
        "similarity": 0.45,
        "skills": ["JavaScript", "React", "Node.js", "CSS", "Git"],
        "experience": 4.0,
        "education": ["Bachelor of Science in Computer Science"],
    },
    {
        "name": "Chris Park",
        "ideal_rank": 7,
        "similarity": 0.30,
        "skills": ["Java", "Spring Boot", "MySQL"],
        "experience": 1.0,
        "education": [],
    },
    {
        "name": "Morgan Taylor",
        "ideal_rank": 8,
        "similarity": 0.15,
        "skills": ["HTML", "CSS", "Excel"],
        "experience": 0.0,
        "education": [],
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ideal_order(candidates: list) -> list:
    """Return candidate names sorted by ideal_rank (ascending = best first)."""
    return [c["name"] for c in sorted(candidates, key=lambda c: c["ideal_rank"])]


def _print_table(ranked, ideal_order):
    """Pretty-print the ranking comparison table."""
    print("\n" + "=" * 80)
    print(f"  {'Rank':<6} {'System Rank':<14} {'Candidate':<20} {'Score':<10} {'Ideal Rank'}")
    print("-" * 80)
    ideal_positions = {name: i + 1 for i, name in enumerate(ideal_order)}
    for r in ranked:
        ideal = ideal_positions.get(r.name, "?")
        match = "OK" if r.rank == ideal else ("^" if r.rank < ideal else "v")
        print(f"  {match:<6} #{r.rank:<13} {r.name:<20} {r.score:<10.2f} #{ideal}")
    print("=" * 80)


def _print_metrics(metrics: dict, k: int):
    """Pretty-print evaluation metrics."""
    print(f"\n{'-' * 50}")
    print(f"  RANKING EVALUATION METRICS  (k={k})")
    print(f"{'-' * 50}")
    print(f"  NDCG@{k:<3}              {metrics['ndcg']:.4f}  {'*****' if metrics['ndcg'] > 0.9 else '****' if metrics['ndcg'] > 0.7 else '***'}")
    print(f"  Precision@{k//2 or 1:<2}          {metrics['precision_at_k']:.4f}")
    print(f"  Spearman rho         {metrics['spearman']:.4f}  {'(strong)' if abs(metrics['spearman']) > 0.8 else '(moderate)' if abs(metrics['spearman']) > 0.5 else '(weak)'}")
    print(f"  MAP                  {metrics['map']:.4f}")
    print(f"{'-' * 50}")

    overall = (metrics["ndcg"] + metrics["precision_at_k"] + max(0, metrics["spearman"]) + metrics["map"]) / 4
    print(f"  Overall accuracy:    {overall:.4f} / 1.0000")
    print(f"{'-' * 50}\n")



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_evaluation(k: int = 5) -> dict:
    """Run the full evaluation pipeline and return the metrics dict."""
    print(f"\n  Job Description Skills : {', '.join(JD_SKILLS)}")
    print(f"  Required Experience    : {JD_EXP} years")
    print(f"  Number of candidates   : {len(CANDIDATES)}")

    # Run ranking
    ranked = rank_candidates(CANDIDATES, jd_skills=JD_SKILLS, jd_exp=JD_EXP)
    predicted_order = [r.name for r in ranked]
    ideal_order = _ideal_order(CANDIDATES)

    print("\n  Ideal order (ground truth):")
    for i, name in enumerate(ideal_order, 1):
        print(f"    #{i}: {name}")

    _print_table(ranked, ideal_order)

    metrics = evaluate_ranking(predicted_order, ideal_order, k=k)
    _print_metrics(metrics, k)

    return {
        "predicted_order": predicted_order,
        "ideal_order": ideal_order,
        "metrics": metrics,
        "k": k,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the ML ranking pipeline against a ground-truth candidate dataset."
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Cutoff k for NDCG@k and Precision@k (default: 5)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional path to save JSON evaluation report (e.g. results/eval_report.json)"
    )
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("  RESUME RANKING SYSTEM — EVALUATION REPORT")
    print("=" * 80)

    results = run_evaluation(k=args.k)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
