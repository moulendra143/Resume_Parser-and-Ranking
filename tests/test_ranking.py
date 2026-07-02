"""Tests for the ranking and scoring modules."""

import pytest
from src.ranking.scorer import compute_score, compute_heuristic_score
from src.ranking.ranker import rank_candidates, CandidateResult


class TestHeuristicScorer:
    """Tests for the heuristic fallback scorer."""

    def test_perfect_candidate(self):
        score = compute_heuristic_score(
            similarity=1.0,
            resume_skills=["Python", "SQL"],
            jd_skills=["Python", "SQL"],
            resume_exp=5.0,
            jd_exp=3.0,
            resume_edu=["Bachelor of Science"],
        )
        assert score == 100.0

    def test_zero_similarity(self):
        score = compute_heuristic_score(
            similarity=0.0,
            resume_skills=[],
            jd_skills=["Python"],
            resume_exp=0.0,
            jd_exp=3.0,
            resume_edu=[],
        )
        assert score == 0.0

    def test_score_within_bounds(self):
        score = compute_heuristic_score(
            similarity=0.5,
            resume_skills=["Python"],
            jd_skills=["Python", "Java", "SQL"],
            resume_exp=2.0,
            jd_exp=5.0,
            resume_edu=["BSc"],
        )
        assert 0 <= score <= 100


class TestMLScorer:
    """Tests for the ML-based scorer."""

    def test_returns_numeric_score(self):
        score = compute_score(
            similarity=0.8,
            resume_skills=["Python", "SQL"],
            jd_skills=["Python", "SQL", "AWS"],
            resume_exp=4.0,
            jd_exp=3.0,
            resume_edu=["Bachelor of Science"],
        )
        assert isinstance(score, float)
        assert 0 <= score <= 100

    def test_higher_similarity_gives_higher_score(self):
        low = compute_score(
            similarity=0.2,
            resume_skills=["Python"],
            jd_skills=["Python", "SQL"],
            resume_exp=1.0,
            jd_exp=3.0,
            resume_edu=[],
        )
        high = compute_score(
            similarity=0.9,
            resume_skills=["Python"],
            jd_skills=["Python", "SQL"],
            resume_exp=1.0,
            jd_exp=3.0,
            resume_edu=[],
        )
        assert high > low


class TestBatchRanker:
    """Tests for the batch candidate ranking module."""

    def test_ranks_candidates_in_order(self):
        candidates = [
            {
                "name": "Weak Candidate",
                "similarity": 0.2,
                "skills": [],
                "experience": 0.0,
                "education": [],
            },
            {
                "name": "Strong Candidate",
                "similarity": 0.95,
                "skills": ["Python", "SQL", "AWS"],
                "experience": 5.0,
                "education": ["BSc Computer Science"],
            },
        ]
        results = rank_candidates(candidates, jd_skills=["Python", "SQL", "AWS"], jd_exp=3.0)
        assert results[0].name == "Strong Candidate"
        assert results[0].rank == 1
        assert results[1].rank == 2

    def test_empty_candidates(self):
        results = rank_candidates([], jd_skills=["Python"], jd_exp=2.0)
        assert results == []

    def test_result_type(self):
        candidates = [{"name": "A", "similarity": 0.5, "skills": [], "experience": 1, "education": []}]
        results = rank_candidates(candidates, jd_skills=["Python"], jd_exp=2.0)
        assert isinstance(results[0], CandidateResult)
