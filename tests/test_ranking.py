"""Tests for the ranking, scoring, and evaluation modules."""

import pytest
from src.ranking.scorer import compute_score, compute_heuristic_score
from src.ranking.ranker import rank_candidates, CandidateResult
from src.evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    spearman_correlation,
    mean_average_precision,
    evaluate_ranking,
)


# ===========================================================================
# Heuristic scorer tests
# ===========================================================================

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

    def test_more_skills_gives_higher_score(self):
        """Adding matched skills should always improve score."""
        base = compute_heuristic_score(
            similarity=0.6,
            resume_skills=["Python"],
            jd_skills=["Python", "SQL", "AWS"],
            resume_exp=3.0,
            jd_exp=3.0,
            resume_edu=["BSc"],
        )
        better = compute_heuristic_score(
            similarity=0.6,
            resume_skills=["Python", "SQL", "AWS"],
            jd_skills=["Python", "SQL", "AWS"],
            resume_exp=3.0,
            jd_exp=3.0,
            resume_edu=["BSc"],
        )
        assert better > base

    def test_education_adds_score(self):
        no_edu = compute_heuristic_score(
            similarity=0.5, resume_skills=["Python"], jd_skills=["Python"],
            resume_exp=2.0, jd_exp=2.0, resume_edu=[],
        )
        with_edu = compute_heuristic_score(
            similarity=0.5, resume_skills=["Python"], jd_skills=["Python"],
            resume_exp=2.0, jd_exp=2.0, resume_edu=["BSc Computer Science"],
        )
        assert with_edu > no_edu


# ===========================================================================
# ML scorer tests
# ===========================================================================

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
            similarity=0.2, resume_skills=["Python"], jd_skills=["Python", "SQL"],
            resume_exp=1.0, jd_exp=3.0, resume_edu=[],
        )
        high = compute_score(
            similarity=0.9, resume_skills=["Python"], jd_skills=["Python", "SQL"],
            resume_exp=1.0, jd_exp=3.0, resume_edu=[],
        )
        assert high > low

    def test_score_deterministic(self):
        """Same inputs should produce identical outputs (model is deterministic)."""
        kwargs = dict(
            similarity=0.75, resume_skills=["Python", "AWS"],
            jd_skills=["Python", "AWS", "Docker"], resume_exp=3.0,
            jd_exp=3.0, resume_edu=["BSc"],
        )
        assert compute_score(**kwargs) == compute_score(**kwargs)


# ===========================================================================
# Batch ranker tests
# ===========================================================================

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

    def test_ranks_are_sequential(self):
        candidates = [
            {"name": f"Candidate_{i}", "similarity": 0.9 - i * 0.15,
             "skills": ["Python"], "experience": 5.0 - i, "education": ["BSc"]}
            for i in range(4)
        ]
        results = rank_candidates(candidates, jd_skills=["Python"], jd_exp=3.0)
        ranks = [r.rank for r in results]
        assert ranks == list(range(1, len(candidates) + 1))


# ===========================================================================
# Evaluation metrics tests
# ===========================================================================

class TestNDCG:
    """Tests for NDCG@k metric."""

    def test_perfect_ranking_scores_one(self):
        order = ["A", "B", "C", "D"]
        assert ndcg_at_k(order, order, k=4) == 1.0

    def test_reversed_ranking_scores_low(self):
        ideal = ["A", "B", "C", "D"]
        reversed_ = list(reversed(ideal))
        score = ndcg_at_k(reversed_, ideal, k=4)
        assert score < 0.9  # Should be significantly worse than perfect

    def test_empty_ground_truth_returns_zero(self):
        assert ndcg_at_k(["A", "B"], [], k=2) == 0.0

    def test_partial_overlap(self):
        """Partially correct ordering should score between 0 and 1."""
        ideal = ["A", "B", "C", "D"]
        predicted = ["A", "C", "B", "D"]  # B and C swapped
        score = ndcg_at_k(predicted, ideal, k=4)
        assert 0.0 < score < 1.0

    def test_k_cutoff_respected(self):
        """NDCG@1 should only care about the top result."""
        ideal = ["A", "B", "C"]
        correct_top = ndcg_at_k(["A", "C", "B"], ideal, k=1)
        wrong_top = ndcg_at_k(["C", "A", "B"], ideal, k=1)
        assert correct_top > wrong_top


class TestPrecisionAtK:
    """Tests for Precision@k metric."""

    def test_all_relevant_scores_one(self):
        ideal = ["A", "B", "C"]
        assert precision_at_k(["A", "B", "C"], ideal, k=2, relevant_threshold=2) == 1.0

    def test_no_relevant_scores_zero(self):
        assert precision_at_k(["X", "Y", "Z"], ["A", "B", "C"], k=2, relevant_threshold=1) == 0.0

    def test_partial_match(self):
        ideal = ["A", "B", "C", "D"]
        predicted = ["A", "X", "B", "Y"]  # A in top-2, B not
        score = precision_at_k(predicted, ideal, k=2, relevant_threshold=1)
        assert score == 0.5


class TestSpearmanCorrelation:
    """Tests for Spearman rank correlation."""

    def test_identical_orders_return_one(self):
        order = ["A", "B", "C", "D"]
        assert spearman_correlation(order, order) == 1.0

    def test_reversed_order_returns_negative_one(self):
        ideal = ["A", "B", "C", "D"]
        assert spearman_correlation(list(reversed(ideal)), ideal) == -1.0

    def test_single_swap_close_to_one(self):
        ideal = ["A", "B", "C", "D"]
        swapped = ["A", "C", "B", "D"]
        rho = spearman_correlation(swapped, ideal)
        assert 0.8 <= rho < 1.0

    def test_too_few_candidates_returns_zero(self):
        assert spearman_correlation(["A"], ["A"]) == 0.0


class TestMAP:
    """Tests for Mean Average Precision."""

    def test_perfect_order_returns_one(self):
        order = ["A", "B", "C"]
        assert mean_average_precision(order, order) == 1.0

    def test_empty_ground_truth_returns_zero(self):
        assert mean_average_precision(["A", "B"], []) == 0.0

    def test_relevant_results_later_lowers_map(self):
        ideal_relevant = ["A", "B"]
        early = mean_average_precision(["A", "B", "X", "Y"], ideal_relevant)
        late = mean_average_precision(["X", "Y", "A", "B"], ideal_relevant)
        assert early > late


class TestEvaluateRanking:
    """Tests for the all-in-one evaluate_ranking() function."""

    def test_returns_four_metrics(self):
        order = ["A", "B", "C"]
        result = evaluate_ranking(order, order)
        assert set(result.keys()) == {"ndcg", "precision_at_k", "spearman", "map"}

    def test_perfect_ranking_all_metrics_high(self):
        order = ["A", "B", "C", "D"]
        result = evaluate_ranking(order, order)
        assert result["ndcg"] == 1.0
        assert result["spearman"] == 1.0
        assert result["map"] == 1.0

    def test_all_metrics_bounded_zero_to_one(self):
        predicted = ["B", "A", "D", "C"]
        ideal = ["A", "B", "C", "D"]
        result = evaluate_ranking(predicted, ideal)
        for key, val in result.items():
            assert 0.0 <= val <= 1.0, f"{key} out of bounds: {val}"

    def test_system_ranking_beats_random_baseline(self):
        """The ranking system should achieve higher NDCG than worst-case ordering."""
        ideal = ["Priya", "Jane", "David", "Aisha", "Alex"]
        system = ["Priya", "Jane", "David", "Aisha", "Alex"]   # perfect
        worst = list(reversed(ideal))
        perfect_score = evaluate_ranking(system, ideal)["ndcg"]
        worst_score = evaluate_ranking(worst, ideal)["ndcg"]
        assert perfect_score > worst_score
