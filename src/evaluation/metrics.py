"""
Ranking evaluation metrics.

Implements standard Information Retrieval ranking quality metrics used to
quantitatively measure how well the ML ranking engine orders candidates
relative to a known ground-truth relevance ordering.

Metrics implemented
-------------------
- **NDCG@k** (Normalised Discounted Cumulative Gain): measures ranking quality
  while accounting for the position of each relevant result.  A perfect ranking
  scores 1.0; random ordering typically scores ~0.5.

- **Precision@k**: fraction of the top-k candidates that are in the ground-truth
  "relevant" set.

- **Spearman Rank Correlation**: measures monotonic agreement between the
  system's rank ordering and the ground-truth ordering.  Range [-1, 1];
  1.0 = perfect agreement, 0.0 = no correlation.

- **Mean Average Precision (MAP)**: average of Precision@k values at each
  relevant result position.  Rewards systems that rank relevant results early.

All metrics accept:
    - ``predicted_order``: list of candidate names sorted by the system (rank 1 first)
    - ``ground_truth_order``: list of candidate names sorted by ideal relevance (rank 1 first)

References
----------
- Manning et al., *Introduction to Information Retrieval*, Ch. 8
- NDCG: Järvelin & Kekäläinen (2002)
"""

import math
from typing import Dict, List


# ---------------------------------------------------------------------------
# Core metric implementations
# ---------------------------------------------------------------------------

def ndcg_at_k(predicted_order: List[str], ground_truth_order: List[str], k: int = 5) -> float:
    """Compute Normalised Discounted Cumulative Gain at rank k.

    Args:
        predicted_order:   Candidate names sorted by system score (best first).
        ground_truth_order: Candidate names sorted by ideal relevance (best first).
        k: Number of top results to evaluate.

    Returns:
        NDCG@k score in [0, 1].  Returns 0.0 if ground_truth_order is empty.

    Example:
        >>> ndcg_at_k(["A", "B", "C"], ["A", "B", "C"], k=3)
        1.0
        >>> ndcg_at_k(["C", "B", "A"], ["A", "B", "C"], k=3)
        # Reversed — score close to but not zero
    """
    if not ground_truth_order:
        return 0.0

    # Build relevance scores: position in ground_truth gives relevance weight
    # Best candidate = relevance len(gt), worst = 1
    n = len(ground_truth_order)
    relevance_map: Dict[str, float] = {
        name: (n - rank) for rank, name in enumerate(ground_truth_order)
    }

    def dcg(order: List[str], cutoff: int) -> float:
        score = 0.0
        for i, name in enumerate(order[:cutoff], start=1):
            rel = relevance_map.get(name, 0.0)
            score += rel / math.log2(i + 1)
        return score

    actual_dcg = dcg(predicted_order, k)
    ideal_dcg = dcg(ground_truth_order, k)

    return round(actual_dcg / ideal_dcg, 4) if ideal_dcg > 0 else 0.0


def precision_at_k(
    predicted_order: List[str],
    ground_truth_order: List[str],
    k: int = 3,
    relevant_threshold: int = 1,
) -> float:
    """Compute Precision@k — fraction of top-k results that are relevant.

    Args:
        predicted_order:    Candidate names sorted by system score (best first).
        ground_truth_order: Candidate names sorted by ideal relevance (best first).
        k:                  Number of top results to evaluate.
        relevant_threshold: Ground-truth candidates ranked within this position
                            are considered "relevant".  E.g. threshold=2 means
                            only the top-2 ideal candidates count as relevant.

    Returns:
        Precision@k in [0, 1].

    Example:
        >>> precision_at_k(["A", "B", "C", "D"], ["A", "B", "C", "D"], k=2)
        1.0  # Both top-2 predicted are in the top-2 ideal
    """
    relevant_set = set(ground_truth_order[:relevant_threshold])
    top_k = predicted_order[:k]
    hits = sum(1 for name in top_k if name in relevant_set)
    return round(hits / k, 4) if k > 0 else 0.0


def spearman_correlation(predicted_order: List[str], ground_truth_order: List[str]) -> float:
    """Compute Spearman rank correlation between two candidate orderings.

    Args:
        predicted_order:    Candidate names sorted by system score (best first).
        ground_truth_order: Candidate names sorted by ideal relevance (best first).

    Returns:
        Spearman ρ in [-1, 1].  1.0 = perfect agreement, 0 = no correlation,
        -1 = perfect inverse.  Returns 0.0 if fewer than 2 candidates are shared.

    Example:
        >>> spearman_correlation(["A","B","C"], ["A","B","C"])
        1.0
        >>> spearman_correlation(["C","B","A"], ["A","B","C"])
        -1.0
    """
    # Only evaluate candidates present in both lists
    common = [c for c in predicted_order if c in ground_truth_order]
    if len(common) < 2:
        return 0.0

    pred_ranks = {name: rank for rank, name in enumerate(predicted_order, start=1)}
    gt_ranks = {name: rank for rank, name in enumerate(ground_truth_order, start=1)}

    n = len(common)
    d_sq_sum = sum((pred_ranks[c] - gt_ranks[c]) ** 2 for c in common)
    rho = 1 - (6 * d_sq_sum) / (n * (n ** 2 - 1))
    return round(rho, 4)


def mean_average_precision(
    predicted_order: List[str],
    ground_truth_order: List[str],
) -> float:
    """Compute Mean Average Precision (MAP).

    MAP rewards systems that place relevant results early.

    Args:
        predicted_order:    Candidate names sorted by system score (best first).
        ground_truth_order: Candidate names sorted by ideal relevance (best first).

    Returns:
        MAP score in [0, 1].  1.0 = all relevant results at the top.

    Example:
        >>> mean_average_precision(["A","B","C","D"], ["A","C","B","D"])
        # A is at rank 1 (hit), C at rank 3 (hit), B at rank 2 (hit) — MAP > 0.8
    """
    if not ground_truth_order:
        return 0.0

    relevant_set = set(ground_truth_order)
    num_relevant = 0
    cumulative_precision = 0.0

    for rank, name in enumerate(predicted_order, start=1):
        if name in relevant_set:
            num_relevant += 1
            cumulative_precision += num_relevant / rank

    if num_relevant == 0:
        return 0.0

    return round(cumulative_precision / len(relevant_set), 4)


# ---------------------------------------------------------------------------
# Convenience all-in-one evaluator
# ---------------------------------------------------------------------------

def evaluate_ranking(
    predicted_order: List[str],
    ground_truth_order: List[str],
    k: int | None = None,
) -> Dict[str, float]:
    """Run all ranking metrics and return a summary dict.

    Args:
        predicted_order:    Candidate names sorted by system score (best first).
        ground_truth_order: Candidate names sorted by ideal relevance (best first).
        k:                  Cutoff for NDCG@k and Precision@k.  Defaults to
                            ``len(ground_truth_order)``.

    Returns:
        Dict with keys: ``ndcg``, ``precision_at_k``, ``spearman``, ``map``.

    Example::

        from src.evaluation.metrics import evaluate_ranking

        results = evaluate_ranking(
            predicted_order=["Alice", "Bob", "Carol", "Dave"],
            ground_truth_order=["Alice", "Carol", "Bob", "Dave"],
        )
        # {'ndcg': 0.95, 'precision_at_k': 1.0, 'spearman': 0.8, 'map': 0.92}
    """
    if k is None:
        k = len(ground_truth_order)

    return {
        "ndcg": ndcg_at_k(predicted_order, ground_truth_order, k=k),
        "precision_at_k": precision_at_k(predicted_order, ground_truth_order, k=max(1, k // 2)),
        "spearman": spearman_correlation(predicted_order, ground_truth_order),
        "map": mean_average_precision(predicted_order, ground_truth_order),
    }
