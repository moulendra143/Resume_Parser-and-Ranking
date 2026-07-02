"""
ML-based candidate scoring with lazy model loading.

The RandomForestRegressor model is loaded/trained only on first use.
Hyperparameters (n_estimators, random_seed, synthetic_samples) are read
from ``config/config.yaml`` via :func:`src.utils.config_loader.get_config`
so they can be tuned without touching source code.

Feature vector fed to the ML model
-----------------------------------
+----------------+--------------------------------------------------+
| Feature        | Description                                      |
+================+==================================================+
| similarity     | Hybrid NLP similarity score (0–1)                |
+----------------+--------------------------------------------------+
| skill_ratio    | Fraction of JD skills present in resume (0–1)    |
+----------------+--------------------------------------------------+
| exp_ratio      | resume_exp / jd_exp capped at 2.0               |
+----------------+--------------------------------------------------+
| has_education  | 1.0 if any degree detected, else 0.0             |
+----------------+--------------------------------------------------+
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

from src.utils.config_loader import get_config


def compute_heuristic_score(
    similarity: float,
    resume_skills: list,
    jd_skills: list,
    resume_exp: float,
    jd_exp: float,
    resume_edu: list,
) -> float:
    """A transparent heuristic scorer used as a fallback and for comparison.

    Scoring breakdown (weights mirror config/config.yaml scoring_weights):
        - Similarity  → up to 60 pts
        - Skills      → up to 20 pts
        - Experience  → up to 10 pts
        - Education   → up to 10 pts

    Args:
        similarity:    Hybrid NLP similarity score (0–1).
        resume_skills: Skills extracted from the resume.
        jd_skills:     Skills required by the job description.
        resume_exp:    Years of experience from the resume.
        jd_exp:        Years of experience required by the JD.
        resume_edu:    Education qualifications from the resume.

    Returns:
        Final score clipped to [0, 100].
    """
    cfg = get_config().scoring_weights

    sim_score = max(0.0, similarity) * (cfg.similarity * 100)

    if jd_skills:
        matched_skills = set(resume_skills).intersection(set(jd_skills))
        skill_score = (len(matched_skills) / len(jd_skills)) * (cfg.skills * 100)
    else:
        skill_score = (cfg.skills * 100) if resume_skills else 0.0

    if resume_exp >= jd_exp:
        exp_score = cfg.experience * 100
    else:
        exp_score = (resume_exp / jd_exp) * (cfg.experience * 100) if jd_exp > 0 else 0.0

    edu_score = (cfg.education * 100) if resume_edu else 0.0

    final_score = sim_score + skill_score + exp_score + edu_score
    return round(min(100.0, final_score), 2)


class MLRanker:
    """RandomForest-based candidate scorer.

    The model is loaded from disk on first use.  If no serialised model is
    found, a new model is trained on synthetic data and saved to
    ``config.paths.model_path``.
    """

    def __init__(self, model_path: str | None = None):
        cfg = get_config()
        self.model_path = model_path or cfg.paths.model_path
        self._n_estimators = cfg.ranking.n_estimators
        self._random_seed = cfg.ranking.random_seed
        self._synthetic_samples = cfg.ranking.synthetic_samples
        self.model: RandomForestRegressor | None = None

    def _ensure_model(self) -> None:
        """Load the model from disk or train a new one if not found."""
        if self.model is not None:
            return

        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            self._train_and_save()

    def _train_and_save(self) -> None:
        """Generate synthetic training data, train, and persist the model.

        Synthetic data schema:
            X[:, 0] = similarity   (uniform 0–1)
            X[:, 1] = skill_ratio  (uniform 0–1)
            X[:, 2] = exp_ratio    (uniform 0–1)
            X[:, 3] = has_edu      (uniform 0–1)

        Labels follow the same weighting as the heuristic scorer so the RF
        learns to capture non-linear interactions on top of the base formula.
        Gaussian noise (σ=2) prevents over-fitting to the synthetic signal.
        """
        np.random.seed(self._random_seed)
        n = self._synthetic_samples
        X = np.random.rand(n, 4)
        y = (X[:, 0] * 60) + (X[:, 1] * 20) + (np.clip(X[:, 2], 0, 1) * 10) + (X[:, 3] * 10)
        y += np.random.normal(0, 2, n)
        y = np.clip(y, 0, 100)

        self.model = RandomForestRegressor(
            n_estimators=self._n_estimators,
            random_state=self._random_seed,
        )
        self.model.fit(X, y)
        joblib.dump(self.model, self.model_path)

    def predict_score(
        self,
        similarity: float,
        resume_skills: list,
        jd_skills: list,
        resume_exp: float,
        jd_exp: float,
        resume_edu: list,
    ) -> float:
        """Predict a candidate fit score using the trained Random Forest.

        Args:
            similarity:    Hybrid NLP similarity score (0–1).
            resume_skills: Skills extracted from the resume.
            jd_skills:     Skills required by the job description.
            resume_exp:    Years of experience from the resume.
            jd_exp:        Years of experience required by the JD.
            resume_edu:    Education qualifications from the resume.

        Returns:
            Predicted fit score clipped to [0, 100].
        """
        self._ensure_model()
        skill_ratio = (
            len(set(resume_skills).intersection(set(jd_skills))) / len(jd_skills)
            if jd_skills
            else 0.0
        )
        # Cap exp_ratio at 2.0 — being 10x over-qualified should not linearly
        # inflate the score beyond what is practically useful.
        exp_ratio = min(resume_exp / jd_exp, 2.0) if jd_exp > 0 else 0.0
        has_edu = 1.0 if resume_edu else 0.0
        features = np.array([[similarity, skill_ratio, exp_ratio, has_edu]])
        predicted_score = self.model.predict(features)[0]
        return round(float(np.clip(predicted_score, 0, 100)), 2)


# ---------------------------------------------------------------------------
# Module-level singleton — model loaded lazily on first predict call
# ---------------------------------------------------------------------------
ml_ranker = MLRanker()


def compute_score(*args, **kwargs) -> float:
    """Module-level wrapper that delegates to the global :class:`MLRanker` instance."""
    return ml_ranker.predict_score(*args, **kwargs)
