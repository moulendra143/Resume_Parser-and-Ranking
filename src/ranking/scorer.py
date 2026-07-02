"""
ML-based candidate scoring with lazy model loading.

The RandomForestRegressor model is loaded/trained only on first use.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os


def compute_heuristic_score(
    similarity: float,
    resume_skills: list,
    jd_skills: list,
    resume_exp: float,
    jd_exp: float,
    resume_edu: list,
) -> float:
    """A simple heuristic scorer as a fallback."""
    sim_score = max(0.0, similarity) * 60.0

    if jd_skills:
        matched_skills = set(resume_skills).intersection(set(jd_skills))
        skill_score = (len(matched_skills) / len(jd_skills)) * 20.0
    else:
        skill_score = 20.0 if resume_skills else 0.0

    if resume_exp >= jd_exp:
        exp_score = 10.0
    else:
        exp_score = (resume_exp / jd_exp) * 10.0 if jd_exp > 0 else 0.0

    edu_score = 10.0 if resume_edu else 0.0

    final_score = sim_score + skill_score + exp_score + edu_score
    return round(min(100.0, final_score), 2)


class MLRanker:
    def __init__(self, model_path="model.joblib"):
        self.model_path = model_path
        self.model = None          # lazy — not loaded until predict

    def _ensure_model(self):
        """Load or train the model only when needed."""
        if self.model is not None:
            return
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            np.random.seed(42)
            X = np.random.rand(1000, 4)
            y = (X[:, 0] * 60) + (X[:, 1] * 20) + (np.clip(X[:, 2], 0, 1) * 10) + (X[:, 3] * 10)
            y += np.random.normal(0, 2, 1000)
            y = np.clip(y, 0, 100)
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
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
        self._ensure_model()
        skill_ratio = len(set(resume_skills).intersection(set(jd_skills))) / len(jd_skills) if jd_skills else 0.0
        exp_ratio = resume_exp / jd_exp if jd_exp > 0 else 0.0
        has_edu = 1.0 if resume_edu else 0.0
        features = np.array([[similarity, skill_ratio, exp_ratio, has_edu]])
        predicted_score = self.model.predict(features)[0]
        return round(np.clip(predicted_score, 0, 100), 2)


# Global instance — model loaded lazily on first predict
ml_ranker = MLRanker()


def compute_score(*args, **kwargs):
    """Wrapper function to use the ML model for scoring."""
    return ml_ranker.predict_score(*args, **kwargs)
