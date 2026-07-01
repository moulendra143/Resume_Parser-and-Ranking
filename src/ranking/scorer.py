import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# A simple heuristic scorer as a fallback
def compute_heuristic_score(
    similarity: float,
    resume_skills: list,
    jd_skills: list,
    resume_exp: float,
    jd_exp: float,
    resume_edu: list
) -> float:
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
        self.model = None
        self._load_or_train_mock_model()

    def _load_or_train_mock_model(self):
        """
        Loads the ML model if it exists, otherwise trains a mock Random Forest
        model using synthetically generated data to demonstrate the ML pipeline.
        In a real scenario, this would be trained on historical hiring data.
        """
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            # Generate synthetic training data
            # Features: [cosine_similarity, skill_match_ratio, experience_ratio, has_education]
            np.random.seed(42)
            X = np.random.rand(1000, 4)
            
            # Target (Scores 0-100) generated based on a logical combination of features with some noise
            y = (X[:, 0] * 60) + (X[:, 1] * 20) + (np.clip(X[:, 2], 0, 1) * 10) + (X[:, 3] * 10)
            y += np.random.normal(0, 2, 1000) # Add noise
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
        resume_edu: list
    ) -> float:
        # Calculate feature ratios
        skill_ratio = len(set(resume_skills).intersection(set(jd_skills))) / len(jd_skills) if jd_skills else 0.0
        exp_ratio = resume_exp / jd_exp if jd_exp > 0 else 0.0
        has_edu = 1.0 if resume_edu else 0.0
        
        # Prepare feature vector for ML Model
        features = np.array([[similarity, skill_ratio, exp_ratio, has_edu]])
        
        # Predict using the trained Random Forest Regressor
        predicted_score = self.model.predict(features)[0]
        return round(np.clip(predicted_score, 0, 100), 2)

# Global instance
ml_ranker = MLRanker()

def compute_score(*args, **kwargs):
    """Wrapper function to use the ML model for scoring."""
    return ml_ranker.predict_score(*args, **kwargs)
