"""Quick smoke test for all core modules - runs without loading the heavy Sentence-Transformer model."""
import sys, json, os
# Insert the project root (parent of scripts/) so 'src' is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

results = []

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    msg = f"  [{status}] {label}"
    if detail:
        msg += f"  ->  {detail}"
    print(msg)
    results.append(condition)

print("\n" + "="*60)
print("  SMOKE TEST — Core modules (no embedding model needed)")
print("="*60)

# 1. Config loader
try:
    from src.utils.config_loader import get_config
    cfg = get_config()
    check("Config loader", True, f"embedding_weight={cfg.similarity.embedding_weight}, n_estimators={cfg.ranking.n_estimators}")
except Exception as e:
    check("Config loader", False, str(e))

# 2. Text cleaner
try:
    from src.parser.text_cleaner import clean_text
    result = clean_text("Python developer with 5 years of experience in Machine Learning")
    check("Text cleaner", len(result) > 0, f"tokens={len(result.split())}")
except Exception as e:
    check("Text cleaner", False, str(e))

# 3. Skills extractor with aliases
try:
    from src.features.skills_extractor import extract_skills
    with open("config/skills_taxonomy.json") as f:
        tax = json.load(f)
    skills = extract_skills("Python developer with ML and k8s experience", tax["skills"], tax.get("aliases", {}))
    check("Skills extractor (aliases)", "Python" in skills and "Machine Learning" in skills and "Kubernetes" in skills,
          f"found: {skills}")
except Exception as e:
    check("Skills extractor", False, str(e))

# 4. Experience extractor (contextual + false-positive filter)
try:
    from src.features.experience_extractor import extract_experience
    exp_correct = extract_experience("I am 28 years old with 5 years of experience in Python")
    exp_fp1     = extract_experience("Founded 10 years ago by engineers")
    exp_fp2     = extract_experience("He is 30 years old and a fresh graduate")
    exp_tech    = extract_experience("4 years in machine learning and cloud computing")
    check("Experience (correct match)",   exp_correct == 5.0, f"got {exp_correct}, expected 5.0")
    check("Experience (years ago FP)",    exp_fp1 == 0.0,     f"got {exp_fp1}, expected 0.0")
    check("Experience (years old FP)",    exp_fp2 == 0.0,     f"got {exp_fp2}, expected 0.0")
    check("Experience (tech domain)",     exp_tech == 4.0,    f"got {exp_tech}, expected 4.0")
except Exception as e:
    check("Experience extractor", False, str(e))

# 5. Education extractor
try:
    from src.features.education_extractor import extract_education
    edu = extract_education("Bachelor of Science in Computer Science from IIT")
    check("Education extractor", len(edu) > 0, f"found: {edu}")
except Exception as e:
    check("Education extractor", False, str(e))

# 6. Contact extractor
try:
    from src.features.contact_extractor import extract_contact_info
    contact = extract_contact_info("Email: john@example.com | Phone: 123-456-7890")
    check("Contact extractor", len(contact["emails"]) > 0 and len(contact["phones"]) > 0,
          f"email={contact['emails']}, phone={contact['phones']}")
except Exception as e:
    check("Contact extractor", False, str(e))

# 7. TF-IDF similarity
try:
    from src.similarity.tfidf_matcher import compute_tfidf_similarity
    score = compute_tfidf_similarity("python machine learning", "python data science machine learning")
    check("TF-IDF similarity", 0.3 < score < 1.0, f"score={score:.4f}")
except Exception as e:
    check("TF-IDF similarity", False, str(e))

# 8. Heuristic scorer
try:
    from src.ranking.scorer import compute_heuristic_score
    score = compute_heuristic_score(1.0, ["Python","SQL"], ["Python","SQL"], 5.0, 3.0, ["BSc"])
    check("Heuristic scorer", score == 100.0, f"perfect candidate score={score}")
except Exception as e:
    check("Heuristic scorer", False, str(e))

# 9. ML scorer (uses saved model.joblib)
try:
    from src.ranking.scorer import compute_score
    ml_score = compute_score(0.8, ["Python","SQL"], ["Python","SQL","AWS"], 4.0, 3.0, ["BSc"])
    check("ML scorer (RandomForest)", 0 <= ml_score <= 100, f"score={ml_score}")
except Exception as e:
    check("ML scorer", False, str(e))

# 10. Batch ranker
try:
    from src.ranking.ranker import rank_candidates
    candidates = [
        {"name": "Strong", "similarity": 0.95, "skills": ["Python","SQL","AWS"], "experience": 5.0, "education": ["BSc"]},
        {"name": "Weak",   "similarity": 0.20, "skills": [],                     "experience": 0.0, "education": []},
    ]
    ranked = rank_candidates(candidates, jd_skills=["Python","SQL","AWS"], jd_exp=3.0)
    check("Batch ranker", ranked[0].name == "Strong" and ranked[1].name == "Weak",
          f"rank1={ranked[0].name}({ranked[0].score:.1f}), rank2={ranked[1].name}({ranked[1].score:.1f})")
except Exception as e:
    check("Batch ranker", False, str(e))

# 11. Evaluation metrics
try:
    from src.evaluation.metrics import evaluate_ranking, ndcg_at_k, spearman_correlation
    order = ["A","B","C","D"]
    metrics = evaluate_ranking(order, order)
    check("Evaluation metrics (perfect)", metrics["ndcg"] == 1.0 and metrics["spearman"] == 1.0,
          f"NDCG={metrics['ndcg']}, Spearman={metrics['spearman']}, MAP={metrics['map']}")
    rev = list(reversed(order))
    bad = evaluate_ranking(rev, order)
    check("Evaluation metrics (reversed)", bad["ndcg"] < 0.9, f"NDCG={bad['ndcg']} (lower than perfect)")
except Exception as e:
    check("Evaluation metrics", False, str(e))

# Summary
print("\n" + "="*60)
passed = sum(results)
total  = len(results)
print(f"  Results: {passed}/{total} checks passed")
if passed == total:
    print("  ALL SYSTEMS: GO [OK]")
else:
    print(f"  {total-passed} checks FAILED -- review output above")
print("="*60 + "\n")
sys.exit(0 if passed == total else 1)
