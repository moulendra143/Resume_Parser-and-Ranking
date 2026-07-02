# Resume Screening & Ranking System Using NLP and ML

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/Tests-pytest-yellow?logo=pytest)
![ML](https://img.shields.io/badge/ML-RandomForest-orange?logo=scikit-learn)
![NLP](https://img.shields.io/badge/NLP-Sentence--Transformers-purple)
![Code Style](https://img.shields.io/badge/Code%20Style-PEP8-lightgrey)

An end-to-end Python system that **parses resumes**, **extracts structured features** (skills, experience, education, contact info), computes **NLP-based semantic similarity** against a job description, and **ranks candidates** using a trained **Machine Learning model** (Random Forest).

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Single Resume Analysis (CLI)](#1-single-resume-analysis-cli)
  - [Batch Candidate Ranking (CLI)](#2-batch-candidate-ranking-cli)
  - [REST API](#3-rest-api)
  - [Streamlit Demo UI](#4-streamlit-demo-ui)
- [Sample Output](#sample-output)
- [Running Tests](#running-tests)
- [Model Choices & Tradeoff Notes](#model-choices--tradeoff-notes)
- [Configuration](#configuration)
- [Out of Scope](#out-of-scope)

---

## Features

| Capability | Description |
|---|---|
| **Resume Parsing** | Extracts raw text from PDF (`pdfplumber`), DOCX (`python-docx`), and TXT files |
| **Text Cleaning** | spaCy-based NLP pipeline — lowercasing, stopword removal, lemmatization |
| **Feature Extraction** | Skills (taxonomy + 60+ aliases), contextual experience (false-positive filtering), education, contact info |
| **NLP Similarity** | Sentence-Transformer embeddings (`all-MiniLM-L6-v2`) + TF-IDF cosine similarity + hybrid scorer |
| **ML Ranking** | `RandomForestRegressor` trained on synthetic hiring data, serialized with `joblib` |
| **Batch Ranking** | Rank an entire directory of resumes against a single job description |
| **Ranking Evaluation** | NDCG@k, Precision@k, Spearman ρ, MAP — quantitative proof of ranking accuracy |
| **REST API** | FastAPI server with Swagger docs at `/docs` |
| **Demo UI** | Streamlit web app for interactive resume analysis |
| **Config-driven** | All weights, model names, and hyperparameters read from `config/config.yaml` — no hardcoding |

---

## Architecture

```
                   ┌─────────────────────────────────────────────┐
                   │             INPUT LAYER                     │
                   │  Resume (PDF/DOCX/TXT) + Job Description    │
                   └──────────────────┬──────────────────────────┘
                                      │
                   ┌──────────────────▼──────────────────────────┐
                   │          PARSING & CLEANING                 │
                   │  pdf_parser → docx_parser → text_cleaner    │
                   │  (pdfplumber)  (python-docx)  (spaCy NLP)   │
                   └──────────────────┬──────────────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
    ┌─────────▼──────────┐ ┌─────────▼──────────┐ ┌─────────▼──────────┐
    │ FEATURE EXTRACTION │ │  NLP SIMILARITY     │ │  TF-IDF SIMILARITY │
    │ • Skills (taxonomy)│ │  Sentence-BERT      │ │  Sklearn TF-IDF    │
    │ • Experience (regex│ │  all-MiniLM-L6-v2   │ │  Cosine Similarity │
    │ • Education (regex)│ │  Cosine similarity  │ │                    │
    │ • Contact (regex)  │ └─────────┬──────────┘ └─────────┬──────────┘
    └─────────┬──────────┘           │                       │
              │              ┌───────▼───────────────────────▼───┐
              │              │      HYBRID SIMILARITY SCORER     │
              │              │  0.7 × Embedding + 0.3 × TF-IDF  │
              │              │  (weights configurable via YAML)  │
              └───────────────┬───────────────────┘
                              │
    ┌─────────▼──────────────────────────────▼───────────────────┐
    │                   ML RANKING ENGINE                        │
    │  RandomForestRegressor (scikit-learn)                      │
    │  Features: [similarity, skill_ratio, exp_ratio, has_edu]  │
    │  Hyperparams read from config/config.yaml                  │
    │  Output: Candidate fit score (0–100)                      │
    └───────────────────────┬────────────────────────────────────┘
                            │
    ┌───────────────────────▼────────────────────────────────────┐
    │                RANKED CANDIDATE LIST                       │
    │  Sorted by ML score — best-fit candidate first             │
    └────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
resume_parser/
├── README.md                         # This file
├── requirements.txt                  # Python dependencies (grouped by purpose)
├── .env.example                      # Environment variable template
├── .gitignore                        # Git ignore rules
├── Makefile                          # Common commands (run, test, api, etc.)
├── model.joblib                      # Serialized ML model (auto-generated on first run)
│
├── config/
│   ├── config.yaml                   # ★ Master config — paths, weights, hyperparams
│   └── skills_taxonomy.json          # 90+ canonical skills + 60+ alias mappings
│
├── data/
│   ├── resumes/                      # Sample resumes (4 included)
│   │   ├── sample_resume.txt
│   │   ├── jane_smith_ds.txt
│   │   ├── alex_johnson_fe.txt
│   │   └── priya_patel_ml.txt
│   └── job_descriptions/
│       └── sample_jd.txt
│
├── src/
│   ├── __init__.py
│   ├── main.py                       # CLI entry point (single + batch modes)
│   ├── api.py                        # FastAPI REST server + SPA dashboard
│   ├── app.py                        # Streamlit demo UI
│   │
│   ├── parser/                       # Resume text extraction
│   │   ├── pdf_parser.py             # PDF → text (pdfplumber)
│   │   ├── docx_parser.py            # DOCX → text (python-docx)
│   │   └── text_cleaner.py           # spaCy NLP cleaning pipeline
│   │
│   ├── features/                     # Structured feature extraction
│   │   ├── skills_extractor.py       # Two-pass: direct match + alias resolution
│   │   ├── experience_extractor.py   # Contextual regex (filters 'years old', 'years ago')
│   │   ├── education_extractor.py    # Regex-based degree extraction
│   │   └── contact_extractor.py      # Email, phone, LinkedIn extraction
│   │
│   ├── similarity/                   # NLP similarity metrics
│   │   ├── embedding_matcher.py      # Sentence-Transformers (model from config)
│   │   ├── tfidf_matcher.py          # TF-IDF cosine similarity (sklearn)
│   │   └── hybrid_scorer.py          # Weighted hybrid (weight from config)
│   │
│   ├── ranking/                      # Candidate scoring & ranking
│   │   ├── scorer.py                 # ML model (RandomForestRegressor) + heuristic fallback
│   │   └── ranker.py                 # Batch candidate ranking engine
│   │
│   └── utils/
│       ├── config_loader.py          # ★ YAML config loader — typed dataclasses, LRU cached
│       ├── logger.py                 # Centralized logging configuration
│       └── io.py                     # File I/O helpers
│
├── notebooks/
│   └── 01_explore_parsing.ipynb      # Exploratory notebook: parsing + feature extraction demo
│
└── tests/                            # Unit tests (pytest)
    ├── test_parser.py                # Text cleaner tests
    ├── test_features.py              # Skills, experience, education, contact tests
    ├── test_similarity.py            # TF-IDF + hybrid scorer + config loader tests
    └── test_ranking.py               # ML scorer accuracy, heuristic fallback, batch ranking
```

> **★ Config-driven design:** All tunable parameters (embedding model name, similarity weights, RF hyperparameters, scoring weights) live exclusively in `config/config.yaml`. The `src/utils/config_loader.py` module loads this file once at startup via an LRU-cached singleton and exposes typed dataclass attributes to every module. There are no magic numbers hardcoded in source files.

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/moulendra143/Resume_Parser-and-Ranking.git
cd Resume_Parser-and-Ranking

# Create virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download the spaCy English model
python -m spacy download en_core_web_sm
```

### 2. Verify Installation

```bash
python -m pytest tests/ -v
```

### 3. (Optional) Run with coverage report

```bash
python -m pytest tests/ --cov=src --cov-report=term-missing
```

---

## Usage

### 1. Single Resume Analysis (CLI)

Score one resume against a job description:

```bash
python -m src.main --resume data/resumes/sample_resume.txt --jd data/job_descriptions/sample_jd.txt
```

### 2. Batch Candidate Ranking (CLI)

Rank all resumes in a directory against a job description:

```bash
python -m src.main --resume-dir data/resumes --jd data/job_descriptions/sample_jd.txt
```

### 3. REST API & Custom Web Dashboard

Start the FastAPI server (which hosts both the REST endpoints and the premium Single Page Application dashboard):

```bash
python -m uvicorn src.api:app --reload --port 8000
```

Then open your browser to:
* **http://localhost:8000** — The premium high-fidelity **AI Resume Screener & Ranker Dashboard** (radar charts, circular progress rings, sandbox feature sandboxes).
* **http://localhost:8000/docs** — Swagger API interactive docs.

**API Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the premium HTML SPA dashboard |
| `POST` | `/api/v1/rank` | Score a single resume (upload file + JD text) |
| `POST` | `/api/v1/rank-batch` | Upload multiple resumes, get ranked list |
| `POST` | `/api/v1/sandbox/skills` | Extract skills from arbitrary text |
| `POST` | `/api/v1/sandbox/experience` | Calculate experience years from text |
| `POST` | `/api/v1/sandbox/education` | Parse education degrees from text |
| `POST` | `/api/v1/sandbox/contact` | Parse contact details from text |

**Example cURL (single resume):**

```bash
curl -X POST "http://localhost:8000/api/v1/rank" \
  -F "resume=@data/resumes/sample_resume.txt" \
  -F "job_description=Looking for a Python developer with 3+ years experience in ML and SQL"
```

**Example cURL (batch ranking):**

```bash
curl -X POST "http://localhost:8000/api/v1/rank-batch" \
  -F "resumes=@data/resumes/priya_patel_ml.txt" \
  -F "resumes=@data/resumes/jane_smith_ds.txt" \
  -F "resumes=@data/resumes/alex_johnson_fe.txt" \
  -F "job_description=Senior ML Engineer with Python, TensorFlow, and AWS"
```

### 4. Streamlit Demo UI (Alternative)

```bash
python -m streamlit run src/app.py
```

An alternative interactive demo UI running on **http://localhost:8501**.

### 5. Ranking Evaluation

Measure quantitative ranking accuracy against a ground-truth dataset:

```bash
# Run the full evaluation and print NDCG, Spearman, Precision@k, MAP
python scripts/evaluate_ranking.py

# Change the k cutoff (default = 5)
python scripts/evaluate_ranking.py --k 3

# Save results to JSON for CI integration
python scripts/evaluate_ranking.py --output results/eval_report.json
```

**Sample evaluation output:**

```
================================================================================
  RANKING EVALUATION METRICS  (k=5)
────────────────────────────────────────────────────
  NDCG@5              0.9612  ★★★★★
  Precision@2         1.0000
  Spearman ρ          0.9524  (strong)
  MAP                 0.9583
────────────────────────────────────────────────────
  Overall accuracy:   0.9680 / 1.0000
────────────────────────────────────────────────────
```

---

## Sample Output

### Single Resume Analysis

```
==================================================
  SINGLE RESUME ANALYSIS
==================================================
  File:               sample_resume.txt
  Embedding Sim:      0.6842
  TF-IDF Sim:         0.4231
  Skills Matched:     Python, Java, SQL, AWS, Machine Learning, Git, Linux
  Experience:         4.0 years
  Education:          Bachelor Of Science In Computer Science
--------------------------------------------------
  ML CANDIDATE SCORE: 78.35 / 100
==================================================
```

### Batch Candidate Ranking

```
======================================================================
  CANDIDATE RANKING RESULTS
======================================================================
  Rank   Candidate                      Score      Skills   Exp
----------------------------------------------------------------------
  #1     priya_patel_ml.txt             92.14      15       5.0
  #2     jane_smith_ds.txt              85.67      12       6.0
  #3     sample_resume.txt              78.35      7        4.0
  #4     alex_johnson_fe.txt            41.22      5        3.0
======================================================================

  Top Candidate: #1 priya_patel_ml.txt (Score: 92.14/100)
```

---

## Running Tests

```bash
# Run all tests with verbose output
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_features.py -v

# Run with coverage report (pytest-cov included in requirements.txt)
python -m pytest tests/ --cov=src --cov-report=term-missing

# Run only fast tests (skip embedding tests that require model download)
python -m pytest tests/ -v -k "not embedding"
```

**Test coverage includes:**
- `test_parser.py` — Text cleaning, stopword removal, lemmatization
- `test_features.py` — Skills, experience, education, and contact extraction
- `test_similarity.py` — TF-IDF, hybrid scorer weights, config loader
- `test_ranking.py` — ML scorer accuracy, heuristic fallback, batch ranking

---

## Model Choices & Tradeoff Notes

### 1. Resume Parsing — spaCy + Regex vs. LLM

| | spaCy + Regex (Chosen) | LLM (e.g., GPT-4) |
|---|---|---|
| **Speed** | ✅ ~10ms per resume | ❌ ~2–5s per resume |
| **Cost** | ✅ Free, local | ❌ API costs per call |
| **Reproducibility** | ✅ Deterministic | ❌ Non-deterministic |
| **Accuracy** | ⚠️ Good for structured resumes | ✅ Handles edge cases better |
| **Offline** | ✅ Fully offline | ❌ Requires internet |

**Why this choice:** For a screening system that processes many resumes, speed and determinism are critical. A curated `skills_taxonomy.json` provides controlled, extensible skill matching. spaCy's lemmatization handles verb/noun variations well (e.g. "developing" → "develop").

---

### 2. spaCy Model Size — `en_core_web_sm` vs. `en_core_web_lg` vs. `en_core_web_trf`

| | `en_core_web_sm` (Chosen) | `en_core_web_lg` | `en_core_web_trf` (RoBERTa) |
|---|---|---|---|
| **Download size** | ✅ ~12 MB | ⚠️ ~750 MB | ❌ ~438 MB + transformers |
| **Lemmatization accuracy** | ✅ Sufficient for resumes | ✅ Same | ✅ Same |
| **NER accuracy** | ⚠️ Good (F1 ~85%) | ✅ Better (F1 ~90%) | ✅ Best (F1 ~95%) |
| **Speed** | ✅ ~0.5ms/token | ⚠️ ~2ms/token | ❌ ~20ms/token (GPU needed) |
| **RAM usage** | ✅ ~50 MB | ⚠️ ~750 MB | ❌ ~2 GB |

**Why `en_core_web_sm`:** This project uses spaCy **only** for lemmatization and stopword removal — not for NER entity detection (skills/experience/education are handled by regex and taxonomy matching). For this specific use case, `sm` delivers identical results to `trf` at 40× less memory. Upgrading to `lg` or `trf` would only be justified if we switched from regex-based feature extraction to spaCy NER pipelines in a future version.

---

### 3. Similarity Metric — Sentence-Transformers vs. TF-IDF

| | Sentence-Transformers (Primary) | TF-IDF (Baseline) |
|---|---|---|
| **Semantic Understanding** | ✅ "Software Engineer" ≈ "Backend Developer" | ❌ Exact keyword match only |
| **Speed** | ⚠️ ~50ms (CPU) | ✅ ~1ms |
| **Model Size** | ⚠️ ~80MB download | ✅ No model needed |
| **OOV Handling** | ✅ Handles unseen words | ❌ Misses unseen vocabulary |

**Why both:** We use **both** in a weighted hybrid. `all-MiniLM-L6-v2` (384-dim) is the smallest Sentence-Transformer with strong semantic performance — it captures synonyms, paraphrases, and conceptual similarity that TF-IDF completely misses. TF-IDF is kept as a secondary signal because it provides strong keyword anchoring: if a JD says "Kubernetes" and the resume says "Kubernetes", TF-IDF gives a strong match that embedding similarity might dilute across the full-text semantic space.

**Why `all-MiniLM-L6-v2` specifically:**

| | `all-MiniLM-L6-v2` (Chosen) | `all-mpnet-base-v2` | `paraphrase-multilingual` |
|---|---|---|---|
| **Embedding dims** | 384 | 768 | 768 |
| **Model size** | ✅ ~80 MB | ⚠️ ~420 MB | ❌ ~970 MB |
| **Inference speed** | ✅ ~50ms/CPU | ⚠️ ~150ms/CPU | ❌ ~200ms/CPU |
| **SBERT benchmark** | ⚠️ 58.8 avg | ✅ 63.3 avg | ⚠️ 53.1 avg |
| **Resume use case** | ✅ Excellent | ✅ Slightly better | ❌ Over-engineered |

`all-mpnet-base-v2` has ~7% higher SBERT benchmark accuracy but is 5× larger and 3× slower — a poor tradeoff for a system that may process hundreds of resumes in batch. `all-MiniLM-L6-v2` is the SBERT team's own recommendation for production deployments where speed matters.

---

### 4. Hybrid Similarity Weight — Why 0.7 Embedding / 0.3 TF-IDF?

The hybrid weight `w_emb = 0.7` was selected based on the following reasoning:

| Weight Scheme | Embedding | TF-IDF | Characteristic |
|---|---|---|---|
| Pure embedding (1.0 / 0.0) | 1.0 | 0.0 | May miss exact keyword requirements |
| **Chosen (0.7 / 0.3)** | **0.7** | **0.3** | **Balances semantic + lexical signal** |
| Equal split (0.5 / 0.5) | 0.5 | 0.5 | TF-IDF over-weighted; penalises paraphrasing |
| Pure TF-IDF (0.0 / 1.0) | 0.0 | 1.0 | Fails on synonyms entirely |

**Rationale:** Job descriptions often contain mandatory keyword requirements (e.g. "must have Kubernetes") that a semantically close but lexically different resume would undervalue with pure embedding. However, semantic similarity should dominate because candidates rarely use the same exact wording as the JD. The 70/30 split was chosen to:
- Penalise semantically similar but lexically divergent resumes less than pure TF-IDF would
- Preserve keyword anchoring for critical skills that appear verbatim in the taxonomy
- Align with findings from industry resume screening literature, where embedding-heavy hybrids (0.6–0.8 embedding) consistently outperform lexical-only baselines

The weight is configurable in `config/config.yaml` (`similarity.embedding_weight`) and can be A/B tested without code changes.

---

### 5. Candidate Ranking — Random Forest vs. Heuristic Formula

| | RandomForestRegressor (Chosen) | Linear Weighted Sum |
|---|---|---|
| **Non-linearity** | ✅ Captures diminishing returns | ❌ Linear only |
| **Feature Interactions** | ✅ Learns interactions (e.g., skills × experience) | ❌ Independent weights |
| **Interpretability** | ⚠️ Feature importances available | ✅ Fully transparent |
| **Data Requirement** | ⚠️ Needs training data | ✅ No data needed |

**Why this choice:** Hiring decisions are non-linear — having 15 years when 3 are required shouldn't linearly double the score. A Random Forest naturally handles this through tree-based splits that create decision boundaries. Experience ratio is capped at 2.0 in the feature vector to prevent over-qualified candidates from inflating their score.

The heuristic fallback (`compute_heuristic_score`) is kept in the codebase as a **transparent comparison baseline** — running both and comparing outputs allows practitioners to validate that the ML model is adding value beyond a simple weighted formula.

---

### 6. Synthetic Training Data Justification

The ML model trains on synthetically generated data because:
- Real labeled hiring datasets (accepted/rejected candidates) are proprietary and extremely rare
- The synthetic data maps feature vectors `[similarity, skill_ratio, exp_ratio, has_education]` → score using a logical formula with Gaussian noise (σ=2) that introduces realistic variance
- This demonstrates the complete ML pipeline: data generation → training → serialization → inference
- Hyperparameters (`n_estimators`, `random_seed`, `synthetic_samples`) are all configurable in `config/config.yaml`

**Production upgrade path:** Replace synthetic data with real company hiring outcome data (hired/not-hired labels) and re-run `MLRanker._train_and_save()`. The rest of the pipeline requires no changes.

---

### 7. Skills Extraction — Taxonomy-based vs. NER-based

| | Taxonomy Matching (Chosen) | spaCy/NER-based |
|---|---|---|
| **Precision** | ✅ Very high (no false positives) | ⚠️ Can mis-classify words as skills |
| **Recall** | ⚠️ Limited to 60+ curated skills | ✅ Open-ended — discovers unseen skills |
| **Extensibility** | ✅ Edit a JSON file | ❌ Requires NER model retraining |
| **Interpretability** | ✅ Exact matches — fully explainable | ⚠️ Model-dependent |
| **Synonyms** | ❌ "JS" ≠ "JavaScript" | ✅ Context-aware |

**Why taxonomy:** For a screening system, **false positives in skill extraction are more harmful than missed skills**. Incorrectly claiming a candidate has "Kubernetes" when they mentioned it tangentially in a project description damages recruiter trust. The taxonomy approach guarantees every extracted skill is a deliberate, verified match. Aliases and synonyms can be added to `skills_taxonomy.json` incrementally without code changes.

---

## Configuration

### `config/config.yaml`

All application parameters are centralised here. The `src/utils/config_loader.py` module loads this file once at startup and provides typed access across the entire codebase.

```yaml
similarity:
  embedding_model: "all-MiniLM-L6-v2"   # Swap model name here — no code changes needed
  embedding_weight: 0.7                  # Hybrid weight: 0.7×embedding + 0.3×TF-IDF

ranking:
  n_estimators: 100     # RF trees — increase for better accuracy at cost of speed
  random_seed: 42       # Reproducibility
  synthetic_samples: 1000

scoring_weights:
  similarity: 0.60      # Must sum to 1.0
  skills: 0.20
  experience: 0.10
  education: 0.10
```

### `config/skills_taxonomy.json`

Curated list of **90+ canonical skills** across:
- Programming languages (Python, Java, C++, Go, Rust, TypeScript, etc.)
- Frameworks (React, Django, Flask, FastAPI, Spring Boot, etc.)
- AI/ML (TensorFlow, PyTorch, Scikit-learn, XGBoost, LightGBM, Hugging Face, MLflow, etc.)
- Cloud & DevOps (AWS, Azure, GCP, Docker, Kubernetes, Terraform, Prometheus, Grafana, etc.)
- Databases (MySQL, PostgreSQL, MongoDB, Redis, Elasticsearch, Snowflake, BigQuery, etc.)
- Testing (PyTest, JUnit, Selenium, Cypress, Jest)

**Alias resolution** — 60+ mappings cover common abbreviations and natural language variants:
`ML` → Machine Learning, `JS` → JavaScript, `k8s` → Kubernetes, `sklearn` → Scikit-learn,
`REST API` → REST, `neural networks` → Deep Learning, `containerization` → Docker, etc.

To add custom skills or aliases, edit `config/skills_taxonomy.json` — no code changes required.

---

## Out of Scope

As specified in the project brief, the following are intentionally excluded:
- **Standalone frontend** — no separate UI framework; the Streamlit demo and FastAPI SPA serve as lightweight testing interfaces only
- **Integration with external job boards or APIs**
- **Advanced features** like interview scheduling or candidate communication

