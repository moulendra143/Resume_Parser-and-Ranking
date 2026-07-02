# Resume Screening & Ranking System Using NLP and ML

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
| **Feature Extraction** | Skills (taxonomy-based), experience (regex), education (regex), contact info (email/phone/LinkedIn) |
| **NLP Similarity** | Sentence-Transformer embeddings (`all-MiniLM-L6-v2`) + TF-IDF cosine similarity + hybrid scorer |
| **ML Ranking** | `RandomForestRegressor` trained on synthetic hiring data, serialized with `joblib` |
| **Batch Ranking** | Rank an entire directory of resumes against a single job description |
| **REST API** | FastAPI server with Swagger docs at `/docs` |
| **Demo UI** | Streamlit web app for interactive resume analysis |

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
              │              └───────────────┬───────────────────┘
              │                              │
    ┌─────────▼──────────────────────────────▼───────────────────┐
    │                   ML RANKING ENGINE                        │
    │  RandomForestRegressor (scikit-learn)                      │
    │  Features: [similarity, skill_ratio, exp_ratio, has_edu]  │
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
Resume_Parser-and-Ranking/
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
├── .env.example                      # Environment variable template
├── .gitignore                        # Git ignore rules
├── Makefile                          # Common commands (run, test, api, etc.)
├── model.joblib                      # Serialized ML model (auto-generated)
│
├── config/
│   ├── config.yaml                   # App configuration (paths, weights, etc.)
│   └── skills_taxonomy.json          # Curated skills dictionary (60+ skills)
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
│   ├── api.py                        # FastAPI REST server
│   ├── app.py                        # Streamlit demo UI
│   │
│   ├── parser/                       # Resume text extraction
│   │   ├── __init__.py
│   │   ├── pdf_parser.py             # PDF → text (pdfplumber)
│   │   ├── docx_parser.py            # DOCX → text (python-docx)
│   │   └── text_cleaner.py           # spaCy NLP cleaning pipeline
│   │
│   ├── features/                     # Structured feature extraction
│   │   ├── __init__.py
│   │   ├── skills_extractor.py       # Taxonomy-based skill matching
│   │   ├── experience_extractor.py   # Regex-based experience (years)
│   │   ├── education_extractor.py    # Regex-based degree extraction
│   │   └── contact_extractor.py      # Email, phone, LinkedIn extraction
│   │
│   ├── similarity/                   # NLP similarity metrics
│   │   ├── __init__.py
│   │   ├── embedding_matcher.py      # Sentence-Transformers (all-MiniLM-L6-v2)
│   │   ├── tfidf_matcher.py          # TF-IDF cosine similarity (sklearn)
│   │   └── hybrid_scorer.py          # Weighted hybrid (embedding + TF-IDF)
│   │
│   ├── ranking/                      # Candidate scoring & ranking
│   │   ├── __init__.py
│   │   ├── scorer.py                 # ML model (RandomForestRegressor) + heuristic fallback
│   │   └── ranker.py                 # Batch candidate ranking engine
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py                 # Centralized logging configuration
│       └── io.py                     # File I/O helpers
│
└── tests/                            # Unit tests (pytest)
    ├── __init__.py
    ├── test_parser.py                # Text cleaner tests
    ├── test_features.py              # Skills, experience, education, contact tests
    ├── test_similarity.py            # TF-IDF similarity tests
    └── test_ranking.py               # ML scorer + batch ranker tests
```

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
pytest tests/ -v
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
uvicorn src.api:app --reload --port 8000
```

Then open your browser to:
* **http://localhost:8000** — The premium high-fidelity **AI Resume Screener & Ranker Dashboard** (styled exactly to the target mockup with radar charts, circular progress rings, and sandbox feature sandboxes).
* **http://localhost:8000/docs** — Swagger API interactive docs.

**API Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the premium HTML SPA dashboard |
| `POST` | `/api/v1/rank` | Score a single resume (upload file + JD text) |
| `POST` | `/api/v1/rank-batch` | Upload multiple resumes, get ranked list |
| `POST` | `/api/v1/sandbox/skills` | Sandbox route to extract skills from text |
| `POST` | `/api/v1/sandbox/experience` | Sandbox route to calculate experience years |
| `POST` | `/api/v1/sandbox/education` | Sandbox route to parse education degrees |
| `POST` | `/api/v1/sandbox/contact` | Sandbox route to parse contact details |

**Example cURL (single resume):**

```bash
curl -X POST "http://localhost:8000/api/v1/rank" \
  -F "resume=@data/resumes/sample_resume.txt" \
  -F "job_description=Looking for a Python developer with 3+ years experience in ML and SQL"
```

### 4. Streamlit Demo UI (Alternative)

```bash
streamlit run src/app.py
```

An alternative interactive demo UI running on **http://localhost:8501**.


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
pytest tests/ -v

# Run a specific test file
pytest tests/test_features.py -v

# Run with coverage (install pytest-cov first)
pip install pytest-cov
pytest tests/ --cov=src --cov-report=term-missing
```

**Test coverage includes:**
- `test_parser.py` — Text cleaning, stopword removal, lemmatization
- `test_features.py` — Skills, experience, education, and contact extraction
- `test_similarity.py` — TF-IDF cosine similarity edge cases
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

**Why this choice:** For a screening system that processes many resumes, speed and determinism are critical. A curated `skills_taxonomy.json` provides controlled, extensible skill matching. spaCy's lemmatization handles verb/noun variations well.

### 2. Similarity Metric — Sentence-Transformers vs. TF-IDF

| | Sentence-Transformers (Primary) | TF-IDF (Baseline) |
|---|---|---|
| **Semantic Understanding** | ✅ "Software Engineer" ≈ "Backend Developer" | ❌ Exact keyword match only |
| **Speed** | ⚠️ ~50ms (CPU) | ✅ ~1ms |
| **Model Size** | ⚠️ ~80MB download | ✅ No model needed |
| **OOV Handling** | ✅ Handles unseen words | ❌ Misses unseen vocabulary |

**Why this choice:** We use **both**. `all-MiniLM-L6-v2` (384-dim) is the primary semantic similarity engine — it's the smallest, fastest Sentence-Transformer with excellent accuracy. TF-IDF serves as a complementary lexical signal. The **hybrid scorer** combines them: `0.7 × Embedding + 0.3 × TF-IDF`.

### 3. Candidate Ranking — Random Forest vs. Heuristic Formula

| | RandomForestRegressor (Chosen) | Linear Weighted Sum |
|---|---|---|
| **Non-linearity** | ✅ Captures diminishing returns | ❌ Linear only |
| **Feature Interactions** | ✅ Learns interactions (e.g., skills × experience) | ❌ Independent weights |
| **Interpretability** | ⚠️ Feature importances available | ✅ Fully transparent |
| **Data Requirement** | ⚠️ Needs training data | ✅ No data needed |

**Why this choice:** Hiring decisions are non-linear — having 15 years when 3 are required shouldn't double the score. A Random Forest naturally handles this. Since real historical hiring data is unavailable for this project, we generate **synthetic training data** (1000 samples) that encodes logical scoring rules with noise, train the model, and serialize it with `joblib`. A **heuristic fallback scorer** (`compute_heuristic_score`) is also provided for comparison.

### 4. Synthetic Training Data Justification

The ML model trains on synthetically generated data because:
- Real labeled hiring datasets (accepted/rejected candidates) are proprietary and rare
- The synthetic data maps feature vectors `[similarity, skill_ratio, exp_ratio, has_education]` → score using a logical formula with Gaussian noise
- This demonstrates the full ML pipeline: data → training → serialization → inference
- In production, this would be replaced with real company hiring decision data

---

## Configuration

### `config/config.yaml`

Controls application settings including:
- File paths (taxonomy, model, data directories)
- Similarity weights (embedding vs TF-IDF ratio)
- ML model hyperparameters (n_estimators, random_seed)
- Heuristic scoring weights

### `config/skills_taxonomy.json`

Curated list of 60+ technical skills across:
- Programming languages (Python, Java, C++, Go, Rust, etc.)
- Frameworks (React, Django, Flask, Spring Boot, etc.)
- AI/ML (TensorFlow, PyTorch, Scikit-learn, NLP, Computer Vision)
- Cloud & DevOps (AWS, Azure, GCP, Docker, Kubernetes, Terraform)
- Databases (MySQL, PostgreSQL, MongoDB, Redis, Elasticsearch)

To add custom skills, simply append them to the `skills` array in the JSON file.

---

## Out of Scope

As specified in the project brief, the following are intentionally excluded:
- **Frontend/UI** (except the Streamlit demo for testing convenience)
- **Integration with external job boards or APIs**
- **Advanced features** like interview scheduling or candidate communication
