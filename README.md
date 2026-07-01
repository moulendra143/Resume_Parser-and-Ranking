# AIML-Template - Resume Screening & Ranking System

A Python project for resume parsing, feature extraction, NLP-based similarity scoring, and machine learning candidate ranking.

## Project Structure
```
AIML-Template/
├── README.md                     # You are here
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variable template
├── .gitignore
├── pyproject.toml                # Build config + linter settings
├── Makefile                      # Common commands (run, test, lint)
│
├── config/
│   ├── config.yaml               # App configuration
│   └── skills_taxonomy.json      # Curated skills dictionary
│
├── data/
│   ├── resumes/                  # Drop sample resumes here
│   ├── job_descriptions/         # Sample JDs
│   └── processed/                # Parsed JSON outputs
│
├── src/
│   ├── __init__.py
│   ├── main.py                   # CLI entry point
│   ├── api.py                    # FastAPI REST server
│   │
│   ├── parser/                   # Resume parsing
│   │   ├── __init__.py
│   │   ├── pdf_parser.py         # pdfplumber
│   │   ├── docx_parser.py        # python-docx
│   │   └── text_cleaner.py       # spaCy text cleaner
│   │
│   ├── features/                 # Feature extraction
│   │   ├── __init__.py
│   │   ├── skills_extractor.py
│   │   ├── experience_extractor.py
│   │   └── education_extractor.py
│   │
│   ├── similarity/               # NLP similarity metrics
│   │   ├── __init__.py
│   │   └── embedding_matcher.py  # Sentence-Transformers (all-MiniLM-L6-v2)
│   │
│   ├── ranking/                  # Candidate ranking
│   │   ├── __init__.py
│   │   └── scorer.py             # Scikit-learn ML scoring model (Random Forest)
│   │
│   └── utils/
│       └── __init__.py
│
└── tests/
    └── __init__.py
```

## Quick Start

### 1. Clone & Set up
```bash
git clone <your-fork-url> && cd AIML-Template
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Run the CLI Pipeline on a single resume
```bash
python -m src.main --resume data/resumes/sample.pdf --jd data/job_descriptions/sample_jd.txt
```

### 3. Start the REST API
```bash
uvicorn src.api:app --reload --port 8000
```
Open `http://localhost:8000/docs` in your browser to test the API via Swagger UI.

---

## Evaluation Criteria: Tradeoffs and Model Choices

This project uses a hybrid approach of classical NLP (rule-based feature extraction) and modern Deep Learning (Sentence Embeddings) combined with a Machine Learning Ranker (Random Forest).

### 1. Parsing & Feature Extraction (Rule-Based vs. LLM)
* **Choice:** We used `spacy` + Regex for parsing (skills, experience, education).
* **Tradeoff:** Using a LLM (like GPT-4) would provide higher accuracy and flexibility out-of-the-box. However, LLMs are expensive, slow, and non-deterministic. A rule-based `spaCy` approach combined with a curated `skills_taxonomy.json` is extremely fast, highly reproducible, and perfectly suited for structured data extraction like degrees and years of experience.

### 2. Similarity Metric (TF-IDF vs. Sentence Transformers)
* **Choice:** We used `all-MiniLM-L6-v2` from `sentence-transformers` for calculating semantic similarity between the resume text and the JD.
* **Tradeoff:** TF-IDF with Cosine Similarity is faster and requires no GPU or heavy models. However, it fails to capture semantic meaning (e.g., "Software Engineer" vs. "Backend Developer"). `all-MiniLM-L6-v2` is a 384-dimensional dense vector model that strikes the perfect balance between speed (very fast even on CPU) and high accuracy in semantic matching.

### 3. Candidate Ranking (Heuristic vs. Machine Learning)
* **Choice:** A Machine Learning ranking pipeline (`RandomForestRegressor` from `scikit-learn`) was implemented inside `src/ranking/scorer.py`.
* **Implementation Details:** Real-world historical hiring data (accepted vs. rejected candidates) is rarely available in take-home assignments. Therefore, the class `MLRanker` is designed to demonstrate an ML pipeline. On first run, it dynamically generates synthetic historical data mapping features (similarity score, skill match ratio, experience ratio, education binary) to a final fit score, trains a `RandomForestRegressor`, and serializes it using `joblib`.
* **Tradeoff:** A heuristic formula (e.g., `0.6*Sim + 0.2*Skills + ...`) is easier to explain and tune initially. However, an ML model is required by the project brief. A Random Forest was chosen over a simple Linear Regression because hiring decisions are often non-linear (e.g., having 10 years of experience when 2 are required shouldn't necessarily give a massive linear boost).

## Out of Scope
As requested, the following are intentionally omitted:
- Frontend/UI
- Integrations with Job Boards
- Email/Interview scheduling systems
