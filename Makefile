# Resume Screening & Ranking System - Makefile

.PHONY: install run run-batch api streamlit test lint clean

# Install all dependencies
install:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm

# Run the CLI pipeline on a single resume
run:
	python -m src.main --resume data/resumes/sample_resume.txt --jd data/job_descriptions/sample_jd.txt

# Run batch ranking on all resumes in data/resumes/
run-batch:
	python -m src.main --resume-dir data/resumes --jd data/job_descriptions/sample_jd.txt

# Start the FastAPI REST server
api:
	uvicorn src.api:app --reload --port 8000

# Start the Streamlit demo UI
streamlit:
	streamlit run src/app.py

# Run all unit tests
test:
	pytest tests/ -v

# Lint with flake8
lint:
	flake8 src/ tests/ --max-line-length 120

# Remove generated files
clean:
	rm -rf __pycache__ .pytest_cache
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
