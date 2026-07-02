"""Tests for the feature extraction modules."""

import pytest
from src.features.skills_extractor import extract_skills
from src.features.experience_extractor import extract_experience
from src.features.education_extractor import extract_education
from src.features.contact_extractor import extract_contact_info


SAMPLE_ALIASES = {
    "ml": "Machine Learning",
    "dl": "Deep Learning",
    "js": "JavaScript",
    "k8s": "Kubernetes",
    "sklearn": "Scikit-learn",
    "reactjs": "React",
    "nodejs": "Node.js",
}


SAMPLE_TAXONOMY = [
    "Python", "Java", "SQL", "AWS", "Machine Learning",
    "Docker", "React", "Git", "Linux", "C++", "JavaScript",
]


class TestSkillsExtractor:
    """Tests for taxonomy-based skill extraction."""

    def test_extracts_matching_skills(self):
        text = "Experienced in Python, Java, and SQL databases"
        result = extract_skills(text, SAMPLE_TAXONOMY)
        assert "Python" in result
        assert "Java" in result
        assert "SQL" in result

    def test_case_insensitive(self):
        text = "worked with PYTHON and machine learning models"
        result = extract_skills(text, SAMPLE_TAXONOMY)
        assert "Python" in result
        assert "Machine Learning" in result

    def test_no_match_returns_empty(self):
        text = "No relevant technical skills mentioned here"
        result = extract_skills(text, SAMPLE_TAXONOMY)
        assert result == []

    def test_special_chars_in_skills(self):
        text = "Proficient in C++ programming language"
        result = extract_skills(text, SAMPLE_TAXONOMY)
        assert "C++" in result


class TestExperienceExtractor:
    """Tests for regex-based experience extraction."""

    def test_extracts_years(self):
        text = "I have 5 years of experience in software development"
        result = extract_experience(text)
        assert result == 5.0

    def test_extracts_decimal_years(self):
        text = "2.5 years of work experience"
        result = extract_experience(text)
        assert result == 2.5

    def test_returns_max_when_multiple(self):
        text = "3 years at Company A and 5 years at Company B"
        result = extract_experience(text)
        assert result == 5.0

    def test_handles_plus_notation(self):
        text = "Must have 3+ years of experience"
        result = extract_experience(text)
        assert result == 3.0

    def test_no_experience_returns_zero(self):
        text = "Fresh graduate looking for opportunities"
        result = extract_experience(text)
        assert result == 0.0

    def test_ignores_age_false_positive(self):
        """'10 years old' must not be detected as professional experience."""
        text = "I am 28 years old and a recent graduate"
        result = extract_experience(text)
        assert result == 0.0

    def test_ignores_years_ago_false_positive(self):
        """'10 years ago' must not be detected as professional experience."""
        text = "The company was founded 10 years ago and has grown rapidly"
        result = extract_experience(text)
        assert result == 0.0

    def test_mixed_text_picks_correct_value(self):
        """With both a false-positive and a real experience mention, return only the real one."""
        text = "I am 25 years old with 3 years of experience in Python"
        result = extract_experience(text)
        assert result == 3.0

    def test_contextual_at_company(self):
        """'N years at <Company>' should be detected as experience."""
        text = "Worked 4 years at Google and 2 years at Amazon"
        result = extract_experience(text)
        assert result == 4.0

    def test_tech_domain_context(self):
        """'N years in software/data/ML' should match."""
        text = "5 years in machine learning and data science"
        result = extract_experience(text)
        assert result == 5.0


class TestEducationExtractor:
    """Tests for regex-based education extraction."""

    def test_extracts_bachelors(self):
        text = "Bachelor of Science in Computer Science"
        result = extract_education(text)
        assert len(result) >= 1
        assert any("bachelor" in r.lower() for r in result)

    def test_extracts_masters(self):
        text = "Master of Technology in Artificial Intelligence"
        result = extract_education(text)
        assert len(result) >= 1
        assert any("master" in r.lower() for r in result)

    def test_extracts_btech(self):
        text = "B.Tech in Computer Science from IIT"
        result = extract_education(text)
        assert len(result) >= 1

    def test_no_education_returns_empty(self):
        text = "I like hiking and cooking"
        result = extract_education(text)
        assert result == []


class TestSkillsExtractorAliases:
    """Tests for alias-based skill resolution in the skills extractor."""

    def test_ml_alias_resolves_to_machine_learning(self):
        text = "Experienced in ML and data science"
        result = extract_skills(text, SAMPLE_TAXONOMY, SAMPLE_ALIASES)
        assert "Machine Learning" in result

    def test_js_alias_resolves_to_javascript(self):
        text = "Proficient in JS and frontend development"
        result = extract_skills(text, SAMPLE_TAXONOMY, SAMPLE_ALIASES)
        assert "JavaScript" in result

    def test_alias_does_not_duplicate_canonical_match(self):
        """If 'Machine Learning' is already matched directly, ML alias must not add a duplicate."""
        text = "Machine Learning and ML engineering"
        result = extract_skills(text, SAMPLE_TAXONOMY, SAMPLE_ALIASES)
        count = result.count("Machine Learning")
        assert count == 1

    def test_alias_and_direct_match_combined(self):
        """Both canonical and alias-resolved skills should appear together."""
        text = "Python developer with ML and Docker experience"
        result = extract_skills(text, SAMPLE_TAXONOMY, SAMPLE_ALIASES)
        assert "Python" in result          # direct match
        assert "Machine Learning" in result  # alias 'ml'
        assert "Docker" in result           # direct match


class TestContactExtractor:
    """Tests for contact information extraction."""

    def test_extracts_email(self):
        text = "Contact me at john.doe@example.com for details"
        result = extract_contact_info(text)
        assert "john.doe@example.com" in result["emails"]

    def test_extracts_phone(self):
        text = "Phone: 123-456-7890"
        result = extract_contact_info(text)
        assert len(result["phones"]) >= 1

    def test_extracts_linkedin(self):
        text = "LinkedIn: https://www.linkedin.com/in/johndoe"
        result = extract_contact_info(text)
        assert len(result["linkedin"]) >= 1

    def test_no_contact_returns_empty_lists(self):
        text = "No contact information here"
        result = extract_contact_info(text)
        assert result["emails"] == []
        assert result["phones"] == []
        assert result["linkedin"] == []
