"""Tests for the feature extraction modules."""

import pytest
from src.features.skills_extractor import extract_skills
from src.features.experience_extractor import extract_experience
from src.features.education_extractor import extract_education
from src.features.contact_extractor import extract_contact_info


SAMPLE_TAXONOMY = [
    "Python", "Java", "SQL", "AWS", "Machine Learning",
    "Docker", "React", "Git", "Linux", "C++",
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
