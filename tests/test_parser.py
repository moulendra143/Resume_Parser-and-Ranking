"""Tests for the resume parsing module."""

import pytest
from src.parser.text_cleaner import clean_text


class TestTextCleaner:
    """Tests for spaCy-based text cleaning."""

    def test_clean_text_removes_stopwords(self):
        text = "I am a software engineer with experience in Python"
        result = clean_text(text)
        assert "i" not in result.split()
        assert "am" not in result.split()
        assert "a" not in result.split()

    def test_clean_text_lowercases(self):
        text = "PYTHON JAVA SQL"
        result = clean_text(text)
        assert result == result.lower()

    def test_clean_text_removes_extra_whitespace(self):
        text = "Python    Java     SQL"
        result = clean_text(text)
        assert "    " not in result

    def test_clean_text_empty_input(self):
        result = clean_text("")
        assert result == ""

    def test_clean_text_lemmatizes(self):
        text = "developing applications and running tests"
        result = clean_text(text)
        # "developing" should be lemmatized to "develop"
        assert "develop" in result
