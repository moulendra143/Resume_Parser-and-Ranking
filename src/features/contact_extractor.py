"""
Contact information extractor.

Extracts email addresses, phone numbers, and LinkedIn profile URLs
from raw resume text using regex patterns.
"""

import re
from typing import Dict, List


def extract_emails(text: str) -> List[str]:
    """Extract email addresses from text."""
    pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    return list(set(re.findall(pattern, text)))


def extract_phones(text: str) -> List[str]:
    """Extract phone numbers from text."""
    pattern = r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
    return list(set(re.findall(pattern, text)))


def extract_linkedin(text: str) -> List[str]:
    """Extract LinkedIn profile URLs from text."""
    pattern = r"(?:https?://)?(?:www\.)?linkedin\.com/in/[a-zA-Z0-9_-]+"
    return list(set(re.findall(pattern, text, re.IGNORECASE)))


def extract_contact_info(text: str) -> Dict[str, List[str]]:
    """Extract all contact information from text.

    Returns:
        Dictionary with ``emails``, ``phones``, and ``linkedin`` keys.
    """
    return {
        "emails": extract_emails(text),
        "phones": extract_phones(text),
        "linkedin": extract_linkedin(text),
    }
