"""
Contextual experience extractor.

Extracts total years of professional experience from resume text using a
two-stage approach:

1. **Contextual patterns** (high precision): regex patterns that anchor the
   year count to professional keywords such as "experience", "work", "employed",
   "at <Company>", "as a <Role>", etc.  These match common resume phrasing
   while rejecting false positives like "10 years old" or "founded 10 years ago".

2. **Filtered general fallback**: if no contextual match is found, a broad
   year-number pattern is used, but hits are filtered against an exclusion
   list of known false-positive phrase shapes.

The final result is the maximum value found among all valid matches,
which correlates to total (not per-role) experience on a resume.
"""

import re
from typing import List


# ---------------------------------------------------------------------------
# Contextual patterns — HIGH PRECISION
# These only match years explicitly linked to professional experience.
# ---------------------------------------------------------------------------
_CONTEXTUAL_PATTERNS: List[str] = [
    # "5 years of experience", "3+ years of professional experience"
    r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\s+of\s+(?:professional\s+|industry\s+|hands[- ]on\s+|relevant\s+)?experience",
    # "2.5 years of work", "4 years of employment"
    r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\s+of\s+(?:work|employment|working)",
    # "worked for 6 years", "working for 3+ years"
    r"(?:worked?|working)\s+(?:for\s+)?(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)",
    # "experience: 5 years", "experience of 3 years"
    r"experience\s*(?:of\s+|:\s*)?(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)",
    # "5+ years experience" (no "of")
    r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\s+(?:experience|exp\.?)\b",
    # "3 years at Google / Company"
    r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\s+at\s+[A-Z]",
    # "joined ... 3 years ago" — present tense tenure indicator
    r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\s+(?:as\s+a|as\s+an|in\s+the\s+role)",
    # "3 years in software / data / cloud / ML engineering"
    r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\s+in\s+"
    r"(?:software|tech(?:nology)?|it\b|data|web|cloud|machine\s+learning|"
    r"deep\s+learning|ai\b|ml\b|backend|frontend|full[- ]stack|devops)",
    # "total experience: 5 years", "total exp: 5 yrs"
    r"total\s+(?:experience|exp\.?)\s*[:\-]?\s*(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)",
]

# ---------------------------------------------------------------------------
# Exclusion patterns — false-positive phrases to IGNORE in the fallback
# ---------------------------------------------------------------------------
_EXCLUSION_PATTERNS: List[re.Pattern] = [
    re.compile(r"\d+(?:\.\d+)?\s*(?:years?|yrs?)\s*old", re.IGNORECASE),
    re.compile(r"\d+(?:\.\d+)?\s*(?:years?|yrs?)\s*(?:of\s+)?age", re.IGNORECASE),
    re.compile(r"\d+(?:\.\d+)?\s*(?:years?|yrs?)\s*ago\b", re.IGNORECASE),
    re.compile(r"founded\s+\d+\s*(?:years?|yrs?)", re.IGNORECASE),
    re.compile(r"established\s+\d+\s*(?:years?|yrs?)", re.IGNORECASE),
    re.compile(r"over\s+\d+\s*(?:years?|yrs?)\s*(?:of\s+)?(?:history|since)", re.IGNORECASE),
    re.compile(r"\d+(?:\.\d+)?\s*(?:years?|yrs?)\s*(?:warranty|guarantee)", re.IGNORECASE),
]

# Generic pattern for the fallback sweep
_GENERAL_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)", re.IGNORECASE
)


def _is_excluded(text: str, start: int, end: int, window: int = 60) -> bool:
    """Return True if the match at [start, end] falls inside an exclusion phrase."""
    ctx_start = max(0, start - window)
    ctx_end = min(len(text), end + window)
    context = text[ctx_start:ctx_end]
    return any(pat.search(context) for pat in _EXCLUSION_PATTERNS)


def extract_experience(text: str) -> float:
    """Extract the total years of professional experience from resume text.

    Uses a two-stage contextual approach:
    1. High-precision contextual patterns anchored to professional keywords.
    2. A general year-number sweep with exclusion filtering as fallback.

    Args:
        text: Raw resume text (not cleaned — preserves original phrasing for
              accurate context window matching).

    Returns:
        Total years of experience as a float.  Returns 0.0 if no professional
        experience years can be detected.

    Examples:
        >>> extract_experience("5 years of experience in Python and ML")
        5.0
        >>> extract_experience("I am 28 years old with 3 years of work")
        3.0  # "28 years old" is excluded; "3 years of work" matches
        >>> extract_experience("Founded 10 years ago")
        0.0  # No professional experience match
    """
    if not text:
        return 0.0

    found_years: List[float] = []

    # ---- Stage 1: contextual high-precision patterns ----
    for pattern in _CONTEXTUAL_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            # Extract the numeric group — some patterns have one capture group
            val = m.group(1)
            if val:
                found_years.append(float(val))

    if found_years:
        return max(found_years)

    # ---- Stage 2: general fallback with exclusion filter ----
    for m in _GENERAL_PATTERN.finditer(text):
        if not _is_excluded(text, m.start(), m.end()):
            found_years.append(float(m.group(1)))

    return max(found_years) if found_years else 0.0
