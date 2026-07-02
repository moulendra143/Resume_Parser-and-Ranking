"""
Taxonomy-based skill extractor with alias resolution.

Supports two extraction modes:

1. **Direct match** — scans text for each canonical skill name in the taxonomy
   using word-boundary-aware regex (handles special characters like C++, .NET,
   Node.js correctly).

2. **Alias match** — resolves common abbreviations and variants (e.g. "ML" →
   "Machine Learning", "k8s" → "Kubernetes", "sklearn" → "Scikit-learn")
   before deduplicating against already-matched canonical skills.

Both modes are case-insensitive.  The returned list contains only canonical
skill names from the taxonomy (no raw aliases), and is deduplicated.

Usage::

    from src.features.skills_extractor import extract_skills

    # taxonomy_data is the full dict loaded from skills_taxonomy.json
    skills = extract_skills(text, taxonomy_data["skills"], taxonomy_data.get("aliases", {}))
"""

import re
from typing import Dict, List


def _build_pattern(skill: str) -> re.Pattern:
    """Build a word-boundary-aware regex pattern for a skill name.

    Alphanumeric starts/ends use ``\\b``; special-char starts/ends (e.g. C++,
    .NET) use lookaround assertions to avoid matching inside other tokens.

    Args:
        skill: Canonical skill name (e.g. "C++", "Node.js", "Machine Learning").

    Returns:
        Compiled regex pattern for case-insensitive matching.
    """
    start = r"\b" if (skill[0].isalnum() or skill[0] == "_") else r"(?<=^|[^a-zA-Z0-9_])"
    end = r"\b" if (skill[-1].isalnum() or skill[-1] == "_") else r"(?=$|[^a-zA-Z0-9_])"
    return re.compile(start + re.escape(skill) + end, re.IGNORECASE)


def extract_skills(
    text: str,
    taxonomy: List[str],
    aliases: Dict[str, str] | None = None,
) -> List[str]:
    """Extract canonical skills from text using taxonomy matching and alias resolution.

    Args:
        text:     Raw or cleaned resume / job-description text.
        taxonomy: List of canonical skill names from ``skills_taxonomy.json``.
        aliases:  Optional dict mapping alias strings to canonical skill names,
                  e.g. ``{"ML": "Machine Learning", "k8s": "Kubernetes"}``.
                  Defaults to an empty dict (no alias resolution).

    Returns:
        Deduplicated list of canonical skill names found in the text, in the
        order they first appear in the taxonomy list.

    Examples:
        >>> extract_skills("Experience in ML and k8s", ["Machine Learning", "Kubernetes"],
        ...                {"ml": "Machine Learning", "k8s": "Kubernetes"})
        ['Machine Learning', 'Kubernetes']
    """
    if not text or not taxonomy:
        return []

    aliases = aliases or {}
    text_lower = text.lower()
    matched: set = set()
    results: List[str] = []

    # ------------------------------------------------------------------ #
    # Pass 1: Direct canonical-name matching
    # ------------------------------------------------------------------ #
    for skill in taxonomy:
        if skill in matched:
            continue
        pattern = _build_pattern(skill)
        if pattern.search(text):
            matched.add(skill)
            results.append(skill)

    # ------------------------------------------------------------------ #
    # Pass 2: Alias resolution
    # Scan for alias strings; when found, add the canonical skill name
    # if it hasn't already been matched via Pass 1.
    # ------------------------------------------------------------------ #
    for alias, canonical in aliases.items():
        if canonical in matched:
            continue  # already found via direct match — skip
        if canonical not in taxonomy:
            continue  # alias points to a skill not in this taxonomy — skip

        alias_lower = alias.lower()
        # Build boundary-aware pattern for the alias
        start = r"\b" if (alias_lower[0].isalnum() or alias_lower[0] == "_") else r"(?<=^|[^a-zA-Z0-9_])"
        end = r"\b" if (alias_lower[-1].isalnum() or alias_lower[-1] == "_") else r"(?=$|[^a-zA-Z0-9_])"
        alias_pattern = re.compile(start + re.escape(alias_lower) + end, re.IGNORECASE)

        if alias_pattern.search(text_lower):
            matched.add(canonical)
            results.append(canonical)

    return results
