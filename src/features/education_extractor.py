import re

def extract_education(text: str) -> list:
    """Extract education degree names from text."""
    degree_patterns = [
        r"\b(bachelor(?:\s+of)?\s+[a-zA-Z\s]+)\b",
        r"\b(master(?:\s+of)?\s+[a-zA-Z\s]+)\b",
        r"\b(ph\.?d(?:\s+in\s+[a-zA-Z\s]+)?)\b",
        r"\b(b\.?tech(?:\s+in\s+[a-zA-Z\s]+)?)\b",
        r"\b(m\.?tech(?:\s+in\s+[a-zA-Z\s]+)?)\b",
        r"\b(b\.?sc(?:\s+in\s+[a-zA-Z\s]+)?)\b",
        r"\b(m\.?sc(?:\s+in\s+[a-zA-Z\s]+)?)\b",
        r"\b(b\.?a(?:\s+in\s+[a-zA-Z\s]+)?)\b",
        r"\b(m\.?a(?:\s+in\s+[a-zA-Z\s]+)?)\b",
        r"\b(b\.?s(?:\s+in\s+[a-zA-Z\s]+)?)\b",
        r"\b(m\.?s(?:\s+in\s+[a-zA-Z\s]+)?)\b",
    ]
    
    matches = []
    for pat in degree_patterns:
        found = re.findall(pat, text, re.IGNORECASE)
        # Clean up matched strings and add to list
        for match in found:
            # Handle possible capture groups
            if isinstance(match, tuple):
                match = match[0]
            matches.append(re.sub(r"\s+", " ", match).strip().title())
            
    # Filter out duplicates while preserving some order
    return list(dict.fromkeys(matches))
