import re

def extract_skills(text: str, taxonomy: list) -> list:
    """Extract skills from text based on a provided taxonomy list."""
    extracted = []
    text_lower = text.lower()
    
    for skill in taxonomy:
        # If the skill starts/ends with alphanumeric/underscore, enforce boundary.
        # Otherwise, use lookaround to prevent matching part of another word.
        # e.g., .NET should not match inside "subnet", and C++ should match correctly.
        start_boundary = r"\b" if (skill[0].isalnum() or skill[0] == '_') else r"(?<=^|[^a-zA-Z0-9_])"
        end_boundary = r"\b" if (skill[-1].isalnum() or skill[-1] == '_') else r"(?=$|[^a-zA-Z0-9_])"
        
        pattern = start_boundary + re.escape(skill.lower()) + end_boundary
        if re.search(pattern, text_lower):
            extracted.append(skill)
            
    return extracted

