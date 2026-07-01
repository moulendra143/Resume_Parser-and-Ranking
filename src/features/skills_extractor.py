import re

def extract_skills(text: str, taxonomy: list) -> list:
    """Extract skills from text based on a provided taxonomy list."""
    extracted = []
    text_lower = text.lower()
    
    for skill in taxonomy:
        # Create a regex pattern to match whole words/phrases
        # Escaping the skill to handle C++, C#, etc.
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, text_lower):
            extracted.append(skill)
            
    return extracted
