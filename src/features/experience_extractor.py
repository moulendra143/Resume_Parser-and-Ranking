import re

def extract_experience(text: str) -> float:
    """Extract total years of experience from text using heuristic regex matching."""
    # Look for patterns like "3 years", "5+ yrs", "2.5 years of experience"
    exp_patterns = r"(\d+(?:\.\d+)?)(?:\s*\+)?\s*(?:years?|yrs?)"
    matches = re.findall(exp_patterns, text, re.IGNORECASE)
    
    if not matches:
        return 0.0
        
    years = [float(m) for m in matches]
    # Often the highest number associated with "years" in a resume correlates to total experience
    # Though this is a simple heuristic and prone to edge cases (e.g., "10 years old")
    return max(years)
