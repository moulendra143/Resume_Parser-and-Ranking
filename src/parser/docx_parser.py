import docx

def extract_text(docx_path: str) -> str:
    """Extract raw text from a DOCX file."""
    doc = docx.Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])
