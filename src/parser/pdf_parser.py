import pdfplumber

def extract_text(pdf_path: str) -> str:
    """Extract raw text from a PDF file."""
    text_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_pages.append(page_text)
    return "\n".join(text_pages)
