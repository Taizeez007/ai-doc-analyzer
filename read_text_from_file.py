try:
   import pdfplumber
except ImportError:
   pdfplumber = None

try:
    import docx
except ImportError:
    docx = None

try:
    import requests
except ImportError:
    requests = None

def read_raw_text_from_file(path: str) -> str:
    lower = path.lower()
    try:
        if lower.endswith('.pdf') and pdfplumber:
            with pdfplumber.open(path) as pdf:
                return '\n'.join([p.extract_text() or '' for p in pdf.pages])
        elif lower.endswith('.docx') and docx:
            d = docx.Document(path)
            return '\n'.join([p.text.strip() for p in d.paragraphs if p.text.strip()])
        else:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return ""
