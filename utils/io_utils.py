from __future__ import annotations

import re

def safe(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    _map = {'—': '-', '–': '-', '‒': '-', '‐': '-', '‑': '-', '‘': "'", '’': "'", '“': '"', '”': '"', '•': '-', '…': '...', '²': '2', '³': '3', '°': ' deg', '±': '+/-', '→': '->', '≥': '>=', '≤': '<=', '≠': '!=', 'µ': 'u', '×': 'x'}
    for c, r in _map.items():
        text = text.replace(c, r)
    return text.encode('latin-1', errors='ignore').decode('latin-1')


def clean_text(text: str) -> str:
    text = re.sub('#{1,6}\\s*', '', text)
    text = re.sub('\\*\\*(.*?)\\*\\*', '\\1', text)
    text = re.sub('\\*(.*?)\\*', '\\1', text)
    text = re.sub('`(.*?)`', '\\1', text)
    return safe(text)
