from __future__ import annotations

from typing import List


def simple_chunks(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    """Простой чанкер по символам (для MVP).
    Для продакшена лучше семантический/структурный чанкинг.
    """
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + max_chars)
        chunks.append(text[i:j].strip())
        i = max(i + (max_chars - overlap), j)
    return [c for c in chunks if c]
