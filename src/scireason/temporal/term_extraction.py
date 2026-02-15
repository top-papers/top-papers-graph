from __future__ import annotations

"""Domain-agnostic term / keyphrase extraction.

We deliberately keep this module dependency-light (no spaCy/scispaCy), because the project
is used in a course where students might not have heavy NLP stacks installed.

The default extractor is a simplified RAKE (Rapid Automatic Keyword Extraction):
it works reasonably well on scientific abstracts across domains.
"""

from dataclasses import dataclass
import re
from typing import Iterable, List, Optional, Sequence, Set


_EN_STOPWORDS: Set[str] = {
    # A compact stopword list (avoid huge dependencies)
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "these",
    "this",
    "to",
    "was",
    "were",
    "with",
    "we",
    "our",
    "they",
    "which",
    "can",
    "may",
    "might",
    "using",
    "use",
    "used",
    "based",
    "method",
    "methods",
    "result",
    "results",
    "study",
    "studies",
    "analysis",
    "data",
    "model",
    "models",
}


_RU_STOPWORDS: Set[str] = {
    "и",
    "а",
    "но",
    "что",
    "это",
    "как",
    "в",
    "во",
    "на",
    "по",
    "к",
    "у",
    "о",
    "об",
    "от",
    "до",
    "для",
    "из",
    "при",
    "без",
    "с",
    "со",
    "же",
    "ли",
    "бы",
    "не",
    "нет",
    "мы",
    "вы",
    "они",
    "он",
    "она",
    "оно",
    "их",
    "наш",
    "ваш",
}


_WORD_RE = re.compile(r"[\w\-]+", flags=re.UNICODE)


def default_stopwords(language: str = "en") -> Set[str]:
    lang = (language or "en").lower()
    if lang.startswith("ru"):
        return set(_RU_STOPWORDS)
    return set(_EN_STOPWORDS)


@dataclass(frozen=True)
class TermCandidate:
    term: str
    score: float


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "") if t]


def extract_terms_rake(
    text: str,
    *,
    max_terms: int = 25,
    max_words_per_term: int = 4,
    min_word_len: int = 3,
    language: str = "en",
    extra_stopwords: Optional[Iterable[str]] = None,
    boost_terms: Optional[Sequence[str]] = None,
) -> List[TermCandidate]:
    """Extract keyphrases using a simplified RAKE.

    Parameters
    ----------
    boost_terms:
        Optional domain keywords (e.g., from DomainConfig.keywords). If provided,
        these phrases get a score bump to survive early course datasets.
    """

    stop = default_stopwords(language)
    if extra_stopwords:
        stop.update([str(s).lower().strip() for s in extra_stopwords if str(s).strip()])

    tokens = _tokenize(text)
    # Build candidate phrases split by stopwords
    phrases: List[List[str]] = []
    buf: List[str] = []
    for w in tokens:
        if (w in stop) or (len(w) < min_word_len):
            if buf:
                phrases.append(buf)
                buf = []
            continue
        buf.append(w)
    if buf:
        phrases.append(buf)

    # Limit phrase length (keep phrases that are not too long)
    phrases = [p[:max_words_per_term] for p in phrases if p]

    # Word frequency / degree
    freq: dict[str, int] = {}
    degree: dict[str, int] = {}
    for p in phrases:
        unique = [w for w in p if w]
        if not unique:
            continue
        deg = max(0, len(unique) - 1)
        for w in unique:
            freq[w] = freq.get(w, 0) + 1
            degree[w] = degree.get(w, 0) + deg

    # Word score
    word_score: dict[str, float] = {}
    for w, f in freq.items():
        # Classic RAKE often uses (degree + freq) / freq
        word_score[w] = float(degree.get(w, 0) + f) / float(f)

    # Phrase score
    phrase_score: dict[str, float] = {}
    for p in phrases:
        if not p:
            continue
        term = " ".join(p).strip()
        if not term:
            continue
        score = sum(word_score.get(w, 0.0) for w in p)
        # penalize very short generic terms
        if len(term) < 3:
            score *= 0.5
        phrase_score[term] = max(phrase_score.get(term, 0.0), score)

    # Optional boosting
    if boost_terms:
        for bt in boost_terms:
            t = str(bt or "").strip().lower()
            if not t:
                continue
            if t in phrase_score:
                phrase_score[t] *= 1.25
            else:
                # add with small default score
                phrase_score[t] = max(phrase_score.get(t, 0.0), 0.8)

    ranked = sorted(phrase_score.items(), key=lambda kv: kv[1], reverse=True)

    # Deduplicate by simple normalization
    out: List[TermCandidate] = []
    seen: Set[str] = set()
    for term, score in ranked:
        key = term.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(TermCandidate(term=term, score=float(score)))
        if len(out) >= max_terms:
            break
    return out
