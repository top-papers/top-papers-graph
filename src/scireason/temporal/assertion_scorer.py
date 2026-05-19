"""Скоринг качества утверждений — взвешенная оценочная функция на 10 признаках.

Заменяет ненадёжную LLM-confidence (почти всегда ≈ 1.0) на вычисляемую
метрику качества, основанную исключительно на наблюдаемых данных.

Формула:
    score(edge) = sigmoid(w · features + bias)  ∈ [0, 1]

По сути это weighted scoring function — взвешенная сумма 10 эвристик
с нормализацией через sigmoid.  Логистическая регрессия здесь
используется НЕ как ML-модель для предсказания, а исключительно как
аналитический способ подобрать веса вместо экспертных:
- Начальные веса (DEFAULT_WEIGHTS) задаются экспертно;
- При наличии размеченных данных (accept/reject) веса можно
  «дообучить» через градиентный спуск на комбинированном лоссе:
      L = α·L_rank + β·L_bce + γ·L_reg
  Это позволяет автоматически подстроить баланс признаков под
  конкретный домен, не трогая ручные пороги.

Пример использования::

    scorer = AssertionScorer.load()
    stats  = scorer.compute_corpus_stats(edges, nodes, n_papers)
    for e in edges.values():
        e.score = scorer.score_edge(e, nodes, stats)
"""

from __future__ import annotations

import json
import math
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

# Имена 10 признаков, извлекаемых из каждого ребра графа.
# Каждый признак нормирован в [0, 1] и отражает один аспект качества.
FEATURE_NAMES: List[str] = [
    "redundancy",           # сколько чанков независимо извлекли это ребро
    "entity_specificity",   # IDF сущностей (редкие > частые)
    "predicate_strength",   # экспертная таблица: causes=1.0, is_a=0.2
    "mean_confidence",      # LLM self-reported confidence (слабый сигнал)
    "evidence_grounding",   # токенное пересечение цитаты и сущностей
    "evidence_count",       # количество подтверждающих цитат
    "paper_diversity",      # поддержка из нескольких статей
    "polarity_clarity",     # согласованность полярности (supports/contradicts)
    "graph_integration",    # связанность вершин ребра с остальным графом
    "pmi_normalized",       # pointwise mutual information (информативность)
]

N_FEATURES = len(FEATURE_NAMES)

# Начальные экспертные веса.  Используются как fallback, если файл
# обученных весов отсутствует.  При обучении (train_scorer_weights)
# логистическая регрессия корректирует эти веса на основе размеченных
# данных — это единственная причина использования LR: аналитический
# подбор весов взамен ручного.
DEFAULT_WEIGHTS = np.array([
    1.5,   # redundancy — повторное извлечение из разных чанков ≈ надёжность
    0.8,   # entity_specificity — конкретные сущности ценнее общих
    1.2,   # predicate_strength — causes >> is_a
    0.3,   # mean_confidence — LLM-confidence ненадёжна, вес занижен
    1.0,   # evidence_grounding — цитата упоминает сущности = хорошо
    0.7,   # evidence_count — больше цитат ≈ реальнее
    1.0,   # paper_diversity — кросс-документная поддержка = сильный сигнал
    0.5,   # polarity_clarity — согласованная полярность
    0.6,   # graph_integration — связанные вершины ≈ реальные
    0.4,   # pmi_normalized — умеренный сигнал информативности
], dtype=np.float64)

# Смещение подобрано так, чтобы «типичное» ребро получало score ≈ 0.5
DEFAULT_BIAS: float = -4.5

PREDICATE_STRENGTH: Dict[str, float] = {
    # Strong causal / directional
    "causes": 1.0, "leads_to": 0.95, "results_in": 0.90,
    "prevents": 0.90, "inhibits": 0.90, "drives": 0.90,
    "induces": 0.90, "triggers": 0.90, "promotes": 0.85,
    # Effect / comparison
    "improves": 0.85, "reduces": 0.85, "increases": 0.85,
    "decreases": 0.85, "outperforms": 0.85,
    "affects": 0.80, "influences": 0.80,
    # Dependency / prerequisite
    "depends_on": 0.75, "requires": 0.75, "enables": 0.75,
    "assumes": 0.70,
    # Prediction / temporal
    "predicts": 0.80, "precedes": 0.75, "follows": 0.75,
    # Usage / application
    "uses": 0.60, "used_for": 0.60, "applied_to": 0.60,
    "are_used_for": 0.60, "is_used_for": 0.60, "used_in": 0.60,
    "combined_with": 0.55, "compared_to": 0.55,
    "incorporates": 0.55, "leverages": 0.55, "captures": 0.55,
    # Relational
    "measures": 0.50, "reflects": 0.50, "manages": 0.50,
    "addresses": 0.50, "accounts_for": 0.50,
    "is_computed_from": 0.45, "is_determined_by": 0.45,
    "is_measured_by": 0.45, "arises_from": 0.45,
    "involves": 0.40, "allows": 0.40,
    # Weak / associative (high rejection rates: 75% and 50%)
    "associated_with": 0.10, "related_to": 0.10,
    "contradicts": 0.70, "supports": 0.65, "extends": 0.60,
    # Taxonomic / definitional
    "is_a": 0.20, "is_type_of": 0.20, "is_part_of": 0.25,
    # Junk
    "cooccurs_with": 0.05,
    "is_segmented_into": 0.05, "consists_of": 0.10,
    "was_destroyed_after_experiment": 0.0, "calculated_at": 0.05,
    "is_estimated_to": 0.05, "is_equal_to": 0.05,
    "acknowledges": 0.0, "thanks": 0.0,
}

PREDICATE_STRENGTH_DEFAULT = 0.40


# ---------------------------------------------------------------------------
# Извлечение признаков
# ---------------------------------------------------------------------------

def _sigmoid(z: float) -> float:
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def _token_set(text: str) -> set:
    return set(text.lower().replace("_", " ").split())


def _token_overlap(quote: str, entities: str) -> float:
    """Jaccard-сходство токенов цитаты и токенов сущностей."""
    q = _token_set(quote)
    e = _token_set(entities)
    if not q or not e:
        return 0.0
    inter = len(q & e)
    union = len(q | e)
    return inter / union if union else 0.0


_ENTITY_STOPWORDS = frozenset({
    "the", "a", "an", "of", "in", "to", "for", "and", "or", "is", "are",
    "was", "were", "be", "been", "by", "with", "from", "on", "at", "as",
    "it", "this", "that", "which", "their", "our", "we", "they", "also",
    "can", "may", "more", "most", "such", "some", "these", "those", "other",
    "all", "each", "both", "its", "has", "have", "had", "will", "would",
    "could", "should", "do", "does", "did", "not", "no", "but", "if", "than",
    "when", "where", "who", "how", "what", "let", "moreover", "furthermore",
    "however", "thus", "therefore", "hence", "about", "into", "many",
})


def _stopword_ratio(text: str) -> float:
    """Доля стоп-слов среди токенов."""
    tokens = text.lower().replace("_", " ").split()
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in _ENTITY_STOPWORDS) / len(tokens)


def _word_count_quality(text: str) -> float:
    """1.0 для сущностей из 2–5 слов, штраф за более короткие/длинные."""
    n = len(text.replace("_", " ").split())
    if 2 <= n <= 5:
        return 1.0
    if n == 1:
        return 0.3  # too short
    if n == 6:
        return 0.6
    return max(0.0, 1.0 - (n - 5) * 0.15)  # long → penalty


def _alpha_ratio(text: str) -> float:
    """Доля буквенных символов (vs цифры/знаки)."""
    if not text:
        return 0.0
    alpha = sum(1 for c in text if c.isalpha() or c in " _")
    return alpha / len(text)


def _entity_contained_in_quote(entity: str, quote: str) -> bool:
    """Содержится ли строка сущности в цитате (подстрока без учёта регистра)."""
    return entity.lower().replace("_", " ") in quote.lower()


@dataclass
class CorpusStats:
    """Заранее посчитанная корпусная статистика для нормализации признаков."""
    max_total_count: int = 1
    max_doc_freq: int = 1
    max_evidence_count: int = 1
    n_total_papers: int = 1
    node_degree: Dict[str, int] = field(default_factory=dict)


def compute_corpus_stats(
    edges: Iterable[Any],
    nodes: Dict[str, Any],
    n_total_papers: int,
) -> CorpusStats:
    """Посчитать нормировочные константы по всему набору рёбер/нод."""
    max_tc = 1
    max_ec = 1
    degree: Dict[str, int] = {}
    for e in edges:
        tc = getattr(e, "total_count", 1)
        if tc > max_tc:
            max_tc = tc
        ec = len(getattr(e, "evidence_quotes", []))
        if ec > max_ec:
            max_ec = ec
        s = getattr(e, "source", "")
        t = getattr(e, "target", "")
        degree[s] = degree.get(s, 0) + 1
        degree[t] = degree.get(t, 0) + 1

    max_df = 1
    for n in nodes.values():
        df = getattr(n, "doc_freq", 0)
        if df > max_df:
            max_df = df

    return CorpusStats(
        max_total_count=max_tc,
        max_doc_freq=max_df,
        max_evidence_count=max_ec,
        n_total_papers=max(1, n_total_papers),
        node_degree=degree,
    )


def extract_features(
    edge: Any,
    nodes: Dict[str, Any],
    stats: CorpusStats,
) -> np.ndarray:
    """Извлечь 10-мерный нормализованный вектор признаков из ребра графа."""
    f = np.zeros(N_FEATURES, dtype=np.float64)

    # 0: redundancy
    tc = getattr(edge, "total_count", 1)
    f[0] = math.log1p(tc) / math.log1p(stats.max_total_count) if stats.max_total_count > 0 else 0.0

    # 1: entity_specificity
    src = getattr(edge, "source", "")
    tgt = getattr(edge, "target", "")
    src_df = getattr(nodes.get(src), "doc_freq", 0) if nodes.get(src) else 0
    tgt_df = getattr(nodes.get(tgt), "doc_freq", 0) if nodes.get(tgt) else 0
    log_max = math.log1p(stats.max_doc_freq)
    if log_max > 0:
        spec_s = 1.0 - math.log1p(src_df) / log_max
        spec_t = 1.0 - math.log1p(tgt_df) / log_max
        f[1] = (spec_s + spec_t) / 2.0
    else:
        f[1] = 0.5

    # 2: predicate_strength
    pred = getattr(edge, "predicate", "").lower().strip()
    f[2] = PREDICATE_STRENGTH.get(pred, PREDICATE_STRENGTH_DEFAULT)

    # 3: mean_confidence
    f[3] = max(0.0, min(1.0, getattr(edge, "mean_confidence", 0.5)))

    # 4: evidence_grounding
    quotes = getattr(edge, "evidence_quotes", [])
    entities_text = f"{src} {tgt}"
    if quotes:
        best_overlap = 0.0
        for q in quotes[:5]:  # check up to 5 quotes
            quote_text = ""
            if isinstance(q, dict):
                quote_text = str(q.get("quote") or q.get("snippet_or_summary") or "")
            elif isinstance(q, str):
                quote_text = q
            ov = _token_overlap(quote_text, entities_text)
            if ov > best_overlap:
                best_overlap = ov
        f[4] = best_overlap
    else:
        f[4] = 0.0

    # 5: evidence_count
    ec = len(quotes)
    f[5] = math.log1p(ec) / math.log1p(stats.max_evidence_count) if stats.max_evidence_count > 0 else 0.0

    # 6: paper_diversity
    papers = getattr(edge, "papers", set())
    f[6] = len(papers) / stats.n_total_papers

    # 7: polarity_clarity
    pol = getattr(edge, "polarity_counts", {})
    pol_vals = [v for v in pol.values() if isinstance(v, (int, float))]
    total_pol = sum(pol_vals)
    f[7] = max(pol_vals) / total_pol if total_pol > 0 else 0.5

    # 8: graph_integration
    deg_s = stats.node_degree.get(src, 0)
    deg_t = stats.node_degree.get(tgt, 0)
    if deg_s > 1 and deg_t > 1:
        f[8] = 1.0
    elif deg_s > 1 or deg_t > 1:
        f[8] = 0.5
    else:
        f[8] = 0.0

    # 9: pmi_normalized
    pmi_raw = 0.0
    feats = getattr(edge, "features", {})
    if isinstance(feats, dict):
        pmi_raw = float(feats.get("pmi", 0.0) or 0.0)
    f[9] = _sigmoid(pmi_raw)

    return f


# ---------------------------------------------------------------------------
# Вычисление score
# ---------------------------------------------------------------------------

def assertion_score(features: np.ndarray, weights: np.ndarray, bias: float) -> float:
    """Взвешенная оценочная функция: score = sigmoid(w · features + bias) ∈ [0, 1].

    Sigmoid используется исключительно для нормализации суммы в диапазон [0, 1].
    """
    z = float(np.dot(weights, features)) + bias
    return _sigmoid(z)


# ---------------------------------------------------------------------------
# Подбор весов через логистическую регрессию
#
# Логистическая регрессия здесь — это НЕ предсказательная модель.
# Это аналитический способ подобрать веса оценочной функции на основе
# размеченных данных (accept/reject), чтобы не оставлять их экспертными.
# Результат — тот же вектор весов w и смещение bias, но подстроенные
# под конкретный домен.
#
# Лосс: BCE + L2-регуляризация (стандартная логистическая регрессия).
# ---------------------------------------------------------------------------

def _sigmoid_vec(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def train_scorer_weights(
    features_accept: np.ndarray,
    features_reject: np.ndarray,
    features_review: Optional[np.ndarray] = None,
    *,
    init_weights: Optional[np.ndarray] = None,
    init_bias: Optional[float] = None,
    lr: float = 0.01,
    n_epochs: int = 500,
    reg_lambda: float = 0.01,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """Подбор весов оценочной функции через градиентный спуск.

    Стандартная логистическая регрессия (BCE + L2-регуляризация).
    Начальная точка — экспертные веса (DEFAULT_WEIGHTS).

    Returns (weights, bias, training_log).
    """
    w = (init_weights.copy() if init_weights is not None else DEFAULT_WEIGHTS.copy())
    b = (init_bias if init_bias is not None else DEFAULT_BIAS)

    n_accept = len(features_accept)
    n_reject = len(features_reject)

    loss_history: List[float] = []

    for epoch in range(n_epochs):
        # --- Прямой проход: score = sigmoid(w · f + b) ---
        z_acc = features_accept @ w + b
        s_acc = _sigmoid_vec(z_acc)

        z_rej = features_reject @ w + b
        s_rej = _sigmoid_vec(z_rej)

        # --- BCE: accept → y=1, reject → y=0 ---
        eps = 1e-12
        bce_acc = -np.log(s_acc + eps).mean() if n_accept > 0 else 0.0
        bce_rej = -np.log(1.0 - s_rej + eps).mean() if n_reject > 0 else 0.0
        l_bce = (bce_acc + bce_rej) / 2.0

        # Градиент BCE: dL/dw = mean((s - y) * f)
        grad_w = np.zeros(N_FEATURES, dtype=np.float64)
        grad_b = 0.0
        if n_accept > 0:
            delta_acc = s_acc - 1.0
            grad_w += (delta_acc[:, None] * features_accept).mean(axis=0)
            grad_b += delta_acc.mean()
        if n_reject > 0:
            delta_rej = s_rej
            grad_w += (delta_rej[:, None] * features_reject).mean(axis=0)
            grad_b += delta_rej.mean()
        grad_w /= 2.0
        grad_b /= 2.0

        # --- L2-регуляризация ---
        l_reg = float(np.sum(w ** 2))
        grad_w += reg_lambda * 2.0 * w

        total_loss = l_bce + reg_lambda * l_reg

        # --- Обновление весов ---
        w -= lr * grad_w
        b -= lr * grad_b

        loss_history.append(float(total_loss))

    # Итоговые score для отчёта
    z_acc_final = features_accept @ w + b
    s_acc_final = _sigmoid_vec(z_acc_final)
    z_rej_final = features_reject @ w + b
    s_rej_final = _sigmoid_vec(z_rej_final)

    training_log = {
        "n_accept": n_accept,
        "n_reject": n_reject,
        "n_epochs": n_epochs,
        "final_loss": loss_history[-1] if loss_history else 0.0,
        "loss_history_sample": loss_history[::50],
        "mean_score_accept": float(s_acc_final.mean()) if n_accept > 0 else 0.0,
        "mean_score_reject": float(s_rej_final.mean()) if n_reject > 0 else 0.0,
        "hyperparameters": {"lr": lr, "reg_lambda": reg_lambda},
    }

    return w, float(b), training_log


# ---------------------------------------------------------------------------
# Сериализация весов
# ---------------------------------------------------------------------------

def save_weights(
    path: str | Path,
    weights: np.ndarray,
    bias: float,
    training_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Сохранить веса оценочной функции в JSON."""
    doc = {
        "version": "1.0",
        "feature_names": FEATURE_NAMES,
        "weights": weights.tolist(),
        "bias": bias,
        "predicate_strength_table": PREDICATE_STRENGTH,
        "training_meta": training_meta or {},
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")


def load_weights(path: str | Path) -> Tuple[np.ndarray, float]:
    """Загрузить веса из JSON. При ошибке возвращает экспертные DEFAULT_WEIGHTS."""
    p = Path(path)
    if not p.exists():
        log.info("Assertion scorer weights file not found at %s, using defaults", p)
        return DEFAULT_WEIGHTS.copy(), DEFAULT_BIAS
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Assertion scorer weights file at %s is unreadable (%s), using defaults", p, exc)
        return DEFAULT_WEIGHTS.copy(), DEFAULT_BIAS
    w = np.array(doc["weights"], dtype=np.float64)
    b = float(doc["bias"])
    if len(w) != N_FEATURES:
        log.warning("Assertion scorer weights dimension mismatch (%d vs %d), using defaults", len(w), N_FEATURES)
        return DEFAULT_WEIGHTS.copy(), DEFAULT_BIAS
    return w, b


# ---------------------------------------------------------------------------
# Класс-обёртка для удобного использования
# ---------------------------------------------------------------------------

class AssertionScorer:
    """Оценщик качества утверждений: загружает веса и вычисляет score для рёбер."""

    def __init__(
        self,
        weights: Optional[np.ndarray] = None,
        bias: Optional[float] = None,
    ) -> None:
        self.weights = weights if weights is not None else DEFAULT_WEIGHTS.copy()
        self.bias = bias if bias is not None else DEFAULT_BIAS

    @classmethod
    def load(cls, path: Optional[str | Path] = None) -> "AssertionScorer":
        """Загрузить веса из JSON или использовать экспертные по умолчанию."""
        if path is None:
            from ..config import settings
            path = getattr(settings, "assertion_scorer_weights_path", "data/derived/assertion_scorer_weights.json")
        w, b = load_weights(path)
        return cls(weights=w, bias=b)

    def score_edge(
        self,
        edge: Any,
        nodes: Dict[str, Any],
        stats: CorpusStats,
    ) -> float:
        """Извлечь признаки и посчитать оценку качества для одного ребра."""
        feats = extract_features(edge, nodes, stats)
        return assertion_score(feats, self.weights, self.bias)

    def score_edges_batch(
        self,
        edges: Iterable[Any],
        nodes: Dict[str, Any],
        stats: CorpusStats,
    ) -> List[float]:
        """Посчитать оценку качества для набора рёбер."""
        return [self.score_edge(e, nodes, stats) for e in edges]
