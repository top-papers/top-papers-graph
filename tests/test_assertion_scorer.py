"""Unit tests for assertion quality scorer (logistic regression on 10 features)."""
import numpy as np
import pytest
from dataclasses import dataclass, field
from typing import Dict, Set, List, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scireason.temporal.assertion_scorer import (
    extract_features, compute_corpus_stats, assertion_score,
    train_scorer_weights, save_weights, load_weights,
    AssertionScorer, CorpusStats,
    DEFAULT_WEIGHTS, DEFAULT_BIAS, N_FEATURES, FEATURE_NAMES,
    PREDICATE_STRENGTH,
    _token_overlap, _sigmoid, _stopword_ratio, _word_count_quality, _alpha_ratio,
)


# === Fixtures ===

@dataclass
class MockNode:
    term: str = ""
    doc_freq: int = 1


@dataclass
class MockEdge:
    source: str = ""
    target: str = ""
    predicate: str = ""
    total_count: int = 1
    mean_confidence: float = 0.8
    papers: set = field(default_factory=lambda: {"p1"})
    evidence_quotes: list = field(default_factory=list)
    polarity_counts: dict = field(default_factory=lambda: {"supports": 1, "contradicts": 0, "unknown": 0})
    features: dict = field(default_factory=dict)


def make_good_edge():
    return MockEdge(
        source="causal_inference", target="credit_limit_management",
        predicate="causes", total_count=5,
        mean_confidence=0.9, papers={"p1", "p2"},
        evidence_quotes=[{"quote": "causal inference improves credit limit management decisions"}],
        features={"pmi": 2.0},
    )


def make_bad_edge():
    return MockEdge(
        source="model", target="data",
        predicate="is_a", total_count=1,
        mean_confidence=1.0, papers={"p1"},
        evidence_quotes=[],
        features={"pmi": 0.1},
    )


def make_nodes():
    return {
        "causal_inference": MockNode(term="causal_inference", doc_freq=2),
        "credit_limit_management": MockNode(term="credit_limit_management", doc_freq=3),
        "model": MockNode(term="model", doc_freq=10),
        "data": MockNode(term="data", doc_freq=10),
    }


# === Tests: Feature Extraction ===

class TestFeatureExtraction:
    def test_shape(self):
        nodes = make_nodes()
        edge = make_good_edge()
        stats = compute_corpus_stats([edge], nodes, 3)
        f = extract_features(edge, nodes, stats)
        assert f.shape == (N_FEATURES,)
        # На «хорошем» ребре хотя бы часть признаков должна быть ненулевой —
        # иначе extract_features не считает контент, а просто возвращает нули.
        assert (f > 0).sum() >= 5

    def test_all_features_in_range(self):
        nodes = make_nodes()
        edge = make_good_edge()
        stats = compute_corpus_stats([edge], nodes, 3)
        f = extract_features(edge, nodes, stats)
        for i, val in enumerate(f):
            assert 0.0 <= val <= 1.0, f"Feature {FEATURE_NAMES[i]} = {val} out of [0,1]"

    def test_good_edge_scores_higher(self):
        nodes = make_nodes()
        good = make_good_edge()
        bad = make_bad_edge()
        stats = compute_corpus_stats([good, bad], nodes, 3)
        f_good = extract_features(good, nodes, stats)
        f_bad = extract_features(bad, nodes, stats)
        s_good = assertion_score(f_good, DEFAULT_WEIGHTS, DEFAULT_BIAS)
        s_bad = assertion_score(f_bad, DEFAULT_WEIGHTS, DEFAULT_BIAS)
        assert s_good > s_bad, f"Good={s_good:.3f} should be > Bad={s_bad:.3f}"

    def test_predicate_strength_feature(self):
        nodes = make_nodes()
        edge_causal = MockEdge(source="a", target="b", predicate="causes")
        edge_weak = MockEdge(source="a", target="b", predicate="is_a")
        nodes["a"] = MockNode(doc_freq=1)
        nodes["b"] = MockNode(doc_freq=1)
        stats = compute_corpus_stats([edge_causal, edge_weak], nodes, 1)
        f1 = extract_features(edge_causal, nodes, stats)
        f2 = extract_features(edge_weak, nodes, stats)
        # predicate_strength is feature index 2
        assert f1[2] > f2[2]

    def test_empty_evidence(self):
        nodes = {"a": MockNode(doc_freq=1), "b": MockNode(doc_freq=1)}
        edge = MockEdge(source="a", target="b", predicate="causes", evidence_quotes=[])
        stats = compute_corpus_stats([edge], nodes, 1)
        f = extract_features(edge, nodes, stats)
        assert f[4] == 0.0  # evidence_grounding
        assert f[5] == 0.0  # evidence_count (log1p(0)/log1p(max) with max=0 edge case)

    def test_missing_node(self):
        """Edge with entity not in nodes dict should not crash."""
        nodes = {"a": MockNode(doc_freq=1)}  # "b" missing
        edge = MockEdge(source="a", target="b", predicate="causes")
        stats = compute_corpus_stats([edge], nodes, 1)
        f = extract_features(edge, nodes, stats)
        assert f.shape == (N_FEATURES,)


# === Tests: Scoring ===

class TestScoring:
    def test_score_responsive_to_input(self):
        """Sigmoid реально реагирует на вход: нулевой → средний → максимальный."""
        s_zero = assertion_score(np.zeros(N_FEATURES), DEFAULT_WEIGHTS, DEFAULT_BIAS)
        s_half = assertion_score(np.full(N_FEATURES, 0.5), DEFAULT_WEIGHTS, DEFAULT_BIAS)
        s_one = assertion_score(np.ones(N_FEATURES), DEFAULT_WEIGHTS, DEFAULT_BIAS)
        assert 0.0 <= s_zero < s_half < s_one <= 1.0

    def test_predicate_strength_contributes_positively(self):
        """Бамп одного признака с положительным весом → скор растёт.

        Проверяет, что линейная комбинация реально складывается из вкладов
        отдельных признаков, а не игнорирует часть из них.
        """
        # predicate_strength (idx 2) должен иметь положительный вес в дефолтных.
        assert DEFAULT_WEIGHTS[2] > 0
        base = np.full(N_FEATURES, 0.5)
        bumped = base.copy()
        bumped[2] = 1.0
        s_base = assertion_score(base, DEFAULT_WEIGHTS, DEFAULT_BIAS)
        s_bumped = assertion_score(bumped, DEFAULT_WEIGHTS, DEFAULT_BIAS)
        assert s_bumped > s_base

    def test_sigmoid_bounds(self):
        assert _sigmoid(100) == pytest.approx(1.0, abs=1e-6)
        assert _sigmoid(-100) == pytest.approx(0.0, abs=1e-6)
        assert _sigmoid(0) == pytest.approx(0.5, abs=1e-6)


# === Tests: Training ===

class TestTraining:
    def test_convergence(self):
        """Loss should decrease over epochs."""
        np.random.seed(42)
        pos = np.random.uniform(0.5, 1.0, (20, N_FEATURES))
        neg = np.random.uniform(0.0, 0.5, (5, N_FEATURES))
        w, b, log = train_scorer_weights(pos, neg, n_epochs=100, lr=0.01)
        assert log["final_loss"] <= log["loss_history_sample"][0]

    def test_separation_generalises_to_heldout(self):
        """После обучения скор разделяет pos/neg на ОТЛОЖЕННОЙ выборке.

        Прежний вариант сравнивал скоры на обучающих данных — это
        тавтология (следует прямо из BCE-лосса). Здесь проверяется
        обобщение на новые сэмплы из того же распределения, и требуется
        заметный (> 0.3) разрыв средних скоров, а не любая разница.
        """
        np.random.seed(42)
        pos_train = np.random.uniform(0.5, 1.0, (30, N_FEATURES))
        neg_train = np.random.uniform(0.0, 0.4, (10, N_FEATURES))
        w, b, _ = train_scorer_weights(pos_train, neg_train, n_epochs=200, lr=0.01)
        pos_test = np.random.uniform(0.5, 1.0, (15, N_FEATURES))
        neg_test = np.random.uniform(0.0, 0.4, (15, N_FEATURES))
        s_pos = np.array([assertion_score(x, w, b) for x in pos_test])
        s_neg = np.array([assertion_score(x, w, b) for x in neg_test])
        assert s_pos.mean() - s_neg.mean() > 0.3

    def test_output_shapes(self):
        pos = np.random.uniform(0, 1, (10, N_FEATURES))
        neg = np.random.uniform(0, 1, (5, N_FEATURES))
        w, b, log = train_scorer_weights(pos, neg, n_epochs=10)
        assert w.shape == (N_FEATURES,)
        assert isinstance(b, float)
        assert "final_loss" in log

    def test_empty_negatives_does_not_crash(self):
        pos = np.random.uniform(0, 1, (10, N_FEATURES))
        neg = np.zeros((0, N_FEATURES))
        w, b, log = train_scorer_weights(pos, neg, n_epochs=10)
        assert w.shape == (N_FEATURES,)


# === Tests: Serialization ===

class TestSerialization:
    def test_roundtrip(self, tmp_path):
        w = np.random.randn(N_FEATURES)
        b = -3.5
        path = tmp_path / "weights.json"
        save_weights(str(path), w, b, training_meta={"test": True})
        w2, b2 = load_weights(str(path))
        np.testing.assert_array_almost_equal(w, w2)
        assert abs(b - b2) < 1e-10

    def test_missing_file_returns_defaults(self, tmp_path):
        w, b = load_weights(str(tmp_path / "nonexistent.json"))
        np.testing.assert_array_equal(w, DEFAULT_WEIGHTS)
        assert b == DEFAULT_BIAS


# === Tests: AssertionScorer class ===

class TestAssertionScorer:
    def test_load_defaults(self):
        scorer = AssertionScorer()
        assert scorer.weights is not None
        assert scorer.bias == DEFAULT_BIAS

    def test_score_edge(self):
        scorer = AssertionScorer()
        nodes = make_nodes()
        edge = make_good_edge()
        stats = compute_corpus_stats([edge], nodes, 3)
        s = scorer.score_edge(edge, nodes, stats)
        assert 0.0 < s < 1.0

    def test_batch_scoring(self):
        scorer = AssertionScorer()
        nodes = make_nodes()
        edges = [make_good_edge(), make_bad_edge()]
        stats = compute_corpus_stats(edges, nodes, 3)
        scores = scorer.score_edges_batch(edges, nodes, stats)
        assert len(scores) == 2
        assert scores[0] > scores[1]


# === Tests: Helper functions ===

class TestHelpers:
    def test_token_overlap(self):
        assert _token_overlap("causal inference improves decisions", "causal inference") > 0
        assert _token_overlap("completely unrelated text", "causal inference") == 0.0
        assert _token_overlap("", "") == 0.0

    def test_stopword_ratio(self):
        assert _stopword_ratio("moreover however thus") == 1.0
        assert _stopword_ratio("causal inference method") == 0.0
        assert _stopword_ratio("") == 0.0

    def test_word_count_quality(self):
        assert _word_count_quality("one") == 0.3  # too short
        assert _word_count_quality("two words") == 1.0
        assert _word_count_quality("three good words") == 1.0
        assert _word_count_quality("a b c d e f g h i j k") < 0.5  # too long

    def test_alpha_ratio(self):
        assert _alpha_ratio("hello world") > 0.9
        assert _alpha_ratio("x=0.5*y+z") < 0.5
        assert _alpha_ratio("") == 0.0

    def test_predicate_strength_coverage(self):
        """Key predicates should be in the strength table."""
        assert "causes" in PREDICATE_STRENGTH
        assert "outperforms" in PREDICATE_STRENGTH
        assert "is_a" in PREDICATE_STRENGTH
        assert PREDICATE_STRENGTH["causes"] > PREDICATE_STRENGTH["is_a"]


# === Tests: Corpus Stats ===

class TestCorpusStats:
    def test_compute(self):
        nodes = make_nodes()
        edges = [make_good_edge(), make_bad_edge()]
        stats = compute_corpus_stats(edges, nodes, 3)
        assert stats.max_total_count >= 5
        assert stats.max_doc_freq >= 10
        assert stats.n_total_papers == 3
        assert len(stats.node_degree) > 0

    def test_empty_edges(self):
        stats = compute_corpus_stats([], {}, 0)
        assert stats.max_total_count == 1
        assert stats.n_total_papers == 1
