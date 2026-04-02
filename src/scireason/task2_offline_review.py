from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .temporal.schemas import normalize_granularity


_MANIFEST_LIST_KEYS = ("assertions", "rows", "triplets", "edges", "corrections", "hits")


def _safe_json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_maybe_object(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list, tuple, int, float, bool)):
        return value
    text = str(value).strip()
    if not text:
        return ""
    if text[0] in "[{":
        try:
            return json.loads(text)
        except Exception:
            return text
    return text


def _pick_first(row: Dict[str, Any], keys: Iterable[str], default: Any = "") -> Any:
    for key in keys:
        if key not in row:
            continue
        value = row.get(key)
        if value is None:
            continue
        text = str(value)
        if text.strip() == "":
            continue
        return value
    return default


def _truncate_text(value: Any, limit: int = 220) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.replace("\n", " ").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _format_evidence_full(value: Any) -> str:
    obj = _parse_maybe_object(value)
    if obj is None:
        return ""
    if isinstance(obj, (dict, list)):
        try:
            return json.dumps(obj, ensure_ascii=False, indent=2)
        except Exception:
            return str(obj)
    return str(obj)


def _format_evidence_short(value: Any) -> str:
    obj = _parse_maybe_object(value)
    if isinstance(obj, dict):
        parts: list[str] = []
        paper_id = obj.get("paper_id") or obj.get("id")
        snippet = obj.get("snippet_or_summary") or obj.get("snippet") or obj.get("summary") or obj.get("quote")
        page = obj.get("page")
        locator = obj.get("figure_or_table") or obj.get("locator") or obj.get("caption")
        if paper_id:
            parts.append(f"paper: {paper_id}")
        if page not in (None, "", "nan"):
            parts.append(f"page: {page}")
        if locator:
            parts.append(f"locator: {locator}")
        if snippet:
            parts.append(str(snippet))
        text = " | ".join(parts) if parts else json.dumps(obj, ensure_ascii=False)
        return _truncate_text(text, limit=220)
    if isinstance(obj, list):
        return _truncate_text("; ".join(str(x) for x in obj[:5]), limit=220)
    return _truncate_text("" if obj is None else str(obj), limit=220)


def _read_rows(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    if p.suffix.lower() == ".json":
        payload = _safe_json_load(p)
        if isinstance(payload, list):
            return [x for x in payload if isinstance(x, dict)]
        if isinstance(payload, dict):
            for key in _MANIFEST_LIST_KEYS:
                value = payload.get(key)
                if isinstance(value, list):
                    return [x for x in value if isinstance(x, dict)]
            return [payload]
        return []

    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _resolve_bundle_artifact(manifest: Dict[str, Any], bundle_dir: Path, *keys: str, default_rel: str | None = None) -> Path | None:
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    for key in keys:
        candidate = manifest.get(key)
        if candidate:
            p = Path(candidate)
            if not p.is_absolute():
                p = bundle_dir / p
            if p.exists():
                return p
        art_candidate = artifacts.get(key)
        if art_candidate:
            p = bundle_dir / str(art_candidate)
            if p.exists():
                return p
    if default_rel:
        p = bundle_dir / default_rel
        if p.exists():
            return p
    return None

def _normalize_time_granularity(value: Any, *, start_date: Any = None, end_date: Any = None) -> str:
    raw = str(value or '').strip().lower()
    if raw == 'unknown':
        return 'unknown'
    return normalize_granularity(raw, start=start_date, end=end_date, default='unknown')

def _infer_hypothesis_role(row: Dict[str, Any]) -> str:
    predicate = str(row.get("predicate") or "").lower()
    subj = str(row.get("subject") or "").lower()
    obj = str(row.get("object") or "").lower()
    evidence = str(row.get("evidence_text") or "").lower()
    text = " ".join([predicate, subj, obj, evidence])
    if any(tok in text for tok in ("contrad", "inconsistent", "opposite", "negative result", "null")):
        return "contradiction"
    if any(tok in text for tok in ("increase", "decrease", "effect", "improve", "reduce", "cause", "lead", "induces")):
        return "mechanism"
    if any(tok in text for tok in ("treatment", "intervention", "drug", "stimulation", "protocol")):
        return "intervention"
    if any(tok in text for tok in ("assay", "measure", "detect", "marker", "readout", "metric")):
        return "measurement"
    if any(tok in text for tok in ("only when", "under", "condition", "environment", "temperature", "ph", "context")):
        return "boundary_condition"
    return "background"


def _infer_causal_status(row: Dict[str, Any]) -> str:
    predicate = str(row.get("predicate") or "").lower()
    text = " ".join([predicate, str(row.get("evidence_text") or "").lower()])
    if any(tok in text for tok in ("cause", "lead", "drives", "induces", "mediates")):
        return "causal"
    if any(tok in text for tok in ("associate", "correl", "linked", "related")):
        return "correlational"
    if any(tok in text for tok in ("model", "hypothesis", "suggest", "propose")):
        return "theoretical"
    return "descriptive"


def _default_time_type(row: Dict[str, Any]) -> str:
    ts = str(row.get("time_source") or "").lower()
    if ts in {"metadata", "publication_time"}:
        return "publication_time"
    return "observation_period"


def _extract_paper_refs(value: Any) -> tuple[list[str], list[str], list[str]]:
    obj = _parse_maybe_object(value)
    paper_ids: list[str] = []
    titles: list[str] = []
    source_refs: list[str] = []

    def _push_unique(target: list[str], candidate: Any) -> None:
        if candidate is None:
            return
        text = str(candidate).strip()
        if not text or text in target:
            return
        target.append(text)

    def _walk(node: Any) -> None:
        if node is None:
            return
        if isinstance(node, dict):
            _push_unique(paper_ids, node.get('paper_id') or node.get('id') or node.get('corpus_id') or node.get('doi'))
            _push_unique(titles, node.get('title') or node.get('paper_title') or node.get('name'))
            _push_unique(source_refs, node.get('source_ref') or node.get('url') or node.get('landing_page') or node.get('pdf_url'))
            for value in node.values():
                if isinstance(value, (dict, list, tuple)):
                    _walk(value)
            return
        if isinstance(node, (list, tuple)):
            for item in node:
                _walk(item)
            return
        _push_unique(paper_ids, node)

    _walk(obj)
    return paper_ids, titles, source_refs


def _default_review_state(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "verdict": str(row.get("verdict") or ""),
        "rationale": str(row.get("rationale") or ""),
        "time_source_note": str(row.get("time_source_note") or ""),
        "semantic_correctness": str(row.get("semantic_correctness") or ""),
        "evidence_sufficiency": str(row.get("evidence_sufficiency") or ""),
        "scope_match": str(row.get("scope_match") or ""),
        "system_match": str(row.get("system_match") or ""),
        "environment_match": str(row.get("environment_match") or ""),
        "protocol_match": str(row.get("protocol_match") or ""),
        "scope_overgeneralized": bool(row.get("scope_overgeneralized") or False),
        "corrected_scope_note": str(row.get("corrected_scope_note") or ""),
        "hypothesis_role": str(row.get("hypothesis_role") or _infer_hypothesis_role(row)),
        "hypothesis_relevance": str(row.get("hypothesis_relevance") or "1"),
        "testability_signal": str(row.get("testability_signal") or "1"),
        "causal_status": str(row.get("causal_status") or _infer_causal_status(row)),
        "severity": str(row.get("severity") or "warning"),
        "evidence_before_cutoff": str(row.get("evidence_before_cutoff") or ""),
        "leakage_risk": str(row.get("leakage_risk") or "possible"),
        "time_type": str(row.get("time_type") or _default_time_type(row)),
        "time_granularity": _normalize_time_granularity(row.get("time_granularity"), start_date=row.get("start_date"), end_date=row.get("end_date")),
        "time_confidence": str(row.get("time_confidence") or "medium"),
        "mm_verdict": str(row.get("mm_verdict") or ("needs_fix" if bool(row.get("needs_mm_review")) else "")),
        "mm_rationale": str(row.get("mm_rationale") or ""),
        "corrected_start_date": str(row.get("corrected_start_date") or ""),
        "corrected_end_date": str(row.get("corrected_end_date") or ""),
        "corrected_valid_from": str(row.get("corrected_valid_from") or ""),
        "corrected_valid_to": str(row.get("corrected_valid_to") or ""),
        "corrected_time_source": str(row.get("corrected_time_source") or ""),
        "correction_comment": str(row.get("correction_comment") or ""),
    }


def _normalize_assertions(rows: List[Dict[str, Any]], graph_kind: str) -> List[Dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for idx, row_dict in enumerate(rows, start=1):
        assertion_id = str(_pick_first(row_dict, ["assertion_id", "id"], f"{graph_kind}-{idx:05d}"))
        subject = str(_pick_first(row_dict, ["subject", "source", "src", "from"], ""))
        predicate = str(_pick_first(row_dict, ["predicate", "relation", "label", "type"], ""))
        object_ = str(_pick_first(row_dict, ["object", "target", "dst", "to"], ""))
        start_date = str(_pick_first(row_dict, ["start_date", "start", "begin", "year_start"], ""))
        end_date = str(_pick_first(row_dict, ["end_date", "end", "finish", "year_end"], ""))
        valid_from = str(_pick_first(row_dict, ["valid_from", "vf"], start_date))
        valid_to = str(_pick_first(row_dict, ["valid_to", "vt"], end_date or "+inf"))
        time_source = str(_pick_first(row_dict, ["time_source", "temporal_source"], ""))
        time_interval = str(_pick_first(row_dict, ["time_interval"], ""))
        papers_text = _parse_maybe_object(_pick_first(row_dict, ["papers"], []))
        if isinstance(papers_text, list):
            papers_text = ", ".join(str(x) for x in papers_text)
        evidence_raw = _pick_first(row_dict, ["evidence", "snippet", "quote", "summary"], "")
        evidence_text = _format_evidence_full(evidence_raw)
        evidence_short = _format_evidence_short(evidence_raw)
        evidence_obj = _parse_maybe_object(_pick_first(row_dict, ["evidence"], {}))
        locator = ""
        needs_mm_review = False
        if isinstance(evidence_obj, dict):
            locator = str(evidence_obj.get("figure_or_table") or evidence_obj.get("locator") or evidence_obj.get("caption") or "")
            needs_mm_review = bool(locator) or any(tok in evidence_text.lower() for tok in ("figure", "fig.", "table", "табл", "рис."))
        normalized_row = {
            "graph_kind": graph_kind,
            "assertion_id": assertion_id,
            "edge_uid": f"{graph_kind}:{assertion_id}",
            "subject": subject,
            "predicate": predicate,
            "object": object_,
            "subject_short": _truncate_text(subject, 90),
            "predicate_short": _truncate_text(predicate, 70),
            "object_short": _truncate_text(object_, 140),
            "start_date": start_date,
            "end_date": end_date,
            "valid_from": valid_from,
            "valid_to": valid_to,
            "time_source": time_source,
            "time_interval": time_interval,
            "evidence_text": evidence_text,
            "evidence_text_short": evidence_short,
            "evidence_payload_full": _format_evidence_full(evidence_obj if isinstance(evidence_obj, (dict, list)) else evidence_raw),
            "evidence_locator": locator,
            "papers_text": papers_text or "",
            "papers_text_short": _truncate_text(papers_text or "", 180),
            "paper_ids": _extract_paper_refs(_pick_first(row_dict, ["papers", "paper_ids", "paper_id"], []))[0],
            "paper_titles": _extract_paper_refs(_pick_first(row_dict, ["papers", "paper_titles", "title"], []))[1],
            "paper_source_refs": _extract_paper_refs(_pick_first(row_dict, ["papers", "source_refs", "source_ref", "url"], []))[2],
            "score": _pick_first(row_dict, ["score"], ""),
            "mean_confidence": _pick_first(row_dict, ["mean_confidence"], ""),
            "importance_score": _pick_first(row_dict, ["importance_score"], 0),
            "importance_model": str(_pick_first(row_dict, ["importance_model"], "")),
            "importance_reasons": _parse_maybe_object(_pick_first(row_dict, ["importance_reasons"], [])),
            "topic_overlap_tokens": _parse_maybe_object(_pick_first(row_dict, ["topic_overlap_tokens"], [])),
            "verdict": str(_pick_first(row_dict, ["verdict"], "")),
            "rationale": str(_pick_first(row_dict, ["rationale"], "")),
            "time_source_note": str(_pick_first(row_dict, ["time_source_note"], "")),
            "raw_record_json": json.dumps(row_dict, ensure_ascii=False, indent=2),
            "cutoff_year": str(_pick_first(row_dict, ["cutoff_year"], "")),
            "needs_mm_review": needs_mm_review,
            "time_granularity": _normalize_time_granularity(_pick_first(row_dict, ["time_granularity", "granularity"], "unknown"), start_date=start_date, end_date=end_date),
        }
        normalized_row["default_review_state"] = _default_review_state(normalized_row)
        normalized.append(normalized_row)
    return normalized


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__PAGE_TITLE__</title>
  <style>
    :root {
      --bg: #f8fafc;
      --card: #ffffff;
      --border: #d0d7de;
      --text: #111827;
      --muted: #475569;
      --accent: #0f766e;
      --accent-soft: #e6fffb;
      --warning: #b45309;
      --danger: #b91c1c;
      --shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background: var(--bg);
      line-height: 1.45;
    }
    .page { max-width: 1400px; margin: 0 auto; padding: 20px; }
    h1, h2, h3, h4 { margin: 0 0 8px 0; }
    .note, .card, details {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      box-shadow: var(--shadow);
    }
    .note { padding: 14px 16px; margin-bottom: 14px; }
    .note small, .muted { color: var(--muted); }
    .nav { display: flex; gap: 8px; flex-wrap: wrap; margin: 16px 0; }
    .nav button, .toolbar button, .toolbar label.buttonish {
      border: 1px solid var(--border);
      background: #fff;
      color: var(--text);
      border-radius: 999px;
      padding: 10px 14px;
      cursor: pointer;
      font-size: 14px;
    }
    .nav button.active, .toolbar button.primary {
      background: var(--accent);
      color: white;
      border-color: var(--accent);
    }
    .toolbar { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin: 10px 0 16px 0; }
    .toolbar input, .toolbar select, .toolbar textarea {
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 8px 10px;
      font: inherit;
      background: #fff;
      color: var(--text);
    }
    .toolbar input[type="search"] { min-width: 280px; }
    .section { display: none; }
    .section.active { display: block; }
    .grid { display: grid; gap: 12px; }
    .grid.cols-2 { grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }
    .grid.cols-3 { grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }
    .card { padding: 14px 16px; margin-bottom: 14px; }
    .card-title { font-size: 15px; font-weight: 700; margin-bottom: 6px; }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12px;
      background: var(--accent-soft);
      color: var(--accent);
      margin-right: 8px;
    }
    .stats { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }
    .stats .pill { background: #eef2ff; color: #3730a3; }
    .graph-shell { display: grid; grid-template-columns: minmax(0, 1.3fr) minmax(0, 1fr); gap: 16px; align-items: start; }
    .graph-shell > * { min-width: 0; }
    @media (max-width: 1100px) {
      .graph-shell { grid-template-columns: 1fr; }
    }
    svg.graph {
      width: 100%;
      min-height: 520px;
      background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
      border-radius: 12px;
      border: 1px solid var(--border);
    }
    .legend { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 8px; }
    .legend span { font-size: 12px; color: var(--muted); }
    .legend i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; vertical-align: middle; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; table-layout: fixed; }
    th, td { border-bottom: 1px solid #e5e7eb; padding: 8px; vertical-align: top; text-align: left; overflow-wrap: anywhere; word-break: break-word; }
    th { background: #f8fafc; position: sticky; top: 0; }
    .table-wrap { max-height: 460px; overflow: auto; border: 1px solid var(--border); border-radius: 12px; min-width: 0; }
    .card details { box-shadow: none; border-radius: 10px; margin-top: 8px; }
    .card details > div, .card details > pre { padding: 0 12px 12px 12px; }
    .card summary { cursor: pointer; padding: 10px 12px; font-weight: 600; }
    .field-grid { display: grid; gap: 10px; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); margin-top: 12px; }
    .field-grid label, .long-field label { display: flex; flex-direction: column; gap: 6px; font-size: 12px; color: var(--muted); }
    .long-field { margin-top: 10px; }
    textarea { min-height: 84px; resize: vertical; }
    code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    pre { white-space: pre-wrap; word-break: break-word; background: #f8fafc; padding: 12px; border-radius: 10px; overflow: auto; }
    .footer { margin-top: 18px; color: var(--muted); font-size: 12px; }
    .graph-label { pointer-events: none; }
    .hidden { display: none !important; }
  </style>
</head>
<body>
  <div class="page">
    <div class="note">
      <h1>Task 2 — автономная форма экспертной валидации</h1>
      <div class="muted">Файл работает локально и не требует ноутбука. Данные эталонного и авто-графа уже встроены в HTML. Черновик можно выгрузить в JSON, а готовые результаты — в ZIP.</div>
    </div>

    <div class="stats" id="meta-stats"></div>

    <div class="nav">
      <button class="active" data-section="graphs">Графы и assertions</button>
      <button data-section="validation">Валидация эксперта</button>
      <button data-section="summary">Сводка</button>
    </div>

    <section id="section-graphs" class="section active"></section>
    <section id="section-validation" class="section"></section>
    <section id="section-summary" class="section"></section>

    <div class="footer">Локальный autosave в браузере выполняется best-effort. Для надёжной передачи между устройствами используйте «Скачать draft JSON» или ZIP с результатами.</div>
  </div>

  <script>
  const APP = __APP_DATA__;

  const clone = (x) => JSON.parse(JSON.stringify(x));
  const el = (tag, attrs, ...children) => {
    const node = document.createElement(tag);
    if (attrs) {
      Object.entries(attrs).forEach(([key, value]) => {
        if (value === null || value === undefined) return;
        if (key === 'class') node.className = value;
        else if (key === 'html') node.innerHTML = value;
        else if (key === 'text') node.textContent = value;
        else if (key.startsWith('on') && typeof value === 'function') node.addEventListener(key.slice(2), value);
        else node.setAttribute(key, value);
      });
    }
    children.flat().forEach((child) => {
      if (child === null || child === undefined) return;
      if (typeof child === 'string') node.appendChild(document.createTextNode(child));
      else node.appendChild(child);
    });
    return node;
  };

  const STORAGE_KEY = `task2-offline-review:${APP.meta.submission_id || APP.meta.topic || 'bundle'}`;
  const APP_FILTER_DEFAULTS = APP.filter_defaults || {};
  const state = {
    reviewer_id: APP.meta.reviewer_default || '',
    filters: {
      graph: 'all',
      verdict: 'pending',
      search: '',
      pageSize: 5,
      page: 1,
      importanceThreshold: Number(APP_FILTER_DEFAULTS.importance_threshold || 0),
      exclusionText: APP_FILTER_DEFAULTS.exclusion_rules ? JSON.stringify(APP_FILTER_DEFAULTS.exclusion_rules, null, 2) : '',
    },
    reviewState: Object.fromEntries(APP.records.map((row) => [row.edge_uid, clone(row.default_review_state)])),
  };

  const graphColors = {
    step: '#4c78a8', paper: '#7f8c8d', term: '#72b7b2', time: '#e0ac2b', assertion: '#b279a2', default: '#9aa5b1'
  };

  function safeLocalGet(key) {
    try { return window.localStorage.getItem(key); } catch (_) { return null; }
  }
  function safeLocalSet(key, value) {
    try { window.localStorage.setItem(key, value); } catch (_) { /* noop */ }
  }
  function htmlEscape(value) {
    return String(value ?? '').replace(/[&<>\"']/g, (ch) => ({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}[ch]));
  }
  function truncate(value, limit = 160) {
    const text = String(value ?? '').replace(/\s+/g, ' ').trim();
    return text.length <= limit ? text : `${text.slice(0, Math.max(0, limit - 1)).trim()}…`;
  }
  function toNumber(value, fallback) {
    const n = Number(value);
    return Number.isFinite(n) ? n : fallback;
  }
  function normalizeTimeGranularity(value, startDate, endDate) {
    const raw = String(value ?? '').trim().toLowerCase();
    if (!raw || raw === 'unknown') {
      if (!startDate && !endDate) return 'unknown';
      return 'unknown';
    }
    if (['range', 'period', 'timespan', 'time_span', 'date_range'].includes(raw)) return 'interval';
    if (['year', 'month', 'day', 'interval'].includes(raw)) return raw;
    if (startDate && endDate && String(startDate) !== String(endDate)) return 'interval';
    return 'unknown';
  }
  function snapshot(reason = 'manual') {
    return {
      artifact_version: 1,
      reason,
      reviewer_id: state.reviewer_id,
      filters: clone(state.filters),
      review_state: clone(state.reviewState),
    };
  }
  function mergeLoaded(payload) {
    if (!payload || typeof payload !== 'object') return;
    if (payload.reviewer_id) state.reviewer_id = String(payload.reviewer_id);
    const filters = payload.filters || {};
    if (filters.graph) state.filters.graph = filters.graph;
    if (filters.verdict) state.filters.verdict = filters.verdict;
    if (typeof filters.search === 'string') state.filters.search = filters.search;
    state.filters.pageSize = toNumber(filters.page_size || filters.pageSize, state.filters.pageSize);
    state.filters.page = toNumber(filters.page, state.filters.page);
    state.filters.importanceThreshold = Math.max(0, Math.min(1, toNumber(filters.importance_threshold ?? filters.importanceThreshold, state.filters.importanceThreshold)));
    if (typeof filters.exclusion_text === 'string') state.filters.exclusionText = filters.exclusion_text;
    if (typeof filters.exclusionText === 'string') state.filters.exclusionText = filters.exclusionText;
    const loaded = payload.review_state || payload.reviewState || {};
    Object.entries(loaded).forEach(([edge, value]) => {
      if (!state.reviewState[edge] || !value || typeof value !== 'object') return;
      const merged = Object.assign({}, state.reviewState[edge], value);
      merged.time_granularity = normalizeTimeGranularity(merged.time_granularity, merged.corrected_start_date || '', merged.corrected_end_date || '');
      state.reviewState[edge] = merged;
    });
  }
  function autosave(reason = 'autosave') {
    safeLocalSet(STORAGE_KEY, JSON.stringify(snapshot(reason)));
  }
  function loadAutosave() {
    const raw = safeLocalGet(STORAGE_KEY);
    if (!raw) return false;
    try {
      mergeLoaded(JSON.parse(raw));
      return true;
    } catch (_) {
      return false;
    }
  }
  function downloadBytes(filename, mime, bytes) {
    const blob = bytes instanceof Blob ? bytes : new Blob([bytes], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function downloadText(filename, mime, text) {
    downloadBytes(filename, mime, new Blob([text], { type: mime }));
  }
  function toCsv(rows, columns) {
    const escape = (value) => {
      const text = value === null || value === undefined ? '' : String(value);
      if (/[",\n]/.test(text)) return '"' + text.replace(/"/g, '""') + '"';
      return text;
    };
    const lines = [columns.join(',')];
    rows.forEach((row) => lines.push(columns.map((col) => escape(row[col])).join(',')));
    return lines.join('\n');
  }
  const CRC_TABLE = (() => {
    const table = new Uint32Array(256);
    for (let n = 0; n < 256; n += 1) {
      let c = n;
      for (let k = 0; k < 8; k += 1) c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
      table[n] = c >>> 0;
    }
    return table;
  })();
  function crc32(bytes) {
    let c = 0xffffffff;
    for (let i = 0; i < bytes.length; i += 1) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
  }
  function u16(n) { return [n & 255, (n >>> 8) & 255]; }
  function u32(n) { return [n & 255, (n >>> 8) & 255, (n >>> 16) & 255, (n >>> 24) & 255]; }
  function zipStore(files) {
    const encoder = new TextEncoder();
    let offset = 0;
    const localParts = [];
    const centralParts = [];
    files.forEach((file) => {
      const nameBytes = encoder.encode(file.name);
      const dataBytes = file.data instanceof Uint8Array ? file.data : encoder.encode(file.data);
      const crc = crc32(dataBytes);
      const local = new Uint8Array(30 + nameBytes.length + dataBytes.length);
      let p = 0;
      local.set([0x50,0x4b,0x03,0x04], p); p += 4;
      local.set(u16(20), p); p += 2;
      local.set(u16(0), p); p += 2;
      local.set(u16(0), p); p += 2;
      local.set(u16(0), p); p += 2;
      local.set(u16(0), p); p += 2;
      local.set(u32(crc), p); p += 4;
      local.set(u32(dataBytes.length), p); p += 4;
      local.set(u32(dataBytes.length), p); p += 4;
      local.set(u16(nameBytes.length), p); p += 2;
      local.set(u16(0), p); p += 2;
      local.set(nameBytes, p); p += nameBytes.length;
      local.set(dataBytes, p);
      localParts.push(local);

      const central = new Uint8Array(46 + nameBytes.length);
      p = 0;
      central.set([0x50,0x4b,0x01,0x02], p); p += 4;
      central.set(u16(20), p); p += 2;
      central.set(u16(20), p); p += 2;
      central.set(u16(0), p); p += 2;
      central.set(u16(0), p); p += 2;
      central.set(u16(0), p); p += 2;
      central.set(u16(0), p); p += 2;
      central.set(u32(crc), p); p += 4;
      central.set(u32(dataBytes.length), p); p += 4;
      central.set(u32(dataBytes.length), p); p += 4;
      central.set(u16(nameBytes.length), p); p += 2;
      central.set(u16(0), p); p += 2;
      central.set(u16(0), p); p += 2;
      central.set(u16(0), p); p += 2;
      central.set(u16(0), p); p += 2;
      central.set(u32(0), p); p += 4;
      central.set(u32(offset), p); p += 4;
      central.set(nameBytes, p);
      centralParts.push(central);
      offset += local.length;
    });

    const centralStart = offset;
    const centralSize = centralParts.reduce((sum, part) => sum + part.length, 0);
    const end = new Uint8Array(22);
    let p = 0;
    end.set([0x50,0x4b,0x05,0x06], p); p += 4;
    end.set(u16(0), p); p += 2;
    end.set(u16(0), p); p += 2;
    end.set(u16(files.length), p); p += 2;
    end.set(u16(files.length), p); p += 2;
    end.set(u32(centralSize), p); p += 4;
    end.set(u32(centralStart), p); p += 4;
    end.set(u16(0), p);

    const total = localParts.reduce((sum, part) => sum + part.length, 0) + centralSize + end.length;
    const out = new Uint8Array(total);
    let cursor = 0;
    localParts.forEach((part) => { out.set(part, cursor); cursor += part.length; });
    centralParts.forEach((part) => { out.set(part, cursor); cursor += part.length; });
    out.set(end, cursor);
    return out;
  }

  function graphNodes(payload) {
    return Array.isArray(payload?.nodes) ? payload.nodes.filter((x) => x && typeof x === 'object') : [];
  }
  function graphEdges(payload) {
    return Array.isArray(payload?.edges) ? payload.edges.filter((x) => x && typeof x === 'object') : [];
  }
  function nodeGroup(node) {
    const rawType = String(node.type || node.node_type || '').toLowerCase();
    if (rawType.includes('trajectory') || rawType.includes('step')) return 'step';
    if (rawType.includes('paper') || node.paper_id || node.papers) return 'paper';
    if (rawType.includes('time') || node.valid_from || node.valid_to || node.yearly_doc_freq) return 'time';
    if (rawType.includes('assertion')) return 'assertion';
    if (node.term || ['term','entity','concept'].includes(rawType)) return 'term';
    return 'default';
  }
  function wrapGraphLabel(value, maxLine = 18, maxLines = 2) {
    const text = String(value ?? '').replace(/\s+/g, ' ').trim();
    if (!text) return [''];
    const words = text.split(' ');
    const lines = [];
    let current = '';
    words.forEach((word) => {
      if (!current) {
        current = word;
        return;
      }
      if ((current + ' ' + word).length <= maxLine) {
        current += ' ' + word;
      } else {
        lines.push(current);
        current = word;
      }
    });
    if (current) lines.push(current);
    if (lines.length <= maxLines) return lines;
    const clipped = lines.slice(0, maxLines);
    const last = clipped[maxLines - 1];
    clipped[maxLines - 1] = last.length > maxLine - 1 ? `${last.slice(0, Math.max(0, maxLine - 1)).trim()}…` : `${last}…`;
    return clipped;
  }

  function renderGraphSvg(target, payload) {
    const nodes = graphNodes(payload);
    const edges = graphEdges(payload);
    target.innerHTML = '';
    if (!nodes.length) {
      target.appendChild(el('div', { class: 'note muted', text: 'Граф пустой или не был построен.' }));
      return;
    }
    const width = 980;
    const height = 620;
    const cx = width / 2;
    const cy = height / 2;
    const radius = Math.max(180, Math.min(width, height) / 2 - 70);
    const pos = {};
    nodes.forEach((node, idx) => {
      const angle = (Math.PI * 2 * idx) / Math.max(1, nodes.length);
      const nodeId = String(node.id || node.term || node.label || `node-${idx}`);
      pos[nodeId] = {
        x: cx + radius * Math.cos(angle),
        y: cy + radius * Math.sin(angle),
      };
    });
    const svg = el('svg', { class: 'graph', viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': 'Graph visualization' });
    const defs = el('defs');
    const marker = el('marker', { id: 'arrowhead', markerWidth: '10', markerHeight: '7', refX: '9', refY: '3.5', orient: 'auto' },
      el('polygon', { points: '0 0, 10 3.5, 0 7', fill: '#94a3b8' })
    );
    defs.appendChild(marker);
    svg.appendChild(defs);
    edges.forEach((edge, idx) => {
      const src = pos[String(edge.source || '')];
      const tgt = pos[String(edge.target || edge.object || '')];
      if (!src || !tgt) return;
      const line = el('line', {
        x1: src.x, y1: src.y, x2: tgt.x, y2: tgt.y,
        stroke: '#94a3b8', 'stroke-width': '1.4', 'marker-end': 'url(#arrowhead)', opacity: '0.85'
      });
      line.appendChild(el('title', { text: String(edge.predicate || edge.label || 'related_to') }));
      svg.appendChild(line);
    });
    nodes.forEach((node, idx) => {
      const nodeId = String(node.id || node.term || node.label || `node-${idx}`);
      const { x, y } = pos[nodeId];
      const group = nodeGroup(node);
      const color = graphColors[group] || graphColors.default;
      const label = String(node.label || node.term || nodeId);
      const labelLines = wrapGraphLabel(label, 18, 2);
      const circle = el('circle', { cx: x, cy: y, r: 11, fill: color, opacity: '0.92' });
      circle.appendChild(el('title', { text: label }));
      svg.appendChild(circle);
      labelLines.forEach((lineText, lineIdx) => {
        svg.appendChild(el('text', {
          x,
          y: y + 26 + (lineIdx * 12),
          fill: '#111827',
          'font-size': '11',
          'text-anchor': 'middle',
          class: 'graph-label'
        }, lineText));
      });
    });
    target.appendChild(svg);
  }

  function normalizeTextList(value) {
    if (!value) return [];
    if (Array.isArray(value)) return value.map((item) => String(item ?? '').trim()).filter(Boolean);
    return String(value).split(/[\n,;]+/).map((item) => item.trim()).filter(Boolean);
  }

  function parseExclusionText(raw) {
    const text = String(raw || '').trim();
    if (!text) return {};
    try { return JSON.parse(text); } catch (_) { /* noop */ }
    const spec = { paper_ids: [], titles: [], source_refs: [], match_substrings: [], url_substrings: [] };
    const keyMap = {
      paper_ids: 'paper_ids', ids: 'paper_ids', articles: 'paper_ids',
      titles: 'titles', source_refs: 'source_refs', urls: 'source_refs',
      match_substrings: 'match_substrings', substrings: 'match_substrings',
      url_substrings: 'url_substrings'
    };
    let currentKey = '';
    text.split(/\r?\n/).forEach((line) => {
      const rawLine = String(line || '');
      const trimmed = rawLine.trim();
      if (!trimmed || trimmed.startsWith('#')) return;
      const keyMatch = trimmed.match(/^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$/);
      if (keyMatch) {
        const mapped = keyMap[keyMatch[1]] || keyMatch[1];
        currentKey = mapped;
        if (mapped === 'max_year') {
          const year = Number(keyMatch[2]);
          if (Number.isFinite(year)) spec.max_year = year;
          return;
        }
        if (!spec[mapped]) spec[mapped] = [];
        const rest = String(keyMatch[2] || '').trim();
        if (rest && rest !== '[]') {
          if (rest.startsWith('[') && rest.endsWith(']')) {
            rest.slice(1, -1).split(',').map((item) => item.trim().replace(/^['"]|['"]$/g, '')).filter(Boolean).forEach((item) => spec[mapped].push(item));
          } else {
            spec[mapped].push(rest.replace(/^['"]|['"]$/g, ''));
          }
        }
        return;
      }
      const itemMatch = rawLine.match(/^\s*-\s*(.+?)\s*$/);
      if (itemMatch && currentKey) {
        const val = itemMatch[1].trim().replace(/^['"]|['"]$/g, '');
        if (currentKey === 'max_year') {
          const year = Number(val);
          if (Number.isFinite(year)) spec.max_year = year;
        } else if (spec[currentKey]) {
          spec[currentKey].push(val);
        }
      }
    });
    return spec;
  }

  function exclusionMatchesRow(row, rawSpec) {
    const spec = rawSpec || {};
    const paperIds = normalizeTextList(row.paper_ids).concat(normalizeTextList(row.papers_text));
    const titles = normalizeTextList(row.paper_titles);
    const sourceRefs = normalizeTextList(row.paper_source_refs);
    const rawJson = String(row.raw_record_json || '').toLowerCase();
    const hay = [row.subject, row.predicate, row.object, row.evidence_text, row.papers_text, row.raw_record_json].join(' ').toLowerCase();
    const exactIn = (needles, values) => normalizeTextList(needles).some((needle) => values.some((value) => String(value).toLowerCase() === String(needle).toLowerCase()));
    const containsIn = (needles, values) => normalizeTextList(needles).some((needle) => values.some((value) => String(value).toLowerCase().includes(String(needle).toLowerCase())));
    if (exactIn(spec.paper_ids, paperIds) || containsIn(spec.paper_ids, paperIds)) return true;
    if (containsIn(spec.titles, titles) || normalizeTextList(spec.titles).some((needle) => hay.includes(String(needle).toLowerCase()))) return true;
    if (containsIn(spec.source_refs, sourceRefs) || normalizeTextList(spec.source_refs).some((needle) => rawJson.includes(String(needle).toLowerCase()))) return true;
    if (normalizeTextList(spec.match_substrings).some((needle) => hay.includes(String(needle).toLowerCase()))) return true;
    if (normalizeTextList(spec.url_substrings).some((needle) => rawJson.includes(String(needle).toLowerCase()))) return true;
    if (spec.max_year !== undefined && spec.max_year !== null && String(row.start_date || row.valid_from || '').slice(0, 4)) {
      const year = Number(String(row.start_date || row.valid_from || '').slice(0, 4));
      if (Number.isFinite(year) && year > Number(spec.max_year)) return true;
    }
    return false;
  }

  function filteredRecords() {
    const query = state.filters.search.trim().toLowerCase();
    const threshold = Math.max(0, Math.min(1, toNumber(state.filters.importanceThreshold, 0)));
    const exclusionSpec = parseExclusionText(state.filters.exclusionText);
    return APP.records.filter((row) => {
      if (state.filters.graph !== 'all' && row.graph_kind !== state.filters.graph) return false;
      const verdict = (state.reviewState[row.edge_uid] || {}).verdict || '';
      if (state.filters.verdict !== 'all') {
        if (state.filters.verdict === 'pending' && verdict) return false;
        if (state.filters.verdict !== 'pending' && verdict !== state.filters.verdict) return false;
      }
      const importance = Math.max(0, Math.min(1, toNumber(row.importance_score, 0)));
      if (importance < threshold) return false;
      if (exclusionMatchesRow(row, exclusionSpec)) return false;
      if (!query) return true;
      const hay = [row.subject, row.predicate, row.object, row.evidence_text, row.papers_text].join(' ').toLowerCase();
      return hay.includes(query);
    });
  }

  function buildExportFrames() {
    const now = new Date().toISOString();
    const reviewRows = [];
    const correctionRows = [];
    APP.records.forEach((row) => {
      const current = Object.assign({}, row, state.reviewState[row.edge_uid] || {});
      const merged = Object.assign({}, row, {
        reviewer_id: state.reviewer_id,
        review_timestamp: now,
        expert_verdict: current.verdict,
        expert_rationale: current.rationale,
        expert_time_source_note: current.time_source_note,
        corrected_start_date: current.corrected_start_date,
        corrected_end_date: current.corrected_end_date,
        corrected_valid_from: current.corrected_valid_from,
        corrected_valid_to: current.corrected_valid_to,
        corrected_time_source: current.corrected_time_source,
        correction_comment: current.correction_comment,
        semantic_correctness: current.semantic_correctness,
        evidence_sufficiency: current.evidence_sufficiency,
        scope_match: current.scope_match,
        system_match: current.system_match,
        environment_match: current.environment_match,
        protocol_match: current.protocol_match,
        scope_overgeneralized: current.scope_overgeneralized,
        corrected_scope_note: current.corrected_scope_note,
        hypothesis_role: current.hypothesis_role,
        hypothesis_relevance: current.hypothesis_relevance,
        testability_signal: current.testability_signal,
        causal_status: current.causal_status,
        severity: current.severity,
        evidence_before_cutoff: current.evidence_before_cutoff,
        leakage_risk: current.leakage_risk,
        time_type: current.time_type,
        time_granularity: current.time_granularity,
        time_confidence: current.time_confidence,
        mm_verdict: current.mm_verdict,
        mm_rationale: current.mm_rationale,
      });
      reviewRows.push(merged);
      if ([current.corrected_start_date, current.corrected_end_date, current.corrected_valid_from, current.corrected_valid_to, current.corrected_time_source, current.correction_comment].some(Boolean)) {
        correctionRows.push({
          edge_uid: row.edge_uid,
          graph_kind: row.graph_kind,
          assertion_id: row.assertion_id,
          subject: row.subject,
          predicate: row.predicate,
          object: row.object,
          original_start_date: row.start_date,
          original_end_date: row.end_date,
          original_valid_from: row.valid_from,
          original_valid_to: row.valid_to,
          corrected_start_date: current.corrected_start_date,
          corrected_end_date: current.corrected_end_date,
          corrected_valid_from: current.corrected_valid_from,
          corrected_valid_to: current.corrected_valid_to,
          corrected_time_source: current.corrected_time_source,
          comment: current.correction_comment,
          rationale: current.rationale,
          reviewer_id: state.reviewer_id,
        });
      }
    });
    const summary = {
      total_edges: reviewRows.length,
      decided_edges: reviewRows.filter((r) => String(r.expert_verdict || '').trim()).length,
      accepted_edges: reviewRows.filter((r) => r.expert_verdict === 'accepted').length,
      rejected_edges: reviewRows.filter((r) => r.expert_verdict === 'rejected').length,
      uncertain_edges: reviewRows.filter((r) => r.expert_verdict === 'uncertain').length,
      needs_time_fix_edges: reviewRows.filter((r) => r.expert_verdict === 'needs_time_fix').length,
      needs_evidence_fix_edges: reviewRows.filter((r) => r.expert_verdict === 'needs_evidence_fix').length,
      added_edges: reviewRows.filter((r) => r.expert_verdict === 'added').length,
      severity_violation_edges: reviewRows.filter((r) => r.severity === 'violation').length,
      high_hypothesis_relevance_edges: reviewRows.filter((r) => String(r.hypothesis_relevance) === '2').length,
      multimodal_needs_fix_edges: reviewRows.filter((r) => r.mm_verdict === 'needs_fix').length,
      temporal_corrections: correctionRows.length,
      active_filters: {
        importance_threshold: state.filters.importanceThreshold,
        exclusion_rules: parseExclusionText(state.filters.exclusionText),
      },
    };
    return { reviewRows, correctionRows, summary };
  }

  function renderMeta() {
    const host = document.getElementById('meta-stats');
    host.innerHTML = '';
    [
      ['topic', APP.meta.topic || '—'],
      ['submission_id', APP.meta.submission_id || '—'],
      ['cutoff_year', APP.meta.cutoff_year || '—'],
      ['bundle_dir', APP.meta.bundle_dir || '—'],
      ['rows', String(APP.records.length)],
      ['excluded_papers', String((APP.excluded_papers || []).length)],
    ].forEach(([label, value]) => host.appendChild(el('span', { class: 'pill', text: `${label}: ${value}` })));
  }

  function renderGraphs() {
    const host = document.getElementById('section-graphs');
    host.innerHTML = '';
    ['gold', 'auto'].forEach((kind) => {
      const graph = APP.graphs[kind];
      const rows = APP.records.filter((row) => row.graph_kind === kind);
      if (!graph && !rows.length) return;
      const title = kind === 'gold' ? 'Эталонный граф' : 'Авто-граф';
      const section = el('div', { class: 'card' });
      section.appendChild(el('h2', { text: title }));
      section.appendChild(el('div', { class: 'muted', text: `Assertions: ${rows.length} · nodes: ${graphNodes(graph || {}).length} · edges: ${graphEdges(graph || {}).length}` }));
      const legend = el('div', { class: 'legend' });
      Object.entries(graphColors).forEach(([name, color]) => legend.appendChild(el('span', { html: `<i style="background:${color}"></i>${htmlEscape(name)}` })));
      section.appendChild(legend);
      const shell = el('div', { class: 'graph-shell' });
      const graphCard = el('div');
      const graphMount = el('div');
      renderGraphSvg(graphMount, graph || {});
      graphCard.appendChild(graphMount);
      shell.appendChild(graphCard);

      const tableCard = el('div');
      tableCard.appendChild(el('div', { class: 'muted', text: 'Ниже — данные assertions, встроенные в офлайн-форму.' }));
      const tableWrap = el('div', { class: 'table-wrap' });
      const table = el('table');
      const thead = el('thead');
      thead.appendChild(el('tr', null,
        el('th', { text: 'Assertion' }),
        el('th', { text: 'Временные поля' }),
        el('th', { text: 'Evidence' })
      ));
      table.appendChild(thead);
      const tbody = el('tbody');
      rows.forEach((row) => {
        tbody.appendChild(el('tr', null,
          el('td', { html: `<b>${htmlEscape(truncate(row.subject, 60))}</b><br><span class="muted">${htmlEscape(truncate(row.predicate, 40))} → ${htmlEscape(truncate(row.object, 90))}</span>` }),
          el('td', { html: `<b>start:</b> ${htmlEscape(row.start_date || '—')}<br><b>end:</b> ${htmlEscape(row.end_date || '—')}<br><b>valid:</b> ${htmlEscape(row.valid_from || '—')} → ${htmlEscape(row.valid_to || '—')}<br><b>source:</b> ${htmlEscape(row.time_source || '—')}` }),
          el('td', { html: `${htmlEscape(truncate(row.evidence_text_short || row.evidence_text, 180))}<br><span class="muted">papers: ${htmlEscape(truncate(row.papers_text, 90) || '—')}</span>` })
        ));
      });
      table.appendChild(tbody);
      tableWrap.appendChild(table);
      tableCard.appendChild(tableWrap);
      tableCard.appendChild(el('details', null,
        el('summary', { text: 'Показать raw graph JSON' }),
        el('pre', { text: JSON.stringify(graph || {}, null, 2) })
      ));
      shell.appendChild(tableCard);
      section.appendChild(shell);
      host.appendChild(section);
    });
  }

  function renderValidation() {
    const host = document.getElementById('section-validation');
    host.innerHTML = '';
    const toolbar = el('div', { class: 'toolbar' });
    const reviewer = el('input', { type: 'text', value: state.reviewer_id, placeholder: 'Reviewer id / имя эксперта' });
    reviewer.addEventListener('input', (e) => { state.reviewer_id = e.target.value; autosave(); renderValidation(); renderSummary(); });
    const graphSelect = el('select');
    [['all','Все графы'], ['gold','Эталонный'], ['auto','Авто']].forEach(([value, label]) => graphSelect.appendChild(el('option', { value, text: label })));
    graphSelect.value = state.filters.graph;
    graphSelect.addEventListener('change', (e) => { state.filters.graph = e.target.value; state.filters.page = 1; autosave(); renderValidation(); });
    const verdictSelect = el('select');
    [['pending','Без решения'], ['all','Все'], ['accepted','accepted'], ['rejected','rejected'], ['uncertain','uncertain'], ['needs_time_fix','needs_time_fix'], ['needs_evidence_fix','needs_evidence_fix'], ['added','added']].forEach(([value, label]) => verdictSelect.appendChild(el('option', { value, text: label })));
    verdictSelect.value = state.filters.verdict;
    verdictSelect.addEventListener('change', (e) => { state.filters.verdict = e.target.value; state.filters.page = 1; autosave(); renderValidation(); });
    const search = el('input', { type: 'search', value: state.filters.search, placeholder: 'поиск по subject / predicate / object / evidence' });
    search.addEventListener('input', (e) => { state.filters.search = e.target.value; state.filters.page = 1; autosave(); renderValidation(); });
    const pageSize = el('select');
    [5, 10, 20, 50].forEach((value) => pageSize.appendChild(el('option', { value: String(value), text: `${value} на странице` })));
    pageSize.value = String(state.filters.pageSize);
    pageSize.addEventListener('change', (e) => { state.filters.pageSize = toNumber(e.target.value, 5); state.filters.page = 1; autosave(); renderValidation(); });

    const importanceThreshold = el('input', { type: 'number', min: '0', max: '1', step: '0.05', value: String(state.filters.importanceThreshold ?? 0), title: 'Минимальная важность триплета' });
    importanceThreshold.addEventListener('input', (e) => { state.filters.importanceThreshold = Math.max(0, Math.min(1, toNumber(e.target.value, 0))); state.filters.page = 1; autosave(); renderValidation(); });
    const exclusionInput = el('textarea', { rows: '5', placeholder: `paper_ids:
  - PMID:12345
match_substrings:
  - review after discovery`, style: 'min-width:360px;min-height:96px;' });
    exclusionInput.value = state.filters.exclusionText || '';
    exclusionInput.addEventListener('input', (e) => { state.filters.exclusionText = e.target.value; state.filters.page = 1; autosave(); renderValidation(); });
    const exclusionUpload = el('input', { type: 'file', class: 'hidden', accept: '.yaml,.yml,.json,text/plain' });
    exclusionUpload.addEventListener('change', async (e) => {
      const file = e.target.files && e.target.files[0];
      if (!file) return;
      state.filters.exclusionText = await file.text();
      autosave('exclusion-import');
      renderValidation();
    });
    const uploadExclusionBtn = el('button', { text: 'Загрузить YAML исключений' });
    uploadExclusionBtn.addEventListener('click', () => exclusionUpload.click());
    const clearExclusionBtn = el('button', { text: 'Очистить исключения' });
    clearExclusionBtn.addEventListener('click', () => { state.filters.exclusionText = ''; autosave('exclusion-clear'); renderValidation(); });

    const saveDraft = el('button', { text: 'Скачать draft JSON' });
    saveDraft.addEventListener('click', () => downloadText('task2_review_draft.json', 'application/json;charset=utf-8', JSON.stringify(snapshot('manual'), null, 2)));
    const importInput = el('input', { type: 'file', class: 'hidden', accept: '.json,application/json' });
    importInput.addEventListener('change', async (e) => {
      const file = e.target.files && e.target.files[0];
      if (!file) return;
      const raw = await file.text();
      mergeLoaded(JSON.parse(raw));
      autosave('import');
      renderValidation();
      renderSummary();
    });
    const importBtn = el('button', { text: 'Загрузить draft JSON' });
    importBtn.addEventListener('click', () => importInput.click());
    const exportZip = el('button', { class: 'primary', text: 'Скачать результаты ZIP' });
    exportZip.addEventListener('click', () => {
      const { reviewRows, correctionRows, summary } = buildExportFrames();
      const reviewColumns = [
        'graph_kind','assertion_id','edge_uid','subject','predicate','object','start_date','end_date','valid_from','valid_to','time_source','time_interval',
        'score','mean_confidence','reviewer_id','review_timestamp','expert_verdict','expert_rationale','expert_time_source_note','semantic_correctness',
        'evidence_sufficiency','scope_match','system_match','environment_match','protocol_match','scope_overgeneralized','corrected_scope_note',
        'hypothesis_role','hypothesis_relevance','testability_signal','causal_status','severity','evidence_before_cutoff','leakage_risk','time_type',
        'time_granularity','time_confidence','mm_verdict','mm_rationale','corrected_start_date','corrected_end_date','corrected_valid_from','corrected_valid_to',
        'corrected_time_source','correction_comment','papers_text','evidence_text'
      ];
      const correctionColumns = [
        'edge_uid','graph_kind','assertion_id','subject','predicate','object','original_start_date','original_end_date','original_valid_from','original_valid_to',
        'corrected_start_date','corrected_end_date','corrected_valid_from','corrected_valid_to','corrected_time_source','comment','rationale','reviewer_id'
      ];
      const reviewPayload = {
        artifact_version: 5,
        domain: APP.meta.domain || '',
        topic: APP.meta.topic || '',
        trajectory_submission_id: APP.meta.submission_id || '',
        cutoff_year: APP.meta.cutoff_year || '',
        reviewer_id: state.reviewer_id,
        timestamp: new Date().toISOString(),
        filter_settings: {
          importance_threshold: state.filters.importanceThreshold,
          exclusion_rules: parseExclusionText(state.filters.exclusionText),
        },
        assertions: reviewRows,
        added_edges: reviewRows.filter((row) => row.expert_verdict === 'added'),
      };
      const correctionPayload = {
        artifact_version: 3,
        domain: APP.meta.domain || '',
        paper_id: '',
        reviewer_id: state.reviewer_id,
        trajectory_submission_id: APP.meta.submission_id || '',
        corrections: correctionRows,
      };
      const files = [
        { name: 'edge_reviews.csv', data: toCsv(reviewRows, reviewColumns) },
        { name: 'edge_reviews.json', data: JSON.stringify(reviewPayload, null, 2) },
        { name: 'temporal_corrections.csv', data: toCsv(correctionRows, correctionColumns) },
        { name: 'temporal_corrections.json', data: JSON.stringify(correctionPayload, null, 2) },
        { name: 'validation_summary.json', data: JSON.stringify(summary, null, 2) },
        { name: 'review_state_latest.json', data: JSON.stringify(snapshot('export'), null, 2) },
      ];
      downloadBytes('expert_validation_bundle.zip', 'application/zip', zipStore(files));
    });

    toolbar.append(
      el('label', { class: 'muted', text: 'Reviewer' }), reviewer,
      graphSelect, verdictSelect, search, pageSize,
      el('label', { class: 'muted', text: 'Порог важности' }), importanceThreshold,
      saveDraft, importBtn, importInput, exportZip
    );
    host.appendChild(toolbar);

    const extraFilters = el('details', { open: false },
      el('summary', { text: 'Дополнительные фильтры: исключения из YAML / JSON' }),
      el('div', { class: 'task2-note task2-ui' },
        el('div', { class: 'task2-small', text: 'Исключите статьи и триплеты по paper_ids / titles / source_refs / match_substrings, чтобы убрать утечку знаний из будущего в прошлое.' }),
        exclusionInput,
        el('div', { class: 'toolbar' }, uploadExclusionBtn, clearExclusionBtn, exclusionUpload)
      )
    );
    host.appendChild(extraFilters);

    const rows = filteredRecords();
    const totalPages = Math.max(1, Math.ceil(rows.length / state.filters.pageSize));
    if (state.filters.page > totalPages) state.filters.page = totalPages;
    const start = (state.filters.page - 1) * state.filters.pageSize;
    const visible = rows.slice(start, start + state.filters.pageSize);
    const summaryRow = el('div', { class: 'stats' });
    const decided = APP.records.filter((row) => String((state.reviewState[row.edge_uid] || {}).verdict || '').trim()).length;
    const corrected = APP.records.filter((row) => {
      const s = state.reviewState[row.edge_uid] || {};
      return [s.corrected_start_date, s.corrected_end_date, s.corrected_valid_from, s.corrected_valid_to, s.corrected_time_source, s.correction_comment].some(Boolean);
    }).length;
    [
      `всего: ${APP.records.length}`,
      `с оценкой: ${decided}`,
      `без оценки: ${APP.records.length - decided}`,
      `с правками: ${corrected}`,
      `важность ≥ ${Number(state.filters.importanceThreshold || 0).toFixed(2)}`,
      `фильтровано: ${rows.length}`,
      `страница: ${state.filters.page}/${totalPages}`,
    ].forEach((text) => summaryRow.appendChild(el('span', { class: 'pill', text })));
    host.appendChild(summaryRow);

    const pager = el('div', { class: 'toolbar' });
    const prev = el('button', { text: '← Назад' });
    prev.disabled = state.filters.page <= 1;
    prev.addEventListener('click', () => { state.filters.page -= 1; autosave(); renderValidation(); });
    const next = el('button', { text: 'Вперёд →' });
    next.disabled = state.filters.page >= totalPages;
    next.addEventListener('click', () => { state.filters.page += 1; autosave(); renderValidation(); });
    pager.append(prev, el('span', { class: 'muted', text: `Показаны ${visible.length} записей` }), next);
    host.appendChild(pager);

    if (!visible.length) {
      host.appendChild(el('div', { class: 'note muted', text: 'Нет строк под выбранный фильтр.' }));
      return;
    }

    visible.forEach((row) => {
      const review = state.reviewState[row.edge_uid] || {};
      const card = el('div', { class: 'card' });
      card.appendChild(el('div', { class: 'card-title', text: `[${row.graph_kind}] ${truncate(row.subject, 80)} — ${truncate(row.predicate, 50)} → ${truncate(row.object, 120)}` }));
      const meta = el('div', { class: 'muted', html: `<b>assertion_id:</b> ${htmlEscape(row.assertion_id)} · <b>time:</b> ${htmlEscape(row.start_date || '—')} → ${htmlEscape(row.end_date || '—')} · <b>valid:</b> ${htmlEscape(row.valid_from || '—')} → ${htmlEscape(row.valid_to || '—')} · <b>papers:</b> ${htmlEscape(truncate(row.papers_text, 80) || '—')}` });
      card.appendChild(meta);
      card.appendChild(el('details', null,
        el('summary', { text: 'Полный текст и provenance' }),
        el('pre', { text: [
          `subject: ${row.subject || ''}`,
          `predicate: ${row.predicate || ''}`,
          `object: ${row.object || ''}`,
          `evidence_text:\n${row.evidence_text || ''}`,
          `papers: ${row.papers_text || ''}`,
          row.evidence_payload_full ? `evidence_payload:\n${row.evidence_payload_full}` : '',
          row.raw_record_json ? `raw_record_json:\n${row.raw_record_json}` : '',
        ].filter(Boolean).join('\n\n') })
      ));

      function field(label, key, type, options) {
        const wrapper = el('label');
        wrapper.appendChild(el('span', { text: label }));
        let input;
        if (type === 'select') {
          input = el('select', { 'data-edge': row.edge_uid, 'data-key': key });
          options.forEach(([value, text]) => input.appendChild(el('option', { value, text })));
          input.value = String(review[key] ?? '');
        } else if (type === 'checkbox') {
          input = el('input', { type: 'checkbox', 'data-edge': row.edge_uid, 'data-key': key });
          input.checked = Boolean(review[key]);
        } else if (type === 'textarea') {
          input = el('textarea', { 'data-edge': row.edge_uid, 'data-key': key, text: String(review[key] ?? '') });
          input.value = String(review[key] ?? '');
        } else {
          input = el('input', { type: 'text', value: String(review[key] ?? ''), 'data-edge': row.edge_uid, 'data-key': key });
        }
        wrapper.appendChild(input);
        return wrapper;
      }

      const grid1 = el('div', { class: 'field-grid' },
        field('Verdict', 'verdict', 'select', [['','—'], ['accepted','accepted'], ['rejected','rejected'], ['uncertain','uncertain'], ['needs_time_fix','needs_time_fix'], ['needs_evidence_fix','needs_evidence_fix'], ['added','added']]),
        field('Semantic correctness', 'semantic_correctness', 'select', [['','—'], ['correct','correct'], ['partial','partial'], ['incorrect','incorrect']]),
        field('Evidence sufficiency', 'evidence_sufficiency', 'select', [['','—'], ['sufficient','sufficient'], ['partial','partial'], ['insufficient','insufficient']]),
        field('Scope match', 'scope_match', 'select', [['','—'], ['match','match'], ['partial','partial'], ['mismatch','mismatch']]),
        field('System match', 'system_match', 'select', [['','—'], ['match','match'], ['partial','partial'], ['mismatch','mismatch']]),
        field('Environment match', 'environment_match', 'select', [['','—'], ['match','match'], ['partial','partial'], ['mismatch','mismatch']]),
        field('Protocol match', 'protocol_match', 'select', [['','—'], ['match','match'], ['partial','partial'], ['mismatch','mismatch']]),
        field('Scope overgeneralized', 'scope_overgeneralized', 'checkbox'),
        field('Hypothesis role', 'hypothesis_role', 'select', [['background','background'],['mechanism','mechanism'],['intervention','intervention'],['measurement','measurement'],['boundary_condition','boundary_condition'],['contradiction','contradiction']]),
        field('Hypothesis relevance', 'hypothesis_relevance', 'select', [['0','0'],['1','1'],['2','2']]),
        field('Testability signal', 'testability_signal', 'select', [['0','0'],['1','1'],['2','2']]),
        field('Causal status', 'causal_status', 'select', [['descriptive','descriptive'],['correlational','correlational'],['causal','causal'],['theoretical','theoretical']]),
        field('Severity', 'severity', 'select', [['warning','warning'],['violation','violation']]),
        field('Evidence before cutoff', 'evidence_before_cutoff', 'select', [['','—'],['yes','yes'],['no','no'],['unclear','unclear']]),
        field('Leakage risk', 'leakage_risk', 'select', [['possible','possible'],['low','low'],['high','high']]),
        field('Time type', 'time_type', 'select', [['observation_period','observation_period'],['publication_time','publication_time']]),
        field('Time granularity', 'time_granularity', 'select', [['unknown','unknown'],['year','year'],['month','month'],['day','day'],['interval','interval']]),
        field('Time confidence', 'time_confidence', 'select', [['low','low'],['medium','medium'],['high','high']]),
        field('MM verdict', 'mm_verdict', 'select', [['','—'],['ok','ok'],['needs_fix','needs_fix'],['not_applicable','not_applicable']])
      );
      const grid2 = el('div', { class: 'field-grid' },
        field('Corrected start date', 'corrected_start_date', 'text'),
        field('Corrected end date', 'corrected_end_date', 'text'),
        field('Corrected valid_from', 'corrected_valid_from', 'text'),
        field('Corrected valid_to', 'corrected_valid_to', 'text'),
        field('Corrected time source', 'corrected_time_source', 'text'),
        field('Time source note', 'time_source_note', 'text')
      );
      const long1 = el('div', { class: 'long-field' }, field('Rationale', 'rationale', 'textarea'));
      const long2 = el('div', { class: 'long-field' }, field('MM rationale', 'mm_rationale', 'textarea'));
      const long3 = el('div', { class: 'long-field' }, field('Corrected scope note', 'corrected_scope_note', 'textarea'));
      const long4 = el('div', { class: 'long-field' }, field('Correction comment', 'correction_comment', 'textarea'));
      card.append(grid1, grid2, long1, long2, long3, long4);
      host.appendChild(card);
    });

    host.querySelectorAll('[data-edge][data-key]').forEach((input) => {
      const handler = (event) => {
        const edge = event.target.getAttribute('data-edge');
        const key = event.target.getAttribute('data-key');
        if (!edge || !key || !state.reviewState[edge]) return;
        state.reviewState[edge][key] = event.target.type === 'checkbox' ? event.target.checked : event.target.value;
        autosave();
        renderSummary();
      };
      input.addEventListener(input.tagName === 'SELECT' ? 'change' : 'input', handler);
      if (input.type === 'checkbox') input.addEventListener('change', handler);
    });
  }

  function renderSummary() {
    const host = document.getElementById('section-summary');
    host.innerHTML = '';
    const { reviewRows, correctionRows, summary } = buildExportFrames();
    const top = el('div', { class: 'card' });
    top.appendChild(el('h2', { text: 'Сводка по текущему состоянию' }));
    const stats = el('div', { class: 'stats' });
    Object.entries(summary).forEach(([key, value]) => stats.appendChild(el('span', { class: 'pill', text: `${key}: ${value}` })));
    top.appendChild(stats);
    top.appendChild(el('details', null, el('summary', { text: 'Показать validation_summary.json' }), el('pre', { text: JSON.stringify(summary, null, 2) })));
    host.appendChild(top);

    if (APP.comparison_summary) {
      host.appendChild(el('div', { class: 'card' },
        el('h3', { text: 'comparison_summary.json' }),
        el('pre', { text: JSON.stringify(APP.comparison_summary, null, 2) })
      ));
    }

    host.appendChild(el('div', { class: 'card' },
      el('h3', { text: 'Черновик, который будет выгружен' }),
      el('pre', { text: JSON.stringify(snapshot('preview'), null, 2) })
    ));

    host.appendChild(el('div', { class: 'card' },
      el('h3', { text: 'Превью edge_reviews.json (первые 3 записи)' }),
      el('pre', { text: JSON.stringify(reviewRows.slice(0, 3), null, 2) })
    ));
    host.appendChild(el('div', { class: 'card' },
      el('h3', { text: 'Превью temporal_corrections.json (первые 3 записи)' }),
      el('pre', { text: JSON.stringify(correctionRows.slice(0, 3), null, 2) })
    ));
  }

  function bindNav() {
    document.querySelectorAll('.nav button[data-section]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const section = btn.getAttribute('data-section');
        document.querySelectorAll('.nav button').forEach((node) => node.classList.remove('active'));
        btn.classList.add('active');
        document.querySelectorAll('.section').forEach((node) => node.classList.remove('active'));
        document.getElementById(`section-${section}`).classList.add('active');
      });
    });
  }

  loadAutosave();
  renderMeta();
  bindNav();
  renderGraphs();
  renderValidation();
  renderSummary();
  </script>
</body>
</html>
"""


def build_task2_offline_review_package(
    manifest: Dict[str, Any],
    task1_doc: Dict[str, Any],
    *,
    output_path: str | Path | None = None,
) -> Path:
    bundle_dir = Path(manifest["bundle_dir"])
    output = Path(output_path) if output_path else bundle_dir / "expert_validation" / "offline_review" / "task2_expert_validation_offline.html"
    output.parent.mkdir(parents=True, exist_ok=True)

    gold_triplets_path = _resolve_bundle_artifact(
        manifest,
        bundle_dir,
        "gold_triplets_csv",
        "reference_triplets",
        default_rel="reference_triplets.csv",
    )
    auto_triplets_path = _resolve_bundle_artifact(
        manifest,
        bundle_dir,
        "auto_triplets_csv",
        "automatic_triplets",
        default_rel="automatic_triplets.csv",
    )
    gold_graph_path = _resolve_bundle_artifact(
        manifest,
        bundle_dir,
        "gold_graph",
        "reference_graph",
        default_rel="reference_graph.json",
    )
    auto_graph_path = _resolve_bundle_artifact(
        manifest,
        bundle_dir,
        "auto_graph_json",
        "automatic_graph",
        default_rel="automatic_graph/temporal_kg.json",
    )
    gold_graph_html_path = _resolve_bundle_artifact(
        manifest,
        bundle_dir,
        "gold_graph_html",
        default_rel="reference_graph.html",
    )
    auto_graph_html_path = _resolve_bundle_artifact(
        manifest,
        bundle_dir,
        "auto_graph_html",
        default_rel="automatic_graph.html",
    )
    gold_graph_analytics_path = _resolve_bundle_artifact(
        manifest,
        bundle_dir,
        "gold_graph_analytics",
        default_rel="reference_graph_analytics.json",
    )
    auto_graph_analytics_path = _resolve_bundle_artifact(
        manifest,
        bundle_dir,
        "auto_graph_analytics",
        default_rel="automatic_graph_analytics.json",
    )
    comparison_summary_path = _resolve_bundle_artifact(
        manifest,
        bundle_dir,
        "comparison_summary",
        default_rel="comparison_summary.json",
    )

    records: list[dict[str, Any]] = []
    if gold_triplets_path is not None:
        records.extend(_normalize_assertions(_read_rows(gold_triplets_path), "gold"))
    if auto_triplets_path is not None:
        records.extend(_normalize_assertions(_read_rows(auto_triplets_path), "auto"))

    app_data = {
        "meta": {
            "topic": str(task1_doc.get("topic") or manifest.get("topic") or ""),
            "submission_id": str(task1_doc.get("submission_id") or ""),
            "cutoff_year": str(task1_doc.get("cutoff_year") or ""),
            "domain": str(task1_doc.get("domain") or ""),
            "bundle_dir": str(bundle_dir),
            "reviewer_default": str((task1_doc.get("expert") or {}).get("latin_slug") or (task1_doc.get("expert") or {}).get("full_name") or "") if isinstance(task1_doc.get("expert"), dict) else "",
        },
        "filter_defaults": manifest.get("filter_defaults") or {"importance_threshold": 0.0, "exclusion_rules": {}},
        "excluded_papers": _safe_json_load(Path(bundle_dir / "excluded_papers.json")) if (bundle_dir / "excluded_papers.json").exists() else [],
        "graph_analytics": {
            "gold": _safe_json_load(gold_graph_analytics_path) if gold_graph_analytics_path is not None else {},
            "auto": _safe_json_load(auto_graph_analytics_path) if auto_graph_analytics_path is not None else {},
        },
        "graph_html_paths": {
            "gold": str(gold_graph_html_path or ""),
            "auto": str(auto_graph_html_path or ""),
        },
        "records": records,
        "graphs": {
            "gold": _safe_json_load(gold_graph_path) if gold_graph_path is not None else {},
            "auto": _safe_json_load(auto_graph_path) if auto_graph_path is not None else {},
        },
        "comparison_summary": _safe_json_load(comparison_summary_path) if comparison_summary_path is not None else None,
    }

    app_json = json.dumps(app_data, ensure_ascii=False).replace("</", "<\\/")
    page_title = html.escape(f"Task 2 offline review — {app_data['meta']['submission_id'] or app_data['meta']['topic'] or 'bundle'}")
    html_text = _HTML_TEMPLATE.replace("__APP_DATA__", app_json).replace("__PAGE_TITLE__", page_title)
    output.write_text(html_text, encoding="utf-8")
    return output
