from __future__ import annotations

import hashlib
import html
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ARTIFACT_VERSION = 1


def _safe_json_load(path: Path | None) -> Any:
    if path is None or not Path(path).exists():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _resolve_bundle_artifact(manifest: Dict[str, Any], bundle_dir: Path, *keys: str) -> Path | None:
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    for key in keys:
        value = artifacts.get(key) if isinstance(artifacts, dict) else None
        if not value:
            value = manifest.get(key)
        if not value:
            continue
        p = Path(str(value))
        if not p.is_absolute():
            p = bundle_dir / p
        if p.exists():
            return p
    return None


def _truncate(value: Any, limit: int = 200) -> str:
    text = str(value or "").replace("\n", " ").strip()
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _coerce_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _supporting_evidence(row: Dict[str, Any], limit: int = 6) -> List[Dict[str, Any]]:
    hyp = row.get("hypothesis") if isinstance(row.get("hypothesis"), dict) else {}
    evidence = hyp.get("supporting_evidence") if isinstance(hyp.get("supporting_evidence"), list) else []
    out: list[dict[str, Any]] = []
    for item in evidence[:limit]:
        if isinstance(item, dict):
            out.append(
                {
                    "source_id": str(item.get("source_id") or item.get("paper_id") or item.get("id") or ""),
                    "text_snippet": _truncate(item.get("text_snippet") or item.get("snippet") or item.get("summary") or item),
                    "page": str(item.get("page") or ""),
                    "locator": str(item.get("locator") or item.get("figure_or_table") or item.get("caption") or ""),
                }
            )
        else:
            out.append({"source_id": "", "text_snippet": _truncate(item), "page": "", "locator": ""})
    if out:
        return out

    for item in _coerce_list(row.get("multimodal_support"))[:limit]:
        if not isinstance(item, dict):
            out.append({"source_id": "", "text_snippet": _truncate(item), "page": "", "locator": ""})
            continue
        out.append(
            {
                "source_id": str(item.get("chunk_id") or item.get("paper_id") or item.get("source_id") or ""),
                "text_snippet": _truncate(item.get("snippet") or item.get("evidence") or item.get("caption") or item),
                "page": str(item.get("page") or ""),
                "locator": str(item.get("locator") or item.get("modality") or ""),
            }
        )
    return out


def _temporal_summary(row: Dict[str, Any]) -> str:
    tc = row.get("temporal_context") if isinstance(row.get("temporal_context"), dict) else {}
    ordering = str(tc.get("ordering") or "predicted_or_missing")
    years = tc.get("years") if isinstance(tc.get("years"), list) else []
    window = tc.get("time_scope") or tc.get("interval") or tc.get("range") or ""
    parts = [f"ordering={ordering}"]
    if years:
        parts.append("years=" + ", ".join(str(x) for x in years[:8]))
    if window:
        parts.append(f"scope={window}")
    prediction = row.get("prediction_support") if isinstance(row.get("prediction_support"), dict) else {}
    if prediction:
        score = prediction.get("score")
        backend = prediction.get("backend")
        if score is not None:
            try:
                parts.append(f"predicted_link={float(score):.3f}")
            except Exception:
                parts.append(f"predicted_link={score}")
        if backend:
            parts.append(f"backend={backend}")
    return "; ".join(parts)


def _graph_signal_summary(row: Dict[str, Any]) -> str:
    cand = row.get("candidate") if isinstance(row.get("candidate"), dict) else {}
    graph_signals = cand.get("graph_signals") if isinstance(cand.get("graph_signals"), dict) else {}
    if not graph_signals:
        return ""
    parts: list[str] = []
    for key in ("support", "novelty", "consistency", "temporal_support", "centrality"):
        if key in graph_signals:
            try:
                parts.append(f"{key}={float(graph_signals[key]):.3f}")
            except Exception:
                parts.append(f"{key}={graph_signals[key]}")
    return "; ".join(parts)


def _normalize_full_variant(row: Dict[str, Any]) -> Dict[str, Any]:
    hyp = row.get("hypothesis") if isinstance(row.get("hypothesis"), dict) else {}
    title = str(hyp.get("title") or "Generated Task 3 hypothesis")
    return {
        "system_id": "task3_full",
        "system_title": "Task 3 full model",
        "title": title,
        "premise": str(hyp.get("premise") or ""),
        "mechanism": str(hyp.get("mechanism") or ""),
        "time_scope": str(hyp.get("time_scope") or _temporal_summary(row)),
        "proposed_experiment": str(hyp.get("proposed_experiment") or ""),
        "supporting_evidence": _supporting_evidence(row),
        "score": float(row.get("final_score") or 0.0),
        "score_label": "final_score",
    }


def _baseline_variant_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    cand = row.get("candidate") if isinstance(row.get("candidate"), dict) else {}
    source = str(cand.get("source") or "source")
    predicate = str(cand.get("predicate") or "related_to")
    target = str(cand.get("target") or "target")
    time_scope = str(cand.get("time_scope") or _temporal_summary(row))
    graph_signal_summary = _graph_signal_summary(row)
    premise = f"Observed candidate relation: {source} {predicate} {target}."
    if graph_signal_summary:
        premise += f" Graph cues: {graph_signal_summary}."
    mechanism = (
        f"Temporal graph evidence suggests the relation may evolve over time; { _temporal_summary(row) }. "
        f"This baseline keeps the claim close to the extracted candidate without extra generative expansion."
    )
    experiment = (
        f"Collect an independent time-sliced corpus where {source} and {target} co-occur, "
        f"then test whether the strength or direction of `{predicate}` remains stable across later time windows."
    )
    return {
        "system_id": "candidate_template",
        "system_title": "Candidate template baseline",
        "title": f"{source} {predicate} {target}",
        "premise": premise,
        "mechanism": mechanism,
        "time_scope": time_scope,
        "proposed_experiment": experiment,
        "supporting_evidence": _supporting_evidence(row),
        "score": float(((cand.get("score") or 0.0) if isinstance(cand, dict) else 0.0) or 0.0),
        "score_label": "candidate_score",
    }


def _stable_ab_order(pair_key: str) -> bool:
    digest = hashlib.sha256(pair_key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 2 == 0


def _build_ab_records(ranked_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(ranked_rows, start=1):
        cand = row.get("candidate") if isinstance(row.get("candidate"), dict) else {}
        pair_key = f"{idx}:{cand.get('source','')}|{cand.get('predicate','')}|{cand.get('target','')}"
        full_variant = _normalize_full_variant(row)
        baseline_variant = _baseline_variant_from_row(row)
        show_full_left = _stable_ab_order(pair_key)
        left_variant = full_variant if show_full_left else baseline_variant
        right_variant = baseline_variant if show_full_left else full_variant
        records.append(
            {
                "pair_id": f"pair-{idx:03d}",
                "rank": int(row.get("rank") or idx),
                "candidate": cand,
                "temporal_context": row.get("temporal_context") if isinstance(row.get("temporal_context"), dict) else {},
                "prediction_support": row.get("prediction_support") if isinstance(row.get("prediction_support"), dict) else {},
                "left_variant": left_variant,
                "right_variant": right_variant,
                "left_truth": left_variant["system_id"],
                "right_truth": right_variant["system_id"],
                "left_label": "A",
                "right_label": "B",
                "final_score": float(row.get("final_score") or 0.0),
                "review_defaults": {
                    "preferred_variant": "",
                    "better_temporal": "",
                    "better_evidence": "",
                    "better_testability": "",
                    "better_novelty": "",
                    "global_verdict": "",
                    "priority": "medium",
                    "confidence": "3",
                    "comments": "",
                },
            }
        )
    return records


def _default_meta(task_meta: Dict[str, Any], manifest: Dict[str, Any]) -> Dict[str, Any]:
    expert = task_meta.get("expert") if isinstance(task_meta.get("expert"), dict) else {}
    reviewer_default = str(expert.get("latin_slug") or expert.get("full_name") or task_meta.get("reviewer_id") or "")
    return {
        "topic": str(task_meta.get("topic") or manifest.get("query") or manifest.get("topic") or ""),
        "submission_id": str(task_meta.get("submission_id") or manifest.get("submission_id") or Path(manifest.get("bundle_dir") or "bundle").name),
        "cutoff_year": str(task_meta.get("cutoff_year") or manifest.get("cutoff_year") or ""),
        "domain": str(task_meta.get("domain") or manifest.get("domain_id") or ""),
        "bundle_dir": str(manifest.get("bundle_dir") or ""),
        "reviewer_default": reviewer_default,
    }


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
      --secondary: #3730a3;
      --shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--text); background: var(--bg); line-height: 1.45; }
    .page { max-width: 1440px; margin: 0 auto; padding: 20px; }
    .note, .card, details { background: var(--card); border: 1px solid var(--border); border-radius: 14px; box-shadow: var(--shadow); }
    .note { padding: 14px 16px; margin-bottom: 14px; }
    .muted { color: var(--muted); }
    .toolbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 12px 0 16px 0; }
    .toolbar input, .toolbar select, .toolbar textarea, .toolbar button {
      border: 1px solid var(--border); border-radius: 10px; padding: 8px 10px; font: inherit; background: #fff; color: var(--text);
    }
    .toolbar input[type="search"] { min-width: 280px; }
    .toolbar button.primary { background: var(--accent); color: white; border-color: var(--accent); }
    .toolbar button.secondary { background: var(--secondary); color: white; border-color: var(--secondary); }
    .stats { display: flex; gap: 8px; flex-wrap: wrap; margin: 12px 0; }
    .pill { display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: 999px; font-size: 12px; background: #eef2ff; color: #3730a3; }
    .grid { display: grid; gap: 14px; }
    .grid.cols-2 { grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); }
    .card { padding: 14px 16px; margin-bottom: 14px; }
    .variant { border: 1px solid var(--border); border-radius: 12px; padding: 12px; background: #fff; min-height: 280px; }
    .variant h4 { margin: 0 0 8px 0; }
    .variant .label { display: inline-flex; align-items: center; border-radius: 999px; padding: 3px 10px; font-size: 12px; background: var(--accent-soft); color: var(--accent); margin-bottom: 8px; }
    .kv { display: grid; grid-template-columns: 170px 1fr; gap: 6px 10px; font-size: 13px; }
    .kv div.key { color: var(--muted); }
    .controls { display: grid; gap: 10px; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); margin-top: 12px; }
    .controls label { display: flex; flex-direction: column; gap: 6px; font-size: 12px; color: var(--muted); }
    textarea { min-height: 88px; resize: vertical; }
    details { box-shadow: none; margin-top: 10px; }
    details > summary { cursor: pointer; padding: 10px 12px; font-weight: 600; }
    details > div, details > pre { padding: 0 12px 12px 12px; }
    pre { white-space: pre-wrap; word-break: break-word; background: #f8fafc; padding: 12px; border-radius: 10px; overflow: auto; }
    .footer { margin-top: 18px; color: var(--muted); font-size: 12px; }
  </style>
</head>
<body>
  <div class="page">
    <div class="note">
      <h1>Task 3 — автономная A/B форма для эксперта</h1>
      <div class="muted">Форма работает локально без ноутбука. Внутри уже встроены гипотезы, временной контекст и ключевые evidence snippets. Варианты A/B показывают <b>полную Task 3 гипотезу</b> и <b>baseline candidate template</b> в случайном слепом порядке.</div>
    </div>

    <div class="stats" id="meta-stats"></div>
    <section id="review-section"></section>
    <section id="summary-section"></section>
    <div class="footer">Для надёжной передачи между устройствами используйте «Скачать draft JSON» или ZIP с результатами review.</div>
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
        else if (key === 'text') node.textContent = value;
        else if (key === 'html') node.innerHTML = value;
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

  const STORAGE_KEY = `task3-offline-review:${APP.meta.submission_id || APP.meta.topic || 'bundle'}`;
  const state = {
    reviewer_id: APP.meta.reviewer_default || '',
    filters: { search: '', verdict: 'all', pageSize: 4, page: 1 },
    reviewState: Object.fromEntries(APP.records.map((row) => [row.pair_id, clone(row.review_defaults)])),
  };

  function safeLocalGet(key) { try { return window.localStorage.getItem(key); } catch (_) { return null; } }
  function safeLocalSet(key, value) { try { window.localStorage.setItem(key, value); } catch (_) { /* noop */ } }
  function htmlEscape(value) {
    return String(value ?? '').replace(/[&<>\"']/g, (ch) => ({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}[ch]));
  }
  function truncate(value, limit = 180) {
    const text = String(value ?? '').replace(/\s+/g, ' ').trim();
    return text.length <= limit ? text : `${text.slice(0, Math.max(0, limit - 1)).trim()}…`;
  }
  function downloadBytes(filename, mime, bytes) {
    const blob = bytes instanceof Blob ? bytes : new Blob([bytes], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename; document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function downloadText(filename, mime, text) { downloadBytes(filename, mime, new Blob([text], { type: mime })); }
  function toCsv(rows, columns) {
    const escape = (value) => {
      const text = value === null || value === undefined ? '' : String(value);
      if (/[\",\n]/.test(text)) return '"' + text.replace(/"/g, '""') + '"';
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
    const centralSize = centralParts.reduce((acc, part) => acc + part.length, 0);
    const end = new Uint8Array(22);
    let p = 0;
    end.set([0x50,0x4b,0x05,0x06], p); p += 4;
    end.set(u16(0), p); p += 2;
    end.set(u16(0), p); p += 2;
    end.set(u16(files.length), p); p += 2;
    end.set(u16(files.length), p); p += 2;
    end.set(u32(centralSize), p); p += 4;
    end.set(u32(offset), p); p += 4;
    end.set(u16(0), p);
    return new Blob([...localParts, ...centralParts, end], { type: 'application/zip' });
  }

  function snapshot(reason = 'manual') {
    return {
      artifact_version: APP.artifact_version || 1,
      reason,
      reviewer_id: state.reviewer_id,
      meta: APP.meta,
      filters: clone(state.filters),
      review_state: clone(state.reviewState),
      generated_at: new Date().toISOString(),
    };
  }
  function autosave(reason = 'autosave') { safeLocalSet(STORAGE_KEY, JSON.stringify(snapshot(reason))); }
  function loadAutosave() {
    const raw = safeLocalGet(STORAGE_KEY);
    if (!raw) return false;
    try {
      const payload = JSON.parse(raw);
      mergeLoaded(payload);
      return true;
    } catch (_) { return false; }
  }
  function mergeLoaded(payload) {
    if (!payload || typeof payload !== 'object') return;
    if (payload.reviewer_id) state.reviewer_id = String(payload.reviewer_id);
    if (payload.filters && typeof payload.filters === 'object') {
      state.filters = Object.assign({}, state.filters, payload.filters);
    }
    const loaded = payload.review_state || {};
    Object.keys(loaded).forEach((pairId) => {
      if (state.reviewState[pairId] && typeof loaded[pairId] === 'object') {
        state.reviewState[pairId] = Object.assign({}, state.reviewState[pairId], loaded[pairId]);
      }
    });
  }
  function currentRows() {
    let rows = APP.records.slice();
    const query = String(state.filters.search || '').trim().toLowerCase();
    if (query) {
      rows = rows.filter((row) => {
        const cand = row.candidate || {};
        const hay = [row.left_variant?.title, row.right_variant?.title, cand.source, cand.predicate, cand.target, row.left_variant?.premise, row.right_variant?.premise].join(' ').toLowerCase();
        return hay.includes(query);
      });
    }
    if (state.filters.verdict && state.filters.verdict !== 'all') {
      rows = rows.filter((row) => String((state.reviewState[row.pair_id] || {}).global_verdict || '') === state.filters.verdict);
    }
    return rows;
  }
  function summaryStats() {
    const rows = APP.records;
    const pref = { task3_full: 0, candidate_template: 0, tie: 0, skip: 0 };
    const verdicts = {};
    rows.forEach((row) => {
      const rv = state.reviewState[row.pair_id] || {};
      const v = String(rv.preferred_variant || '');
      if (!v) pref.skip += 1;
      else if (v === 'tie') pref.tie += 1;
      else if (v === 'A') pref[row.left_truth] = (pref[row.left_truth] || 0) + 1;
      else if (v === 'B') pref[row.right_truth] = (pref[row.right_truth] || 0) + 1;
      const gv = String(rv.global_verdict || '');
      if (gv) verdicts[gv] = (verdicts[gv] || 0) + 1;
    });
    return { pref, verdicts };
  }

  function buildReviewRows() {
    return APP.records.map((row) => {
      const rv = state.reviewState[row.pair_id] || {};
      const preferredSystem = !rv.preferred_variant ? '' : (rv.preferred_variant === 'tie' ? 'tie' : (rv.preferred_variant === 'A' ? row.left_truth : row.right_truth));
      const betterTemporalSystem = !rv.better_temporal ? '' : (rv.better_temporal === 'tie' ? 'tie' : (rv.better_temporal === 'A' ? row.left_truth : row.right_truth));
      const betterEvidenceSystem = !rv.better_evidence ? '' : (rv.better_evidence === 'tie' ? 'tie' : (rv.better_evidence === 'A' ? row.left_truth : row.right_truth));
      const betterTestabilitySystem = !rv.better_testability ? '' : (rv.better_testability === 'tie' ? 'tie' : (rv.better_testability === 'A' ? row.left_truth : row.right_truth));
      const betterNoveltySystem = !rv.better_novelty ? '' : (rv.better_novelty === 'tie' ? 'tie' : (rv.better_novelty === 'A' ? row.left_truth : row.right_truth));
      return {
        pair_id: row.pair_id,
        rank: row.rank,
        reviewer_id: state.reviewer_id,
        hypothesis_title_a: row.left_variant?.title || '',
        hypothesis_title_b: row.right_variant?.title || '',
        displayed_variant_a_system: row.left_truth,
        displayed_variant_b_system: row.right_truth,
        preferred_variant: rv.preferred_variant || '',
        preferred_system: preferredSystem,
        better_temporal: rv.better_temporal || '',
        better_temporal_system: betterTemporalSystem,
        better_evidence: rv.better_evidence || '',
        better_evidence_system: betterEvidenceSystem,
        better_testability: rv.better_testability || '',
        better_testability_system: betterTestabilitySystem,
        better_novelty: rv.better_novelty || '',
        better_novelty_system: betterNoveltySystem,
        global_verdict: rv.global_verdict || '',
        priority: rv.priority || '',
        confidence: rv.confidence || '',
        comments: rv.comments || '',
        candidate_source: row.candidate?.source || '',
        candidate_predicate: row.candidate?.predicate || '',
        candidate_target: row.candidate?.target || '',
        final_score: row.final_score || 0,
      };
    });
  }

  function buildSummaryPayload() {
    const rows = buildReviewRows();
    const completed = rows.filter((row) => row.preferred_variant || row.global_verdict || row.comments).length;
    const stats = summaryStats();
    return {
      artifact_version: APP.artifact_version || 1,
      topic: APP.meta.topic,
      submission_id: APP.meta.submission_id,
      reviewer_id: state.reviewer_id,
      timestamp: new Date().toISOString(),
      total_pairs: APP.records.length,
      completed_pairs: completed,
      preference_counts: stats.pref,
      verdict_counts: stats.verdicts,
    };
  }

  function bindSelect(select, pairId, key) {
    select.value = state.reviewState[pairId][key] || '';
    select.addEventListener('change', () => {
      state.reviewState[pairId][key] = select.value;
      autosave('select');
      renderSummary();
    });
    return select;
  }
  function bindTextarea(area, pairId, key) {
    area.value = state.reviewState[pairId][key] || '';
    area.addEventListener('input', () => {
      state.reviewState[pairId][key] = area.value;
      autosave('textarea');
    });
    return area;
  }

  function variantNode(label, variant) {
    const evidence = Array.isArray(variant.supporting_evidence) ? variant.supporting_evidence : [];
    const evidenceList = evidence.length ? el('ul', null, evidence.map((item) => el('li', { text: `${item.source_id ? `[${item.source_id}] ` : ''}${truncate(item.text_snippet, 180)}` }))) : el('div', { class: 'muted', text: 'Нет встроенных evidence snippets.' });
    return el('div', { class: 'variant' },
      el('div', { class: 'label', text: `Variant ${label}` }),
      el('h4', { text: variant.title || '(без названия)' }),
      el('div', { class: 'kv' },
        el('div', { class: 'key', text: 'Premise' }), el('div', { text: variant.premise || '' }),
        el('div', { class: 'key', text: 'Mechanism' }), el('div', { text: variant.mechanism || '' }),
        el('div', { class: 'key', text: 'Time scope' }), el('div', { text: variant.time_scope || '' }),
        el('div', { class: 'key', text: 'Experiment' }), el('div', { text: variant.proposed_experiment || '' }),
        el('div', { class: 'key', text: variant.score_label || 'score' }), el('div', { text: String(variant.score ?? '') })
      ),
      el('details', { open: false },
        el('summary', { text: 'Evidence' }),
        el('div', null, evidenceList)
      )
    );
  }

  function renderMetaStats() {
    const host = document.getElementById('meta-stats');
    host.innerHTML = '';
    const stats = summaryStats();
    host.append(
      el('span', { class: 'pill', text: `topic: ${APP.meta.topic || '—'}` }),
      el('span', { class: 'pill', text: `submission_id: ${APP.meta.submission_id || '—'}` }),
      el('span', { class: 'pill', text: `hypotheses: ${APP.records.length}` }),
      el('span', { class: 'pill', text: `Task3 full preferred: ${stats.pref.task3_full || 0}` }),
      el('span', { class: 'pill', text: `Baseline preferred: ${stats.pref.candidate_template || 0}` }),
      el('span', { class: 'pill', text: `ties: ${stats.pref.tie || 0}` })
    );
  }

  function renderReview() {
    const host = document.getElementById('review-section');
    host.innerHTML = '';
    const reviewer = el('input', { type: 'text', value: state.reviewer_id || '', placeholder: 'reviewer_id / имя эксперта' });
    reviewer.addEventListener('input', () => { state.reviewer_id = reviewer.value; autosave('reviewer'); renderSummary(); });
    const search = el('input', { type: 'search', value: state.filters.search || '', placeholder: 'Поиск по hypothesis / source / target' });
    search.addEventListener('input', () => { state.filters.search = search.value; state.filters.page = 1; renderReview(); });
    const verdict = el('select', null,
      el('option', { value: 'all', text: 'Все verdict' }),
      el('option', { value: 'accept', text: 'accept' }),
      el('option', { value: 'partial', text: 'partial' }),
      el('option', { value: 'needs_fix', text: 'needs_fix' }),
      el('option', { value: 'reject', text: 'reject' })
    );
    verdict.value = state.filters.verdict || 'all';
    verdict.addEventListener('change', () => { state.filters.verdict = verdict.value; state.filters.page = 1; renderReview(); });
    const pageSize = el('select', null, [4, 8, 12, 20].map((n) => el('option', { value: String(n), text: `${n} на страницу` })));
    pageSize.value = String(state.filters.pageSize || 4);
    pageSize.addEventListener('change', () => { state.filters.pageSize = Number(pageSize.value || 4); state.filters.page = 1; renderReview(); });
    const saveDraft = el('button', { text: 'Скачать draft JSON' });
    saveDraft.addEventListener('click', () => downloadText('review_state_latest.json', 'application/json', JSON.stringify(snapshot('download_draft'), null, 2)));
    const importInput = el('input', { type: 'file', accept: '.json', class: 'hidden' });
    const importBtn = el('button', { text: 'Импорт draft JSON' });
    importBtn.addEventListener('click', () => importInput.click());
    importInput.addEventListener('change', async () => {
      const file = importInput.files && importInput.files[0];
      if (!file) return;
      const text = await file.text();
      mergeLoaded(JSON.parse(text));
      autosave('import');
      renderMetaStats();
      renderReview();
      renderSummary();
    });
    const exportZip = el('button', { class: 'primary', text: 'Скачать ZIP с результатами' });
    exportZip.addEventListener('click', () => {
      const rows = buildReviewRows();
      const summary = buildSummaryPayload();
      const files = [
        { name: 'hypothesis_reviews.csv', data: toCsv(rows, ['pair_id','rank','reviewer_id','hypothesis_title_a','hypothesis_title_b','displayed_variant_a_system','displayed_variant_b_system','preferred_variant','preferred_system','better_temporal','better_temporal_system','better_evidence','better_evidence_system','better_testability','better_testability_system','better_novelty','better_novelty_system','global_verdict','priority','confidence','comments','candidate_source','candidate_predicate','candidate_target','final_score']) },
        { name: 'hypothesis_reviews.json', data: JSON.stringify({ artifact_version: APP.artifact_version || 1, topic: APP.meta.topic, submission_id: APP.meta.submission_id, reviewer_id: state.reviewer_id, reviews: rows }, null, 2) },
        { name: 'ab_test_summary.json', data: JSON.stringify(summary, null, 2) },
        { name: 'review_state_latest.json', data: JSON.stringify(snapshot('export'), null, 2) },
      ];
      downloadBytes('expert_hypothesis_review_bundle.zip', 'application/zip', zipStore(files));
    });
    host.appendChild(el('div', { class: 'toolbar' }, reviewer, search, verdict, pageSize, saveDraft, importBtn, importInput, exportZip));

    const rows = currentRows();
    const totalPages = Math.max(1, Math.ceil(rows.length / state.filters.pageSize));
    if (state.filters.page > totalPages) state.filters.page = totalPages;
    const start = (state.filters.page - 1) * state.filters.pageSize;
    const visible = rows.slice(start, start + state.filters.pageSize);
    host.appendChild(el('div', { class: 'muted', text: `Показано ${visible.length} из ${rows.length} hypotheses. Страница ${state.filters.page}/${totalPages}.` }));

    visible.forEach((row) => {
      const cand = row.candidate || {};
      const rv = state.reviewState[row.pair_id] || {};
      const card = el('div', { class: 'card' });
      card.appendChild(el('div', { html: `<b>H-${String(row.rank).padStart(3, '0')}</b> · ${htmlEscape(String(cand.source || 'source'))} <code>${htmlEscape(String(cand.predicate || 'related_to'))}</code> ${htmlEscape(String(cand.target || 'target'))}` }));
      card.appendChild(el('div', { class: 'muted', text: `final_score=${Number(row.final_score || 0).toFixed(3)}` }));
      card.appendChild(el('div', { class: 'grid cols-2' }, variantNode(row.left_label, row.left_variant), variantNode(row.right_label, row.right_variant)));
      card.appendChild(el('details', { open: false },
        el('summary', { text: 'Candidate + temporal context' }),
        el('pre', { text: JSON.stringify({ candidate: row.candidate, temporal_context: row.temporal_context, prediction_support: row.prediction_support }, null, 2) })
      ));
      const controls = el('div', { class: 'controls' });
      const pref = bindSelect(el('select', null,
        el('option', { value: '', text: 'preferred variant?' }),
        el('option', { value: 'A', text: 'A' }),
        el('option', { value: 'B', text: 'B' }),
        el('option', { value: 'tie', text: 'tie' })
      ), row.pair_id, 'preferred_variant');
      const betterTemporal = bindSelect(el('select', null,
        el('option', { value: '', text: 'better temporal reasoning?' }),
        el('option', { value: 'A', text: 'A' }),
        el('option', { value: 'B', text: 'B' }),
        el('option', { value: 'tie', text: 'tie' })
      ), row.pair_id, 'better_temporal');
      const betterEvidence = bindSelect(el('select', null,
        el('option', { value: '', text: 'better evidence grounding?' }),
        el('option', { value: 'A', text: 'A' }),
        el('option', { value: 'B', text: 'B' }),
        el('option', { value: 'tie', text: 'tie' })
      ), row.pair_id, 'better_evidence');
      const betterTestability = bindSelect(el('select', null,
        el('option', { value: '', text: 'better testability?' }),
        el('option', { value: 'A', text: 'A' }),
        el('option', { value: 'B', text: 'B' }),
        el('option', { value: 'tie', text: 'tie' })
      ), row.pair_id, 'better_testability');
      const betterNovelty = bindSelect(el('select', null,
        el('option', { value: '', text: 'better novelty?' }),
        el('option', { value: 'A', text: 'A' }),
        el('option', { value: 'B', text: 'B' }),
        el('option', { value: 'tie', text: 'tie' })
      ), row.pair_id, 'better_novelty');
      const verdictSelect = bindSelect(el('select', null,
        el('option', { value: '', text: 'global verdict' }),
        el('option', { value: 'accept', text: 'accept' }),
        el('option', { value: 'partial', text: 'partial' }),
        el('option', { value: 'needs_fix', text: 'needs_fix' }),
        el('option', { value: 'reject', text: 'reject' })
      ), row.pair_id, 'global_verdict');
      const priority = bindSelect(el('select', null,
        el('option', { value: 'high', text: 'high priority' }),
        el('option', { value: 'medium', text: 'medium priority' }),
        el('option', { value: 'low', text: 'low priority' })
      ), row.pair_id, 'priority');
      const confidence = bindSelect(el('select', null, [1,2,3,4,5].map((n) => el('option', { value: String(n), text: `confidence ${n}` }))), row.pair_id, 'confidence');
      const comments = bindTextarea(el('textarea', { placeholder: 'Комментарий эксперта / почему выбран вариант / что исправить' }), row.pair_id, 'comments');
      controls.append(
        el('label', null, 'Preferred', pref),
        el('label', null, 'Temporal', betterTemporal),
        el('label', null, 'Evidence', betterEvidence),
        el('label', null, 'Testability', betterTestability),
        el('label', null, 'Novelty', betterNovelty),
        el('label', null, 'Verdict', verdictSelect),
        el('label', null, 'Priority', priority),
        el('label', null, 'Confidence', confidence),
        el('label', { style: 'grid-column: 1 / -1;' }, 'Comments', comments)
      );
      card.appendChild(controls);
      host.appendChild(card);
    });

    const pager = el('div', { class: 'toolbar' });
    const prev = el('button', { text: '← Prev' });
    const next = el('button', { text: 'Next →' });
    prev.disabled = state.filters.page <= 1;
    next.disabled = state.filters.page >= totalPages;
    prev.addEventListener('click', () => { if (state.filters.page > 1) { state.filters.page -= 1; renderReview(); } });
    next.addEventListener('click', () => { if (state.filters.page < totalPages) { state.filters.page += 1; renderReview(); } });
    pager.append(prev, el('span', { class: 'muted', text: `page ${state.filters.page}/${totalPages}` }), next);
    host.appendChild(pager);
  }

  function renderSummary() {
    const host = document.getElementById('summary-section');
    host.innerHTML = '';
    const rows = buildReviewRows();
    const summary = buildSummaryPayload();
    host.appendChild(el('div', { class: 'note', html: `<h2>Сводка A/B review</h2><div class="muted">Completed pairs: <b>${summary.completed_pairs}</b> / ${summary.total_pairs}</div>` }));
    host.appendChild(el('pre', { text: JSON.stringify(summary, null, 2) }));
    host.appendChild(el('details', { open: false }, el('summary', { text: 'Preview exported rows' }), el('pre', { text: JSON.stringify(rows.slice(0, 5), null, 2) })));
  }

  if (!loadAutosave()) autosave('init');
  renderMetaStats();
  renderReview();
  renderSummary();
  </script>
</body>
</html>
"""


def build_task3_offline_review_package(
    manifest: Dict[str, Any] | str | Path,
    task_meta: Optional[Dict[str, Any]] = None,
    *,
    output_path: str | Path | None = None,
) -> Path:
    if not isinstance(manifest, dict):
        manifest_path = Path(manifest)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    task_meta = task_meta or {}
    bundle_dir = Path(manifest["bundle_dir"])
    hypotheses_path = _resolve_bundle_artifact(manifest, bundle_dir, "hypotheses_ranked")
    if hypotheses_path is None:
        raise FileNotFoundError("Task 3 hypotheses_ranked artifact not found in manifest")

    ranked_rows = _safe_json_load(hypotheses_path) or []
    if not isinstance(ranked_rows, list):
        raise ValueError("Task 3 hypotheses_ranked must be a list")

    output = Path(output_path) if output_path else bundle_dir / "expert_review" / "offline_review" / "task3_hypothesis_review_offline_ab.html"
    output.parent.mkdir(parents=True, exist_ok=True)

    app_data = {
        "artifact_version": ARTIFACT_VERSION,
        "meta": _default_meta(task_meta, manifest),
        "records": _build_ab_records(ranked_rows),
    }
    app_json = json.dumps(app_data, ensure_ascii=False).replace("</", "<\\/")
    page_title = html.escape(f"Task 3 offline review — {app_data['meta']['submission_id'] or app_data['meta']['topic'] or 'bundle'}")
    html_text = _HTML_TEMPLATE.replace("__APP_DATA__", app_json).replace("__PAGE_TITLE__", page_title)
    output.write_text(html_text, encoding="utf-8")
    return output


def _artifact_files_for_expert(manifest: Dict[str, Any], bundle_dir: Path) -> List[Path]:
    wanted = [
        bundle_dir / "task3_manifest.json",
        bundle_dir / "query.json",
        bundle_dir / "trajectory_snapshot.json",
        bundle_dir / "papers_selected.json",
        bundle_dir / "paper_records.json",
        bundle_dir / "chunk_registry.jsonl",
        bundle_dir / "hypotheses_candidates.json",
        bundle_dir / "hypotheses_ranked.json",
        bundle_dir / "hypotheses_ranked.md",
        bundle_dir / "automatic_graph" / "temporal_kg.json",
        bundle_dir / "automatic_graph" / "events.jsonl",
        bundle_dir / "automatic_graph" / "multimodal_triplets.jsonl",
        bundle_dir / "automatic_graph" / "link_predictions.json",
        bundle_dir / "automatic_graph" / "vlm_candidate_analysis.jsonl",
        bundle_dir / "annoy" / "annoy_manifest.json",
        bundle_dir / "expert_review" / "offline_review" / "task3_hypothesis_review_offline_ab.html",
    ]
    files: list[Path] = [p for p in wanted if p.exists() and p.is_file()]
    processed_dir = bundle_dir / "processed_papers"
    if processed_dir.exists():
        for path in processed_dir.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(processed_dir)
            name = path.name.lower()
            if name.endswith(".pdf") or name.endswith(".png") or name.endswith(".jpg") or name.endswith(".jpeg"):
                continue
            if path.suffix.lower() in {".json", ".jsonl", ".md", ".txt"}:
                files.append(path)
    seen: set[str] = set()
    uniq: list[Path] = []
    for item in files:
        key = str(item.resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(item)
    return uniq


def build_task3_expert_artifact_bundle(
    manifest: Dict[str, Any] | str | Path,
    task_meta: Optional[Dict[str, Any]] = None,
    *,
    output_path: str | Path | None = None,
) -> Path:
    if not isinstance(manifest, dict):
        manifest_path = Path(manifest)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    task_meta = task_meta or {}
    bundle_dir = Path(manifest["bundle_dir"])
    offline_path = build_task3_offline_review_package(manifest, task_meta)
    output = Path(output_path) if output_path else bundle_dir / "expert_review" / "expert_hypothesis_artifacts_bundle.zip"
    output.parent.mkdir(parents=True, exist_ok=True)

    included_files = _artifact_files_for_expert(manifest, bundle_dir)
    review_manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "topic": str((task_meta or {}).get("topic") or manifest.get("query") or ""),
        "submission_id": str((task_meta or {}).get("submission_id") or Path(bundle_dir).name),
        "bundle_dir": str(bundle_dir),
        "offline_review_html": str(offline_path),
        "included_files": [str(path.relative_to(bundle_dir)) for path in included_files if path.exists() and path.is_relative_to(bundle_dir)],
    }
    review_manifest_path = bundle_dir / "expert_review" / "task3_expert_review_manifest.json"
    _write_json(review_manifest_path, review_manifest)
    if review_manifest_path not in included_files:
        included_files.append(review_manifest_path)

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in included_files:
            if not path.exists() or not path.is_file():
                continue
            arcname = str(path.relative_to(bundle_dir)) if path.is_relative_to(bundle_dir) else path.name
            zf.write(path, arcname=arcname)
    return output


__all__ = [
    "ARTIFACT_VERSION",
    "build_task3_offline_review_package",
    "build_task3_expert_artifact_bundle",
]
