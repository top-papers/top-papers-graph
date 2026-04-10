from __future__ import annotations

import hashlib
import html
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class DualModelReviewAssets:
    offline_html_path: Path
    owner_mapping_path: Path
    public_manifest_path: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def _truncate(value: Any, limit: int = 240) -> str:
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
    time_scope = str(tc.get("time_scope") or tc.get("interval") or tc.get("range") or row.get("time_scope") or "")
    parts = [f"ordering={ordering}"]
    if years:
        parts.append("years=" + ", ".join(str(x) for x in years[:8]))
    if time_scope:
        parts.append(f"scope={time_scope}")
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


def _variant_from_row(row: Dict[str, Any], *, system_id: str, display_label: str, score_label: str = "final_score") -> Dict[str, Any]:
    hyp = row.get("hypothesis") if isinstance(row.get("hypothesis"), dict) else {}
    title = str(hyp.get("title") or row.get("title") or "Generated Task 3 hypothesis")
    score_value = row.get("final_score")
    if score_value is None:
        candidate = row.get("candidate") if isinstance(row.get("candidate"), dict) else {}
        score_value = candidate.get("score")
    try:
        score = float(score_value or 0.0)
    except Exception:
        score = 0.0
    return {
        "system_id": system_id,
        "system_title": display_label,
        "title": title,
        "premise": str(hyp.get("premise") or row.get("premise") or ""),
        "mechanism": str(hyp.get("mechanism") or row.get("mechanism") or ""),
        "time_scope": str(hyp.get("time_scope") or row.get("time_scope") or _temporal_summary(row)),
        "proposed_experiment": str(hyp.get("proposed_experiment") or row.get("proposed_experiment") or ""),
        "supporting_evidence": _supporting_evidence(row),
        "score": score,
        "score_label": score_label,
    }


def _candidate_signature(row: Dict[str, Any], *, include_rank_fallback: bool = False) -> str:
    cand = row.get("candidate") if isinstance(row.get("candidate"), dict) else {}
    fields = [
        str(cand.get("kind") or "").strip().lower(),
        str(cand.get("source") or "").strip().lower(),
        str(cand.get("predicate") or "").strip().lower(),
        str(cand.get("target") or "").strip().lower(),
    ]
    if any(fields[1:]):
        return "|".join(fields)
    hyp = row.get("hypothesis") if isinstance(row.get("hypothesis"), dict) else {}
    title = str(hyp.get("title") or row.get("title") or "").strip().lower()
    time_scope = str(hyp.get("time_scope") or row.get("time_scope") or "").strip().lower()
    if title:
        return f"title|{title}|{time_scope}"
    if include_rank_fallback:
        return f"rank|{int(row.get('rank') or 0)}"
    return ""


def _merge_candidate(row_a: Dict[str, Any], row_b: Dict[str, Any], *, pair_index: int) -> Dict[str, Any]:
    cand_a = row_a.get("candidate") if isinstance(row_a.get("candidate"), dict) else {}
    cand_b = row_b.get("candidate") if isinstance(row_b.get("candidate"), dict) else {}
    candidate = cand_a or cand_b
    merged = {
        "kind": str(candidate.get("kind") or "comparison_pair"),
        "source": str(candidate.get("source") or cand_b.get("source") or f"hypothesis_pair_{pair_index:03d}"),
        "predicate": str(candidate.get("predicate") or cand_b.get("predicate") or "compared_with"),
        "target": str(candidate.get("target") or cand_b.get("target") or "dual_local_model"),
        "score": float((candidate.get("score") or cand_b.get("score") or 0.0) or 0.0),
        "time_scope": str(candidate.get("time_scope") or cand_b.get("time_scope") or ""),
        "graph_signals": candidate.get("graph_signals") or cand_b.get("graph_signals") or {},
    }
    return merged


def _stable_ab_order(pair_key: str) -> bool:
    digest = hashlib.sha256(pair_key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 2 == 0


@dataclass(frozen=True)
class _PairedRows:
    rank_a: int
    rank_b: int
    row_a: Dict[str, Any]
    row_b: Dict[str, Any]
    pair_key: str
    match_mode: str


def _pair_ranked_rows(
    rows_a: Sequence[Dict[str, Any]],
    rows_b: Sequence[Dict[str, Any]],
    *,
    top_pairs: Optional[int] = None,
) -> List[_PairedRows]:
    limit = int(top_pairs or min(len(rows_a), len(rows_b)))
    if limit <= 0:
        return []

    indexed_a = list(enumerate(rows_a, start=1))
    indexed_b = list(enumerate(rows_b, start=1))
    bucket_b: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    for rank_b, row_b in indexed_b:
        sig = _candidate_signature(row_b)
        if not sig:
            continue
        bucket_b.setdefault(sig, []).append((rank_b, row_b))

    used_b: set[int] = set()
    used_a: set[int] = set()
    pairs: list[_PairedRows] = []

    for rank_a, row_a in indexed_a:
        if len(pairs) >= limit:
            break
        sig = _candidate_signature(row_a)
        if not sig:
            continue
        choices = bucket_b.get(sig) or []
        while choices and choices[0][0] in used_b:
            choices.pop(0)
        if not choices:
            continue
        rank_b, row_b = choices.pop(0)
        used_a.add(rank_a)
        used_b.add(rank_b)
        pairs.append(
            _PairedRows(
                rank_a=rank_a,
                rank_b=rank_b,
                row_a=row_a,
                row_b=row_b,
                pair_key=f"candidate:{sig}",
                match_mode="candidate_signature",
            )
        )

    remaining_a = [(rank, row) for rank, row in indexed_a if rank not in used_a]
    remaining_b = [(rank, row) for rank, row in indexed_b if rank not in used_b]
    for (rank_a, row_a), (rank_b, row_b) in zip(remaining_a, remaining_b):
        if len(pairs) >= limit:
            break
        pair_key = f"rank:{rank_a}:{rank_b}:{_candidate_signature(row_a, include_rank_fallback=True)}:{_candidate_signature(row_b, include_rank_fallback=True)}"
        pairs.append(
            _PairedRows(
                rank_a=rank_a,
                rank_b=rank_b,
                row_a=row_a,
                row_b=row_b,
                pair_key=pair_key,
                match_mode="rank_fallback",
            )
        )
    return pairs[:limit]


def _default_meta(task_meta: Dict[str, Any], manifest: Dict[str, Any]) -> Dict[str, Any]:
    expert = task_meta.get("expert") if isinstance(task_meta.get("expert"), dict) else {}
    reviewer_default = str(expert.get("latin_slug") or expert.get("full_name") or task_meta.get("reviewer_id") or "")
    return {
        "topic": str(task_meta.get("topic") or manifest.get("query") or manifest.get("topic") or ""),
        "submission_id": str(task_meta.get("submission_id") or manifest.get("submission_id") or Path(manifest.get("bundle_dir") or "bundle").name),
        "cutoff_year": str(task_meta.get("cutoff_year") or manifest.get("cutoff_year") or ""),
        "reviewer_default": reviewer_default,
    }


def _anon_system_id(seed: str, slot: str) -> str:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:10]
    return f"hidden_{slot}_{digest}"


def _build_records(
    rows_a: Sequence[Dict[str, Any]],
    rows_b: Sequence[Dict[str, Any]],
    *,
    anon_a: str,
    anon_b: str,
    label_a: str,
    label_b: str,
    top_pairs: Optional[int] = None,
) -> List[Dict[str, Any]]:
    pairs = _pair_ranked_rows(rows_a, rows_b, top_pairs=top_pairs)
    records: list[dict[str, Any]] = []
    for idx, pair in enumerate(pairs, start=1):
        variant_a = _variant_from_row(pair.row_a, system_id=anon_a, display_label=label_a)
        variant_b = _variant_from_row(pair.row_b, system_id=anon_b, display_label=label_b)
        show_a_left = _stable_ab_order(f"{pair.pair_key}:{idx}")
        left_variant = variant_a if show_a_left else variant_b
        right_variant = variant_b if show_a_left else variant_a
        candidate = _merge_candidate(pair.row_a, pair.row_b, pair_index=idx)
        records.append(
            {
                "pair_id": f"pair-{idx:03d}",
                "rank": idx,
                "rank_model_a": pair.rank_a,
                "rank_model_b": pair.rank_b,
                "match_mode": pair.match_mode,
                "candidate": candidate,
                "left_variant": left_variant,
                "right_variant": right_variant,
                "left_truth": left_variant["system_id"],
                "right_truth": right_variant["system_id"],
                "left_label": "A",
                "right_label": "B",
                "paired_score": float((variant_a.get("score") or 0.0) + (variant_b.get("score") or 0.0)) / 2.0,
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


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__PAGE_TITLE__</title>
  <style>
    :root {
      --bg: #f7fafc;
      --panel: #ffffff;
      --border: #d9e2ec;
      --text: #123;
      --muted: #5b6b7a;
      --accent: #0f62fe;
      --accent-soft: #edf4ff;
      --ok: #137333;
      --warn: #b26a00;
      --danger: #b42318;
    }
    * { box-sizing: border-box; }
    body { margin: 0; padding: 0; background: var(--bg); color: var(--text); font: 14px/1.5 -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; }
    .page { max-width: 1320px; margin: 0 auto; padding: 24px 18px 48px; }
    .hero { background: linear-gradient(180deg, #ffffff, #f5f9ff); border: 1px solid var(--border); border-radius: 16px; padding: 18px; margin-bottom: 18px; }
    h1, h2, h3, h4 { margin: 0 0 10px 0; }
    .muted { color: var(--muted); }
    .note { border-left: 4px solid var(--accent); background: #fff; border-radius: 10px; border: 1px solid var(--border); padding: 12px 14px; margin: 12px 0; }
    .toolbar { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin: 10px 0 14px; }
    .toolbar > * { flex: 0 0 auto; }
    button, select, input, textarea {
      font: inherit; border: 1px solid var(--border); border-radius: 10px; padding: 8px 10px; background: #fff; color: var(--text);
    }
    button { cursor: pointer; }
    button.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
    button.secondary { background: #fff; color: var(--accent); }
    textarea { width: 100%; min-height: 84px; resize: vertical; }
    .pill { display: inline-flex; align-items: center; gap: 6px; border: 1px solid var(--border); background: #fff; padding: 5px 10px; border-radius: 999px; }
    .card { background: var(--panel); border: 1px solid var(--border); border-radius: 16px; padding: 16px; margin: 16px 0; box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04); }
    .grid { display: grid; gap: 14px; }
    .cols-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .variant { border: 1px solid var(--border); border-radius: 12px; padding: 12px; background: #fff; min-height: 280px; }
    .variant .label { display: inline-flex; align-items: center; border-radius: 999px; padding: 3px 10px; font-size: 12px; background: var(--accent-soft); color: var(--accent); margin-bottom: 8px; }
    .kv { display: grid; grid-template-columns: 150px 1fr; gap: 6px 10px; margin: 10px 0; }
    .kv .key { color: var(--muted); }
    .controls { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-top: 12px; }
    .controls label { display: grid; gap: 4px; }
    details { border-top: 1px dashed var(--border); margin-top: 10px; padding-top: 10px; }
    pre { background: #fff; border: 1px solid var(--border); border-radius: 12px; padding: 12px; overflow: auto; }
    .footer { color: var(--muted); font-size: 12px; margin-top: 18px; }
    @media (max-width: 980px) {
      .cols-2, .controls { grid-template-columns: 1fr; }
      .kv { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Task 3 — Blind A/B review for dual local models</h1>
      <div class="muted">Эксперт видит только варианты A/B и анонимные системы. Ключ соответствия хранится отдельно у владельца запуска.</div>
      <div id="meta-stats" class="toolbar"></div>
    </section>

    <div class="note">
      <b>Как работать с формой:</b>
      1) сравните две гипотезы, 2) отметьте лучший вариант по критериям, 3) регулярно сохраняйте draft JSON или ZIP с review.
      В этой HTML-странице результаты могут также автоматически сохраняться в localStorage браузера.
    </div>

    <section id="review-section"></section>
    <section id="summary-section"></section>
    <div class="footer">Файл автономный: интернет не нужен. Для передачи между устройствами используйте export JSON или ZIP.</div>
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

  const STORAGE_KEY = `task3-dual-local-review:${APP.meta.submission_id || APP.meta.topic || 'bundle'}`;
  const systemLabels = Object.fromEntries((APP.meta.anonymous_systems || []).map((row) => [row.system_id, row.display_label || row.system_id]));
  const state = {
    reviewer_id: APP.meta.reviewer_default || '',
    filters: { search: '', verdict: 'all', pageSize: 4, page: 1 },
    reviewState: Object.fromEntries(APP.records.map((row) => [row.pair_id, clone(row.review_defaults)])),
  };

  function safeLocalGet(key) { try { return window.localStorage.getItem(key); } catch (_) { return null; } }
  function safeLocalSet(key, value) { try { window.localStorage.setItem(key, value); } catch (_) { /* noop */ } }
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
    if (payload.filters && typeof payload.filters === 'object') state.filters = Object.assign({}, state.filters, payload.filters);
    const loaded = payload.review_state || {};
    Object.keys(loaded).forEach((pairId) => {
      if (state.reviewState[pairId] && typeof loaded[pairId] === 'object') state.reviewState[pairId] = Object.assign({}, state.reviewState[pairId], loaded[pairId]);
    });
  }

  function summaryStats() {
    const pref = { tie: 0, skip: 0 };
    (APP.meta.anonymous_systems || []).forEach((row) => { pref[row.system_id] = 0; });
    const verdicts = {};
    APP.records.forEach((row) => {
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
        match_mode: row.match_mode || '',
        rank_model_a: row.rank_model_a || '',
        rank_model_b: row.rank_model_b || '',
        candidate_source: row.candidate?.source || '',
        candidate_predicate: row.candidate?.predicate || '',
        candidate_target: row.candidate?.target || '',
        paired_score: row.paired_score ?? '',
      };
    });
  }

  function buildSummaryPayload() {
    const stats = summaryStats();
    const rows = buildReviewRows();
    const completed = rows.filter((row) => row.preferred_variant || row.global_verdict || row.comments).length;
    return {
      artifact_version: APP.artifact_version || 1,
      review_mode: APP.meta.review_mode || 'dual_local_model_blind_ab',
      topic: APP.meta.topic || '',
      submission_id: APP.meta.submission_id || '',
      reviewer_id: state.reviewer_id || '',
      completed_pairs: completed,
      total_pairs: APP.records.length,
      preference_counts: stats.pref,
      verdict_counts: stats.verdicts,
      anonymous_systems: APP.meta.anonymous_systems || [],
      generated_at: new Date().toISOString(),
    };
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
    if (state.filters.verdict && state.filters.verdict !== 'all') rows = rows.filter((row) => String((state.reviewState[row.pair_id] || {}).global_verdict || '') === state.filters.verdict);
    return rows;
  }

  function bindSelect(node, pairId, field) {
    node.value = String((state.reviewState[pairId] || {})[field] || '');
    node.addEventListener('change', () => {
      state.reviewState[pairId][field] = node.value;
      autosave(`select:${field}`);
      renderMetaStats();
      renderSummary();
    });
    return node;
  }
  function bindTextarea(node, pairId, field) {
    node.value = String((state.reviewState[pairId] || {})[field] || '');
    node.addEventListener('input', () => {
      state.reviewState[pairId][field] = node.value;
      autosave(`textarea:${field}`);
      renderSummary();
    });
    return node;
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
      el('span', { class: 'pill', text: `pairs: ${APP.records.length}` })
    );
    (APP.meta.anonymous_systems || []).forEach((row) => host.append(el('span', { class: 'pill', text: `${row.display_label || row.system_id}: ${stats.pref[row.system_id] || 0}` })));
    host.append(el('span', { class: 'pill', text: `ties: ${stats.pref.tie || 0}` }));
  }

  function exportRows() {
    const rows = buildReviewRows();
    const summary = buildSummaryPayload();
    return { rows, summary };
  }

  function downloadDraftJson() {
    downloadText('task3_dual_model_review_draft.json', 'application/json', JSON.stringify(snapshot('manual_export'), null, 2));
  }
  function downloadReviewJson() {
    const payload = exportRows();
    downloadText('task3_dual_model_review_results.json', 'application/json', JSON.stringify(payload, null, 2));
  }
  function downloadReviewCsv() {
    const { rows } = exportRows();
    const csv = toCsv(rows, ['pair_id','rank','reviewer_id','hypothesis_title_a','hypothesis_title_b','displayed_variant_a_system','displayed_variant_b_system','preferred_variant','preferred_system','better_temporal','better_temporal_system','better_evidence','better_evidence_system','better_testability','better_testability_system','better_novelty','better_novelty_system','global_verdict','priority','confidence','comments','match_mode','rank_model_a','rank_model_b','candidate_source','candidate_predicate','candidate_target','paired_score']);
    downloadText('task3_dual_model_review_results.csv', 'text/csv', csv);
  }
  function downloadReviewZip() {
    const { rows, summary } = exportRows();
    const csv = toCsv(rows, ['pair_id','rank','reviewer_id','hypothesis_title_a','hypothesis_title_b','displayed_variant_a_system','displayed_variant_b_system','preferred_variant','preferred_system','better_temporal','better_temporal_system','better_evidence','better_evidence_system','better_testability','better_testability_system','better_novelty','better_novelty_system','global_verdict','priority','confidence','comments','match_mode','rank_model_a','rank_model_b','candidate_source','candidate_predicate','candidate_target','paired_score']);
    const blob = zipStore([
      { name: 'hypothesis_reviews.csv', data: csv },
      { name: 'hypothesis_reviews.json', data: JSON.stringify(rows, null, 2) },
      { name: 'ab_test_summary.json', data: JSON.stringify(summary, null, 2) },
      { name: 'review_state_latest.json', data: JSON.stringify(snapshot('zip_export'), null, 2) },
    ]);
    downloadBytes('task3_dual_model_review_export.zip', 'application/zip', blob);
  }
  function restoreFromFile(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const payload = JSON.parse(String(reader.result || '{}'));
        mergeLoaded(payload);
        autosave('restore');
        renderMetaStats();
        renderReview();
        renderSummary();
      } catch (err) {
        alert(`Не удалось загрузить JSON: ${err}`);
      }
    };
    reader.readAsText(file, 'utf-8');
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
      el('option', { value: 'needs_revision', text: 'needs_revision' }),
      el('option', { value: 'reject', text: 'reject' })
    );
    verdict.value = state.filters.verdict || 'all';
    verdict.addEventListener('change', () => { state.filters.verdict = verdict.value; state.filters.page = 1; renderReview(); });
    const fileInput = el('input', { type: 'file', accept: 'application/json' });
    fileInput.addEventListener('change', () => restoreFromFile(fileInput.files && fileInput.files[0]));

    host.appendChild(el('div', { class: 'toolbar' },
      el('span', { class: 'pill', text: 'blind mode: on' }),
      reviewer,
      search,
      verdict,
      el('button', { class: 'secondary', text: 'Скачать draft JSON', onclick: downloadDraftJson }),
      el('button', { class: 'secondary', text: 'Скачать review JSON', onclick: downloadReviewJson }),
      el('button', { class: 'secondary', text: 'Скачать review CSV', onclick: downloadReviewCsv }),
      el('button', { class: 'primary', text: 'Скачать ZIP', onclick: downloadReviewZip }),
      fileInput
    ));

    const rows = currentRows();
    const pageSize = Number(state.filters.pageSize || 4);
    const totalPages = Math.max(1, Math.ceil(rows.length / pageSize));
    if (state.filters.page > totalPages) state.filters.page = totalPages;
    const start = (state.filters.page - 1) * pageSize;
    rows.slice(start, start + pageSize).forEach((row) => {
      const rv = state.reviewState[row.pair_id] || {};
      const card = el('article', { class: 'card' });
      card.append(
        el('div', { class: 'toolbar' },
          el('span', { class: 'pill', text: `${row.pair_id}` }),
          el('span', { class: 'pill', text: `match=${row.match_mode || '—'}` }),
          el('span', { class: 'pill', text: `rank α=${row.rank_model_a || '—'}` }),
          el('span', { class: 'pill', text: `rank β=${row.rank_model_b || '—'}` }),
          el('span', { class: 'pill', text: `${row.candidate?.source || '—'} / ${row.candidate?.predicate || '—'} / ${row.candidate?.target || '—'}` })
        )
      );
      card.appendChild(el('div', { class: 'grid cols-2' }, variantNode(row.left_label, row.left_variant), variantNode(row.right_label, row.right_variant)));

      const controls = el('div', { class: 'controls' });
      const pref = bindSelect(el('select', null,
        el('option', { value: '', text: 'preferred variant?' }),
        el('option', { value: 'A', text: 'A' }),
        el('option', { value: 'B', text: 'B' }),
        el('option', { value: 'tie', text: 'tie' })
      ), row.pair_id, 'preferred_variant');
      const betterTemporal = bindSelect(el('select', null,
        el('option', { value: '', text: 'better temporal?' }),
        el('option', { value: 'A', text: 'A' }),
        el('option', { value: 'B', text: 'B' }),
        el('option', { value: 'tie', text: 'tie' })
      ), row.pair_id, 'better_temporal');
      const betterEvidence = bindSelect(el('select', null,
        el('option', { value: '', text: 'better evidence?' }),
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
        el('option', { value: '', text: 'verdict?' }),
        el('option', { value: 'accept', text: 'accept' }),
        el('option', { value: 'needs_revision', text: 'needs_revision' }),
        el('option', { value: 'reject', text: 'reject' })
      ), row.pair_id, 'global_verdict');
      const priority = bindSelect(el('select', null,
        el('option', { value: 'low', text: 'low' }),
        el('option', { value: 'medium', text: 'medium' }),
        el('option', { value: 'high', text: 'high' })
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
    host.appendChild(el('div', { class: 'note', html: `<h2>Сводка blind A/B review</h2><div class="muted">Completed pairs: <b>${summary.completed_pairs}</b> / ${summary.total_pairs}</div>` }));
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


def _load_manifest_and_rows(manifest: Dict[str, Any] | str | Path) -> tuple[Dict[str, Any], List[Dict[str, Any]], Path]:
    if not isinstance(manifest, dict):
        manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
    bundle_dir = Path(str(manifest["bundle_dir"]))
    hypotheses_path = _resolve_bundle_artifact(manifest, bundle_dir, "hypotheses_ranked")
    if hypotheses_path is None:
        raise FileNotFoundError(f"Task 3 hypotheses_ranked artifact not found in manifest for {bundle_dir}")
    rows = _safe_json_load(hypotheses_path) or []
    if not isinstance(rows, list):
        raise ValueError("Task 3 hypotheses_ranked must be a list")
    return manifest, rows, bundle_dir


def build_task3_dual_model_offline_review_package(
    manifest_a: Dict[str, Any] | str | Path,
    manifest_b: Dict[str, Any] | str | Path,
    task_meta: Optional[Dict[str, Any]] = None,
    *,
    output_path: str | Path | None = None,
    top_pairs: Optional[int] = None,
    owner_mapping_path: str | Path | None = None,
    public_manifest_path: str | Path | None = None,
    model_a_descriptor: Optional[Dict[str, Any]] = None,
    model_b_descriptor: Optional[Dict[str, Any]] = None,
) -> DualModelReviewAssets:
    task_meta = task_meta or {}
    manifest_a, rows_a, bundle_dir_a = _load_manifest_and_rows(manifest_a)
    manifest_b, rows_b, bundle_dir_b = _load_manifest_and_rows(manifest_b)

    comparison_dir = Path(output_path).parent if output_path else bundle_dir_a.parent / "dual_local_model_blind_review"
    output = Path(output_path) if output_path else comparison_dir / "expert_review" / "offline_review" / "task3_dual_local_model_review_offline_ab.html"
    output.parent.mkdir(parents=True, exist_ok=True)

    seed_a = json.dumps(model_a_descriptor or manifest_a.get("runtime") or {}, ensure_ascii=False, sort_keys=True) + str(bundle_dir_a)
    seed_b = json.dumps(model_b_descriptor or manifest_b.get("runtime") or {}, ensure_ascii=False, sort_keys=True) + str(bundle_dir_b)
    anon_a = _anon_system_id(seed_a, "alpha")
    anon_b = _anon_system_id(seed_b, "beta")
    label_a = "Hidden model α"
    label_b = "Hidden model β"

    records = _build_records(rows_a, rows_b, anon_a=anon_a, anon_b=anon_b, label_a=label_a, label_b=label_b, top_pairs=top_pairs)

    meta = _default_meta(task_meta, manifest_a)
    meta.update(
        {
            "review_mode": "dual_local_model_blind_ab",
            "generated_at": _utc_now(),
            "anonymous_systems": [
                {"system_id": anon_a, "display_label": label_a},
                {"system_id": anon_b, "display_label": label_b},
            ],
            "owner_key_note": "Model identities are stored only in the owner-side key file and are not included in the expert package.",
        }
    )

    app_data = {"artifact_version": ARTIFACT_VERSION, "meta": meta, "records": records}
    app_json = json.dumps(app_data, ensure_ascii=False).replace("</", "<\\/")
    page_title = html.escape(f"Task 3 dual local blind review — {meta['submission_id'] or meta['topic'] or 'bundle'}")
    html_text = _HTML_TEMPLATE.replace("__APP_DATA__", app_json).replace("__PAGE_TITLE__", page_title)
    output.write_text(html_text, encoding="utf-8")

    comparison_dir = output.parent.parent if output.parent.name == "offline_review" else output.parent
    owner_mapping = {
        "artifact_version": ARTIFACT_VERSION,
        "generated_at": _utc_now(),
        "topic": meta.get("topic") or "",
        "submission_id": meta.get("submission_id") or "",
        "anonymous_systems": [
            {
                "system_id": anon_a,
                "display_label": label_a,
                "role": "base_or_reference_local_model",
                "bundle_dir": str(bundle_dir_a),
                "runtime": manifest_a.get("runtime"),
                "descriptor": model_a_descriptor or {},
            },
            {
                "system_id": anon_b,
                "display_label": label_b,
                "role": "finetuned_local_model",
                "bundle_dir": str(bundle_dir_b),
                "runtime": manifest_b.get("runtime"),
                "descriptor": model_b_descriptor or {},
            },
        ],
        "pair_count": len(records),
        "owner_warning": "Do not share this file with the expert if blind review must be preserved.",
    }
    owner_mapping_path = Path(owner_mapping_path) if owner_mapping_path else comparison_dir / "owner_only" / "task3_dual_local_model_blind_key.json"
    _write_json(owner_mapping_path, owner_mapping)

    public_manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "generated_at": _utc_now(),
        "review_mode": "dual_local_model_blind_ab",
        "topic": meta.get("topic") or "",
        "submission_id": meta.get("submission_id") or "",
        "offline_review_html": str(output),
        "pair_count": len(records),
        "anonymous_systems": meta.get("anonymous_systems") or [],
        "bundled_variants": [
            {"display_label": label_a, "system_id": anon_a},
            {"display_label": label_b, "system_id": anon_b},
        ],
    }
    public_manifest_path = Path(public_manifest_path) if public_manifest_path else comparison_dir / "expert_review" / "task3_dual_local_model_review_manifest.json"
    _write_json(public_manifest_path, public_manifest)

    return DualModelReviewAssets(
        offline_html_path=output,
        owner_mapping_path=owner_mapping_path,
        public_manifest_path=public_manifest_path,
    )


def build_task3_dual_model_expert_bundle(
    manifest_a: Dict[str, Any] | str | Path,
    manifest_b: Dict[str, Any] | str | Path,
    task_meta: Optional[Dict[str, Any]] = None,
    *,
    output_path: str | Path | None = None,
    top_pairs: Optional[int] = None,
    model_a_descriptor: Optional[Dict[str, Any]] = None,
    model_b_descriptor: Optional[Dict[str, Any]] = None,
) -> Path:
    assets = build_task3_dual_model_offline_review_package(
        manifest_a,
        manifest_b,
        task_meta,
        top_pairs=top_pairs,
        model_a_descriptor=model_a_descriptor,
        model_b_descriptor=model_b_descriptor,
    )
    manifest_a, rows_a, bundle_dir_a = _load_manifest_and_rows(manifest_a)
    manifest_b, rows_b, bundle_dir_b = _load_manifest_and_rows(manifest_b)

    output = Path(output_path) if output_path else assets.public_manifest_path.parent / "expert_dual_model_blind_review_bundle.zip"
    output.parent.mkdir(parents=True, exist_ok=True)

    paths: list[tuple[Path, str]] = [
        (assets.offline_html_path, "offline_review/task3_dual_local_model_review_offline_ab.html"),
        (assets.public_manifest_path, "expert_review/task3_dual_local_model_review_manifest.json"),
    ]

    ranked_a = _resolve_bundle_artifact(manifest_a, bundle_dir_a, "hypotheses_ranked")
    ranked_b = _resolve_bundle_artifact(manifest_b, bundle_dir_b, "hypotheses_ranked")
    md_a = _resolve_bundle_artifact(manifest_a, bundle_dir_a, "hypotheses_markdown", "hypotheses_ranked_md", "hypotheses_markdown_path")
    md_b = _resolve_bundle_artifact(manifest_b, bundle_dir_b, "hypotheses_markdown", "hypotheses_ranked_md", "hypotheses_markdown_path")
    if ranked_a is not None:
        paths.append((ranked_a, "variant_alpha/hypotheses_ranked.json"))
    if ranked_b is not None:
        paths.append((ranked_b, "variant_beta/hypotheses_ranked.json"))
    if md_a is not None:
        paths.append((md_a, "variant_alpha/hypotheses_ranked.md"))
    if md_b is not None:
        paths.append((md_b, "variant_beta/hypotheses_ranked.md"))

    readme = (
        "Task 3 dual local model blind review bundle\n"
        "\n"
        "Contents:\n"
        "- offline_review/task3_dual_local_model_review_offline_ab.html\n"
        "- variant_alpha/* and variant_beta/* : anonymized hypothesis outputs\n"
        "- expert_review/task3_dual_local_model_review_manifest.json\n"
        "\n"
        "Important: owner-side identity key is intentionally excluded to preserve blind review.\n"
    )

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("README_EXPERT_REVIEW.txt", readme)
        for path, arcname in paths:
            if path.exists() and path.is_file():
                zf.write(path, arcname=arcname)
    return output


__all__ = [
    "ARTIFACT_VERSION",
    "DualModelReviewAssets",
    "build_task3_dual_model_expert_bundle",
    "build_task3_dual_model_offline_review_package",
]
