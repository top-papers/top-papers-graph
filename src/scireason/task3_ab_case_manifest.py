from __future__ import annotations

import html
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from scireason.task3_dual_model_review import (
    ARTIFACT_VERSION,
    _anon_system_id,
    _candidate_signature,
    _default_meta,
    _load_manifest_and_rows,
    _stable_ab_order,
    _template_assets_for_expert_bundle,
    _utc_now,
    _variant_from_row,
    _write_json,
)

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


DEFAULT_ERROR_TAXONOMY = [
    "missed_visual_fact",
    "wrong_evidence_linkage",
    "needs_time_fix",
    "hallucinated_visual_inference",
]


@dataclass(frozen=True)
class CaseBasedReviewAssets:
    offline_html_path: Path
    owner_mapping_path: Path
    public_manifest_path: Path


def default_case_manifest(
    *,
    topic: str = "",
    submission_id: str = "",
    creator_id: str = "",
    case_count: int = 12,
    cutoff_year: str = "",
) -> dict[str, Any]:
    cases = []
    mm_target = max(1, round(case_count * 0.6))
    temp_target = max(1, round(case_count * 0.25))
    for idx in range(1, case_count + 1):
        if idx <= mm_target:
            stratum = "multimodal_hard"
            focus = ["evidence", "visual_fact"]
            errors = ["missed_visual_fact", "wrong_evidence_linkage"]
        elif idx <= mm_target + temp_target:
            stratum = "temporal_hard"
            focus = ["temporal", "evidence"]
            errors = ["needs_time_fix", "wrong_evidence_linkage"]
        else:
            stratum = "easy_control"
            focus = ["overall"]
            errors = []
        cases.append(
            {
                "case_id": f"CASE-{idx:03d}",
                "enabled": True,
                "primary_endpoint": stratum != "easy_control",
                "stratum": stratum,
                "paper_title": "",
                "paper_id": "",
                "year": "",
                "evidence_kind": "figure_or_table",
                "page_hint": "",
                "creator_prompt": "",
                "creator_rationale": "",
                "review_focus": focus,
                "expected_error_modes": errors,
                "match": {
                    "candidate_signature": "",
                    "candidate_source_contains": "",
                    "candidate_predicate_contains": "",
                    "candidate_target_contains": "",
                    "hypothesis_title_contains": "",
                    "premise_contains": "",
                    "mechanism_contains": "",
                    "time_scope_contains": "",
                    "rank_hint": "",
                },
                "notes": "",
            }
        )
    return {
        "schema_version": "task3_ab_case_manifest_v1",
        "generated_at": _utc_now(),
        "experiment_meta": {
            "topic": topic,
            "submission_id": submission_id,
            "creator_id": creator_id,
            "cutoff_year": cutoff_year,
            "review_goal": "Surface VLM differences on multimodal-hard and temporal-hard Task 3 cases.",
        },
        "strata_targets": {
            "multimodal_hard": sum(1 for x in cases if x["stratum"] == "multimodal_hard"),
            "temporal_hard": sum(1 for x in cases if x["stratum"] == "temporal_hard"),
            "easy_control": sum(1 for x in cases if x["stratum"] == "easy_control"),
        },
        "error_taxonomy": DEFAULT_ERROR_TAXONOMY,
        "cases": cases,
    }


def load_case_manifest(source: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(source, dict):
        payload = source
    else:
        path = Path(str(source))
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path, "r") as zf:
                names = [n for n in zf.namelist() if n.endswith(".json") or n.endswith(".yaml") or n.endswith(".yml")]
                if not names:
                    raise FileNotFoundError("No case manifest found inside ZIP archive")
                name = names[0]
                text = zf.read(name).decode("utf-8")
                if name.endswith((".yaml", ".yml")):
                    if yaml is None:
                        raise RuntimeError("PyYAML is required to read YAML case manifests")
                    payload = yaml.safe_load(text) or {}
                else:
                    payload = json.loads(text)
        else:
            text = path.read_text(encoding="utf-8")
            if path.suffix.lower() in {".yaml", ".yml"}:
                if yaml is None:
                    raise RuntimeError("PyYAML is required to read YAML case manifests")
                payload = yaml.safe_load(text) or {}
            else:
                payload = json.loads(text)
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError("Case manifest must be an object with list field 'cases'")
    return payload


def _norm(text: Any) -> str:
    return " ".join(str(text or "").lower().split())


def _row_fields(row: dict[str, Any]) -> dict[str, str]:
    hyp = row.get("hypothesis") if isinstance(row.get("hypothesis"), dict) else {}
    cand = row.get("candidate") if isinstance(row.get("candidate"), dict) else {}
    return {
        "candidate_signature": _candidate_signature(row),
        "candidate_source": _norm(cand.get("source")),
        "candidate_predicate": _norm(cand.get("predicate")),
        "candidate_target": _norm(cand.get("target")),
        "hypothesis_title": _norm(hyp.get("title") or row.get("title")),
        "premise": _norm(hyp.get("premise") or row.get("premise")),
        "mechanism": _norm(hyp.get("mechanism") or row.get("mechanism")),
        "time_scope": _norm(hyp.get("time_scope") or row.get("time_scope")),
        "rank": str(int(row.get("rank") or 0)),
    }


def _case_match_score(case: dict[str, Any], row: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    match = case.get("match") if isinstance(case.get("match"), dict) else {}
    rf = _row_fields(row)
    hits = []
    score = 0
    requested = 0
    exact = _norm(match.get("candidate_signature"))
    if exact:
        if exact != rf["candidate_signature"]:
            return -1, {"reason": "candidate_signature_mismatch", "hits": []}
        score += 100
        hits.append("candidate_signature")
    checks = [
        ("candidate_source_contains", "candidate_source", 12),
        ("candidate_predicate_contains", "candidate_predicate", 12),
        ("candidate_target_contains", "candidate_target", 12),
        ("hypothesis_title_contains", "hypothesis_title", 10),
        ("premise_contains", "premise", 8),
        ("mechanism_contains", "mechanism", 8),
        ("time_scope_contains", "time_scope", 10),
    ]
    for case_key, row_key, weight in checks:
        want = _norm(match.get(case_key))
        if not want:
            continue
        requested += 1
        if want in rf[row_key]:
            score += weight
            hits.append(case_key)
    rank_hint = str(match.get("rank_hint") or "").strip()
    if rank_hint:
        requested += 1
        try:
            hint = int(rank_hint)
            rank = int(rf["rank"] or 0)
            if rank == hint:
                score += 6
                hits.append("rank_hint_exact")
            elif abs(rank - hint) <= 2:
                score += 3
                hits.append("rank_hint_near")
        except Exception:
            pass
    if requested == 0 and not exact:
        return -1, {"reason": "no_match_fields", "hits": []}
    if not hits:
        return -1, {"reason": "no_positive_hits", "hits": []}
    return score, {"reason": "ok", "hits": hits}


def _select_row_for_case(case: dict[str, Any], rows: Sequence[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    best_row = None
    best_meta = {"score": -1, "reason": "no_rows", "hits": []}
    for row in rows:
        score, meta = _case_match_score(case, row)
        if score > int(best_meta["score"]):
            best_row = row
            best_meta = {"score": score, **meta}
    if int(best_meta["score"]) < 0:
        return None, best_meta
    return best_row, best_meta


def _placeholder_variant(system_id: str, display_label: str) -> dict[str, Any]:
    return {
        "system_id": system_id,
        "system_title": display_label,
        "title": "[No matched output for this case]",
        "premise": "",
        "mechanism": "",
        "time_scope": "",
        "proposed_experiment": "",
        "supporting_evidence": [],
        "score": 0.0,
        "score_label": "final_score",
    }


def _public_case_meta(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": str(case.get("case_id") or ""),
        "stratum": str(case.get("stratum") or ""),
        "paper_title": str(case.get("paper_title") or ""),
        "paper_id": str(case.get("paper_id") or ""),
        "year": str(case.get("year") or ""),
        "evidence_kind": str(case.get("evidence_kind") or ""),
        "page_hint": str(case.get("page_hint") or ""),
        "creator_prompt": str(case.get("creator_prompt") or ""),
        "review_focus": list(case.get("review_focus") or []),
        "expected_error_modes": list(case.get("expected_error_modes") or []),
        "primary_endpoint": bool(case.get("primary_endpoint")),
    }


_AUTHORING_HTML_TEMPLATE = r'''
<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>__PAGE_TITLE__</title>
<style>
body{font-family:Inter,system-ui,sans-serif;background:#f8fafc;color:#0f172a;margin:0}.page{max-width:1200px;margin:0 auto;padding:24px}.hero,.card{background:#fff;border:1px solid #dbe3ef;border-radius:16px;padding:18px;margin-bottom:16px;box-shadow:0 10px 24px rgba(15,23,42,.05)}.toolbar{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}button,.buttonlike{border:1px solid #cbd5e1;border-radius:12px;padding:10px 14px;background:#fff;cursor:pointer;font-weight:600}button.primary{background:#2563eb;color:#fff;border-color:#2563eb}button.success{background:#166534;color:#fff;border-color:#166534}input,textarea,select{width:100%;padding:10px;border:1px solid #cbd5e1;border-radius:10px;font:inherit}textarea{min-height:76px}.grid{display:grid;gap:12px}.cols2{grid-template-columns:repeat(auto-fit,minmax(240px,1fr))}.cols3{grid-template-columns:repeat(auto-fit,minmax(200px,1fr))}.case{border:1px solid #dbe3ef;border-radius:14px;padding:14px;margin-top:12px;background:#fcfdff}.pill{display:inline-flex;padding:6px 10px;border-radius:999px;background:#eff6ff;color:#1d4ed8;font-size:12px;font-weight:700}.muted{color:#475569}
</style></head><body><div class="page"><div class="hero"><h1>Task 3 — creator authoring form</h1><div class="muted">Эту форму заполняет эксперт-создатель набора. Она описывает curated hard cases для будущего blind A/B.</div><div class="toolbar"><button class="primary" id="exportManifest">Скачать filled JSON</button><button id="exportDraft">Скачать draft JSON</button><label class="buttonlike" for="loadDraft">Загрузить JSON</label><input id="loadDraft" type="file" accept="application/json" style="display:none"></div><div id="stats"></div></div><div class="card"><h2>Метаданные</h2><div class="grid cols3" id="meta"></div></div><div class="card"><div class="toolbar"><button id="addCase">Добавить кейс</button></div><div id="cases"></div></div></div>
<script>
const APP=__APP_DATA__;
const STORAGE_KEY=`task3-authoring:${(APP.experiment_meta||{}).submission_id||'draft'}`;
const el=(t,a,...c)=>{const n=document.createElement(t);if(a)Object.entries(a).forEach(([k,v])=>{if(v==null)return;if(k==='class')n.className=v;else if(k==='text')n.textContent=v;else if(k.startsWith('on'))n.addEventListener(k.slice(2),v);else n.setAttribute(k,v)});c.flat().forEach(ch=>{if(ch==null)return;n.appendChild(typeof ch==='string'?document.createTextNode(ch):ch)});return n};
const state=JSON.parse(JSON.stringify(APP));
function save(){try{localStorage.setItem(STORAGE_KEY, JSON.stringify(state));}catch(_){}}
function load(){try{const raw=localStorage.getItem(STORAGE_KEY);if(raw)Object.assign(state, JSON.parse(raw));}catch(_){}}
function download(name, text){const b=new Blob([text],{type:'application/json'});const u=URL.createObjectURL(b);const a=document.createElement('a');a.href=u;a.download=name;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(u),1000)}
function renderStats(){const cases=(state.cases||[]).filter(c=>c.enabled!==false);const by={multimodal_hard:0,temporal_hard:0,easy_control:0};cases.forEach(c=>by[c.stratum]=(by[c.stratum]||0)+1);document.getElementById('stats').innerHTML=`<span class="pill">cases: ${cases.length}</span> <span class="pill">MM: ${by.multimodal_hard||0}</span> <span class="pill">Temporal: ${by.temporal_hard||0}</span> <span class="pill">Easy: ${by.easy_control||0}</span>`}
function renderMeta(){const host=document.getElementById('meta');host.innerHTML='';const meta=state.experiment_meta||(state.experiment_meta={});[['topic','Topic'],['submission_id','Submission ID'],['creator_id','Creator ID'],['cutoff_year','Cutoff year'],['review_goal','Review goal']].forEach(([key,label])=>{const node=key==='review_goal'?el('textarea',{}):el('input',{type:'text'});node.value=meta[key]||'';node.oninput=()=>{meta[key]=node.value;save();renderStats();};host.append(el('label',null,label,node));});}
function caseCard(c,idx){const box=el('div',{class:'case'});const m=c.match||(c.match={});const enabled=el('input',{type:'checkbox'});enabled.checked=!!c.enabled;enabled.onchange=()=>{c.enabled=enabled.checked;save();renderStats();};const primary=el('input',{type:'checkbox'});primary.checked=!!c.primary_endpoint;primary.onchange=()=>{c.primary_endpoint=primary.checked;save();};const stratum=el('select',null,['multimodal_hard','temporal_hard','easy_control'].map(v=>el('option',{value:v,text:v})));stratum.value=c.stratum||'multimodal_hard';stratum.onchange=()=>{c.stratum=stratum.value;save();renderStats();};box.append(el('div',{class:'toolbar'},el('span',{class:'pill',text:c.case_id||`CASE-${String(idx+1).padStart(3,'0')}`}),el('label',null,enabled,' enabled'),el('label',null,primary,' primary'),stratum,el('button',{text:'Удалить',onClick:()=>{state.cases.splice(idx,1);render();save();}})));
const top=el('div',{class:'grid cols3'});[['case_id','Case ID'],['paper_title','Paper title'],['paper_id','Paper ID'],['year','Year'],['evidence_kind','Evidence kind'],['page_hint','Page hint']].forEach(([key,label])=>{const x=el('input',{type:'text',value:c[key]||''});x.oninput=()=>{c[key]=x.value;save();};top.append(el('label',null,label,x));});box.append(top);
['creator_prompt','creator_rationale','notes'].forEach(key=>{const t=el('textarea',{});t.value=c[key]||'';t.oninput=()=>{c[key]=t.value;save();};box.append(el('label',null,key,t));});
const focus=el('input',{type:'text',value:(c.review_focus||[]).join(', ')});focus.oninput=()=>{c.review_focus=focus.value.split(',').map(s=>s.trim()).filter(Boolean);save();};const errs=el('input',{type:'text',value:(c.expected_error_modes||[]).join(', ')});errs.oninput=()=>{c.expected_error_modes=errs.value.split(',').map(s=>s.trim()).filter(Boolean);save();};box.append(el('div',{class:'grid cols2'},el('label',null,'review_focus',focus),el('label',null,'expected_error_modes',errs)));
const match=el('div',{class:'grid cols2'});[['candidate_signature','candidate_signature'],['candidate_source_contains','source contains'],['candidate_predicate_contains','predicate contains'],['candidate_target_contains','target contains'],['hypothesis_title_contains','title contains'],['premise_contains','premise contains'],['mechanism_contains','mechanism contains'],['time_scope_contains','time scope contains'],['rank_hint','rank hint']].forEach(([key,label])=>{const x=el('input',{type:'text',value:m[key]||''});x.oninput=()=>{m[key]=x.value;save();};match.append(el('label',null,label,x));});box.append(el('h3',{text:'Match fields'}),match);return box;}
function renderCases(){const host=document.getElementById('cases');host.innerHTML='';(state.cases||[]).forEach((c,i)=>host.append(caseCard(c,i)));}
function render(){renderMeta();renderCases();renderStats();}
document.getElementById('addCase').onclick=()=>{const idx=(state.cases||[]).length+1;(state.cases||(state.cases=[])).push({case_id:`CASE-${String(idx).padStart(3,'0')}`,enabled:true,primary_endpoint:true,stratum:'multimodal_hard',paper_title:'',paper_id:'',year:'',evidence_kind:'figure_or_table',page_hint:'',creator_prompt:'',creator_rationale:'',review_focus:['evidence'],expected_error_modes:['missed_visual_fact'],match:{candidate_signature:'',candidate_source_contains:'',candidate_predicate_contains:'',candidate_target_contains:'',hypothesis_title_contains:'',premise_contains:'',mechanism_contains:'',time_scope_contains:'',rank_hint:''},notes:''});render();save();};
document.getElementById('exportManifest').onclick=()=>download('task3_ab_case_manifest.filled.json', JSON.stringify(state, null, 2));
document.getElementById('exportDraft').onclick=()=>download('task3_ab_case_manifest.draft.json', JSON.stringify(state, null, 2));
document.getElementById('loadDraft').addEventListener('change',ev=>{const f=ev.target.files&&ev.target.files[0];if(!f)return;const r=new FileReader();r.onload=()=>{Object.assign(state, JSON.parse(String(r.result||'{}')));render();save();};r.readAsText(f,'utf-8');});
load();render();save();
</script></body></html>
'''


_CASE_REVIEW_HTML_TEMPLATE = r'''
<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>__PAGE_TITLE__</title>
<style>
body{margin:0;font-family:Inter,system-ui,sans-serif;background:#f8fafc;color:#0f172a}.page{max-width:1280px;margin:0 auto;padding:24px}.hero,.card{background:#fff;border:1px solid #dbe3ef;border-radius:18px;padding:18px;margin-bottom:16px;box-shadow:0 10px 24px rgba(15,23,42,.05)}.toolbar{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}.controls{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-top:12px}button,.buttonlike{border:1px solid #cbd5e1;border-radius:12px;padding:10px 14px;background:#fff;cursor:pointer;font-weight:600}button.primary{background:#2563eb;color:#fff;border-color:#2563eb}select,input,textarea{width:100%;padding:10px;border:1px solid #cbd5e1;border-radius:10px;font:inherit}textarea{min-height:80px}.pill{display:inline-flex;padding:6px 10px;border-radius:999px;background:#eff6ff;color:#1d4ed8;font-size:12px;font-weight:700}.muted{color:#475569}.grid2{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(340px,1fr))}.variant{border:1px solid #dbe3ef;border-radius:14px;padding:14px;background:#fcfdff}.kv{display:grid;grid-template-columns:140px 1fr;gap:6px 10px}.key{font-weight:700;color:#334155}.casebox{background:#f8fafc;border:1px solid #dbe3ef;border-radius:12px;padding:12px;margin-bottom:12px}ul{margin:8px 0 0 18px}
</style></head><body><div class="page"><div class="hero"><h1>Task 3 — case-based blind A/B review</h1><div class="muted">Эксперт-участник сравнивает Variant A/B по заранее подготовленным кейсам. Ключ соответствия моделей хранится отдельно.</div><div class="toolbar"><button class="primary" id="exportJson">Скачать JSON</button><button id="exportCsv">Скачать CSV</button><label class="buttonlike" for="loadDraft">Загрузить draft JSON</label><input id="loadDraft" type="file" accept="application/json" style="display:none"></div><div id="stats"></div></div><div id="content"></div></div>
<script>
const APP=__APP_DATA__;
const STORAGE_KEY=`task3-case-review:${(APP.meta||{}).submission_id||'bundle'}`;
const el=(t,a,...c)=>{const n=document.createElement(t);if(a)Object.entries(a).forEach(([k,v])=>{if(v==null)return;if(k==='class')n.className=v;else if(k==='text')n.textContent=v;else if(k.startsWith('on'))n.addEventListener(k.slice(2),v);else n.setAttribute(k,v)});c.flat().forEach(ch=>{if(ch==null)return;n.appendChild(typeof ch==='string'?document.createTextNode(ch):ch)});return n};
const state={reviewer_id:(APP.meta||{}).reviewer_default||'',reviewState:Object.fromEntries((APP.records||[]).map(r=>[r.pair_id,JSON.parse(JSON.stringify(r.review_defaults||{}))]))};
function save(){try{localStorage.setItem(STORAGE_KEY, JSON.stringify(state));}catch(_){}}
function load(){try{const raw=localStorage.getItem(STORAGE_KEY);if(raw){const p=JSON.parse(raw);if(p.reviewer_id)state.reviewer_id=p.reviewer_id;Object.keys(p.reviewState||{}).forEach(k=>{if(state.reviewState[k])Object.assign(state.reviewState[k],p.reviewState[k]);});}}catch(_){}}
function download(name,text,mime){const b=new Blob([text],{type:mime||'application/json'});const u=URL.createObjectURL(b);const a=document.createElement('a');a.href=u;a.download=name;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(u),1000)}
function sideSystem(r,val){if(!val)return '';if(val==='A')return r.left_truth;if(val==='B')return r.right_truth;return val;}
function rows(){return (APP.records||[]).map(r=>{const rv=state.reviewState[r.pair_id]||{};return {pair_id:r.pair_id,reviewer_id:state.reviewer_id||'',stratum:(r.case_meta||{}).stratum||'',paper_title:(r.case_meta||{}).paper_title||'',preferred_variant:rv.preferred_variant||'',preferred_system:sideSystem(r,rv.preferred_variant),better_evidence:rv.better_evidence||'',better_evidence_system:sideSystem(r,rv.better_evidence),better_temporal:rv.better_temporal||'',better_temporal_system:sideSystem(r,rv.better_temporal),missed_visual_fact_by:rv.missed_visual_fact_by||'',wrong_evidence_linkage_by:rv.wrong_evidence_linkage_by||'',needs_time_fix_by:rv.needs_time_fix_by||'',hallucinated_visual_inference_by:rv.hallucinated_visual_inference_by||'',confidence:rv.confidence||'',comments:rv.comments||'',displayed_variant_a_system:r.left_truth,displayed_variant_b_system:r.right_truth,rank_model_a:r.rank_model_a||'',rank_model_b:r.rank_model_b||''};});}
function stats(){const rs=rows();return {cases:rs.length,done:rs.filter(r=>r.preferred_variant||r.comments).length};}
function renderStats(){const s=stats();document.getElementById('stats').innerHTML=`<span class="pill">topic: ${(APP.meta||{}).topic||'—'}</span> <span class="pill">cases: ${s.cases}</span> <span class="pill">completed: ${s.done}</span>`}
function evList(items){if(!items||!items.length)return el('div',{class:'muted',text:'Нет evidence snippets.'});return el('ul',null,items.map(it=>el('li',{text:`${it.page?`p.${it.page} `:''}${it.locator?`[${it.locator}] `:''}${String(it.text_snippet||'').slice(0,200)}`})));}
function variant(label,v){return el('div',{class:'variant'},el('div',{class:'pill',text:`Variant ${label}`}),el('h3',{text:v.title||'(без названия)'}),el('div',{class:'kv'},el('div',{class:'key',text:'Premise'}),el('div',{text:v.premise||''}),el('div',{class:'key',text:'Mechanism'}),el('div',{text:v.mechanism||''}),el('div',{class:'key',text:'Time scope'}),el('div',{text:v.time_scope||''}),el('div',{class:'key',text:'Experiment'}),el('div',{text:v.proposed_experiment||''}),el('div',{class:'key',text:v.score_label||'score'}),el('div',{text:String(v.score??'')})),evList(v.supporting_evidence||[]));}
function select3(pairId,field,label){const s=el('select',null,el('option',{value:'',text:`${label}?`}),el('option',{value:'A',text:'A'}),el('option',{value:'B',text:'B'}),el('option',{value:'tie',text:'tie'}));s.value=(state.reviewState[pairId]||{})[field]||'';s.onchange=()=>{state.reviewState[pairId][field]=s.value;save();renderStats();};return el('label',null,label,s);}
function select4(pairId,field,label){const s=el('select',null,el('option',{value:'',text:`${label}?`}),el('option',{value:'none',text:'none'}),el('option',{value:'A',text:'A'}),el('option',{value:'B',text:'B'}),el('option',{value:'both',text:'both'}));s.value=(state.reviewState[pairId]||{})[field]||'';s.onchange=()=>{state.reviewState[pairId][field]=s.value;save();renderStats();};return el('label',null,label,s);}
function render(){const host=document.getElementById('content');host.innerHTML='';const rev=el('input',{type:'text',value:state.reviewer_id||'',placeholder:'reviewer_id'});rev.oninput=()=>{state.reviewer_id=rev.value;save();};host.append(el('div',{class:'card'},el('div',{class:'toolbar'},el('span',{class:'pill',text:'blind mode'}),rev)));
(APP.records||[]).forEach(r=>{const m=r.case_meta||{};const rv=state.reviewState[r.pair_id]||{};const conf=el('select',null,[1,2,3,4,5].map(n=>el('option',{value:String(n),text:`confidence ${n}`})));conf.value=rv.confidence||'3';conf.onchange=()=>{state.reviewState[r.pair_id].confidence=conf.value;save();};const comments=el('textarea',{});comments.value=rv.comments||'';comments.oninput=()=>{state.reviewState[r.pair_id].comments=comments.value;save();};host.append(el('div',{class:'card'},el('div',{class:'casebox'},el('div',{class:'toolbar'},el('span',{class:'pill',text:r.pair_id}),el('span',{class:'pill',text:m.stratum||''}),m.primary_endpoint?el('span',{class:'pill',text:'primary'}):null),el('div',{html:`<b>${m.paper_title||'Untitled paper'}</b> ${m.paper_id?`<span class="muted">(${m.paper_id})</span>`:''}`}),m.page_hint?el('div',{class:'muted',text:`Page hint: ${m.page_hint}`}):null,m.creator_prompt?el('div',{class:'muted',text:`Focus: ${m.creator_prompt}`}):null,(m.review_focus||[]).length?el('div',{class:'muted',text:`Review focus: ${(m.review_focus||[]).join(', ')}`}):null),el('div',{class:'grid2'},variant('A',r.left_variant),variant('B',r.right_variant)),el('div',{class:'controls'},select3(r.pair_id,'preferred_variant','Preferred variant'),select3(r.pair_id,'better_evidence','Better evidence'),select3(r.pair_id,'better_temporal','Better temporal grounding'),select4(r.pair_id,'missed_visual_fact_by','Missed visual fact by'),select4(r.pair_id,'wrong_evidence_linkage_by','Wrong evidence linkage by'),select4(r.pair_id,'needs_time_fix_by','Needs time fix by'),select4(r.pair_id,'hallucinated_visual_inference_by','Hallucinated visual inference by'),el('label',null,'Confidence',conf),el('label',{style:'grid-column:1/-1;'},'Comments',comments))));});renderStats();}
document.getElementById('exportJson').onclick=()=>download('task3_case_based_ab_results.json',JSON.stringify({summary:stats(),rows:rows(),draft:state},null,2),'application/json');
document.getElementById('exportCsv').onclick=()=>{const cols=['pair_id','reviewer_id','stratum','paper_title','preferred_variant','preferred_system','better_evidence','better_evidence_system','better_temporal','better_temporal_system','missed_visual_fact_by','wrong_evidence_linkage_by','needs_time_fix_by','hallucinated_visual_inference_by','confidence','comments','displayed_variant_a_system','displayed_variant_b_system','rank_model_a','rank_model_b'];const data=rows();const esc=t=>{t=t==null?'':String(t);return /[",\n]/.test(t)?'"'+t.replace(/"/g,'""')+'"':t};const csv=[cols.join(',')].concat(data.map(r=>cols.map(c=>esc(r[c])).join(','))).join('\n');download('task3_case_based_ab_results.csv',csv,'text/csv');};
document.getElementById('loadDraft').addEventListener('change',ev=>{const f=ev.target.files&&ev.target.files[0];if(!f)return;const rd=new FileReader();rd.onload=()=>{const p=JSON.parse(String(rd.result||'{}'));if(p.draft){Object.assign(state,p.draft);}else Object.assign(state,p);render();save();};rd.readAsText(f,'utf-8');});
load();render();save();
</script></body></html>
'''


def build_task3_ab_authoring_bundle(*, output_dir: str | Path, seed_manifest: dict[str, Any] | None = None) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = seed_manifest or default_case_manifest()
    template_path = output_dir / "task3_ab_case_manifest.template.json"
    _write_json(template_path, manifest)
    checklist_path = output_dir / "task3_ab_creator_checklist.md"
    checklist_path.write_text(
        "# Checklist for Task 3 A/B test-set creator\n\n"
        "- make sure each case has at least one match field;\n"
        "- bias the set toward multimodal_hard and temporal_hard;\n"
        "- do not reveal the expected winner;\n"
        "- keep easy_control as a minority;\n"
        "- export the filled JSON and pass it to the CLI runner.\n",
        encoding="utf-8",
    )
    html_path = output_dir / "task3_ab_creator_offline_form.html"
    app_json = json.dumps(manifest, ensure_ascii=False).replace("</", "<\\/")
    html_path.write_text(
        _AUTHORING_HTML_TEMPLATE.replace("__APP_DATA__", app_json).replace("__PAGE_TITLE__", html.escape("Task 3 AB creator form")),
        encoding="utf-8",
    )
    readme_path = output_dir / "README.txt"
    readme_path.write_text(
        "1) Open task3_ab_creator_offline_form.html in a browser.\n"
        "2) Fill cases and export task3_ab_case_manifest.filled.json.\n"
        "3) Use that JSON as --case-manifest for the case-based blind A/B runner.\n",
        encoding="utf-8",
    )
    bundle_zip = output_dir / "task3_ab_creator_bundle.zip"
    with zipfile.ZipFile(bundle_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in [template_path, html_path, checklist_path, readme_path]:
            zf.write(path, arcname=path.name)
    return {
        "manifest_template": template_path,
        "offline_form": html_path,
        "checklist": checklist_path,
        "bundle_zip": bundle_zip,
    }


def build_task3_case_based_blind_review_package(
    manifest_a: Dict[str, Any] | str | Path,
    manifest_b: Dict[str, Any] | str | Path,
    case_manifest: Dict[str, Any] | str | Path,
    task_meta: Optional[Dict[str, Any]] = None,
    *,
    output_path: str | Path | None = None,
    owner_mapping_path: str | Path | None = None,
    public_manifest_path: str | Path | None = None,
    model_a_descriptor: Optional[Dict[str, Any]] = None,
    model_b_descriptor: Optional[Dict[str, Any]] = None,
    include_disabled_cases: bool = False,
) -> CaseBasedReviewAssets:
    task_meta = task_meta or {}
    case_payload = load_case_manifest(case_manifest)
    manifest_a, rows_a, bundle_dir_a = _load_manifest_and_rows(manifest_a)
    manifest_b, rows_b, bundle_dir_b = _load_manifest_and_rows(manifest_b)

    comparison_dir = Path(output_path).parent if output_path else bundle_dir_a.parent / "case_based_blind_review"
    output = Path(output_path) if output_path else comparison_dir / "expert_review" / "offline_review" / "task3_case_based_blind_review.html"
    output.parent.mkdir(parents=True, exist_ok=True)

    seed_a = json.dumps(model_a_descriptor or manifest_a.get("runtime") or {}, ensure_ascii=False, sort_keys=True) + str(bundle_dir_a)
    seed_b = json.dumps(model_b_descriptor or manifest_b.get("runtime") or {}, ensure_ascii=False, sort_keys=True) + str(bundle_dir_b)
    anon_a = _anon_system_id(seed_a, "alpha")
    anon_b = _anon_system_id(seed_b, "beta")
    label_a = "Скрытая модель α"
    label_b = "Скрытая модель β"

    records = []
    owner_matches = []
    cases = [c for c in case_payload.get("cases") or [] if include_disabled_cases or c.get("enabled", True)]
    for idx, case in enumerate(cases, start=1):
        row_a, info_a = _select_row_for_case(case, rows_a)
        row_b, info_b = _select_row_for_case(case, rows_b)
        variant_a = _variant_from_row(row_a, system_id=anon_a, display_label=label_a) if row_a else _placeholder_variant(anon_a, label_a)
        variant_b = _variant_from_row(row_b, system_id=anon_b, display_label=label_b) if row_b else _placeholder_variant(anon_b, label_b)
        show_a_left = _stable_ab_order(f"{case.get('case_id') or idx}:{idx}")
        left_variant = variant_a if show_a_left else variant_b
        right_variant = variant_b if show_a_left else variant_a
        records.append(
            {
                "pair_id": str(case.get("case_id") or f"CASE-{idx:03d}"),
                "rank": idx,
                "rank_model_a": int((row_a or {}).get("rank") or 0) if row_a else None,
                "rank_model_b": int((row_b or {}).get("rank") or 0) if row_b else None,
                "match_mode": "case_manifest",
                "case_meta": _public_case_meta(case),
                "left_variant": left_variant,
                "right_variant": right_variant,
                "left_truth": left_variant["system_id"],
                "right_truth": right_variant["system_id"],
                "left_label": "A",
                "right_label": "B",
                "review_defaults": {
                    "preferred_variant": "",
                    "better_evidence": "",
                    "better_temporal": "",
                    "missed_visual_fact_by": "",
                    "wrong_evidence_linkage_by": "",
                    "needs_time_fix_by": "",
                    "hallucinated_visual_inference_by": "",
                    "confidence": "3",
                    "comments": "",
                },
            }
        )
        owner_matches.append(
            {
                "case_id": case.get("case_id") or f"CASE-{idx:03d}",
                "case_meta": case,
                "model_a_match": {
                    "score": info_a.get("score"),
                    "hits": info_a.get("hits"),
                    "reason": info_a.get("reason"),
                    "rank": (row_a or {}).get("rank"),
                    "candidate_signature": _candidate_signature(row_a) if row_a else None,
                    "bundle_dir": str(bundle_dir_a),
                },
                "model_b_match": {
                    "score": info_b.get("score"),
                    "hits": info_b.get("hits"),
                    "reason": info_b.get("reason"),
                    "rank": (row_b or {}).get("rank"),
                    "candidate_signature": _candidate_signature(row_b) if row_b else None,
                    "bundle_dir": str(bundle_dir_b),
                },
            }
        )

    meta = _default_meta(task_meta, manifest_a)
    creator_meta = case_payload.get("experiment_meta") if isinstance(case_payload.get("experiment_meta"), dict) else {}
    if creator_meta:
        meta["topic"] = str(creator_meta.get("topic") or meta.get("topic") or "")
        meta["submission_id"] = str(creator_meta.get("submission_id") or meta.get("submission_id") or "")
    meta.update(
        {
            "review_mode": "task3_case_based_blind_ab",
            "generated_at": _utc_now(),
            "anonymous_systems": [
                {"system_id": anon_a, "display_label": label_a},
                {"system_id": anon_b, "display_label": label_b},
            ],
            "creator_summary": {
                "creator_id": str(creator_meta.get("creator_id") or ""),
                "case_count": len(cases),
            },
            "owner_key_note": "Identities and matching details live only in the owner key.",
        }
    )

    app_data = {"artifact_version": ARTIFACT_VERSION, "meta": meta, "records": records}
    app_json = json.dumps(app_data, ensure_ascii=False).replace("</", "<\\/")
    output.write_text(
        _CASE_REVIEW_HTML_TEMPLATE.replace("__APP_DATA__", app_json).replace("__PAGE_TITLE__", html.escape("Task 3 case-based blind review")),
        encoding="utf-8",
    )

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
        "matching": owner_matches,
        "owner_warning": "Do not share this file with the blind-review participant.",
    }
    owner_mapping_path = Path(owner_mapping_path) if owner_mapping_path else comparison_dir / "owner_only" / "task3_case_based_blind_key.json"
    _write_json(owner_mapping_path, owner_mapping)

    public_manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "generated_at": _utc_now(),
        "review_mode": "task3_case_based_blind_ab",
        "topic": meta.get("topic") or "",
        "submission_id": meta.get("submission_id") or "",
        "offline_review_html": str(output),
        "case_count": len(records),
        "anonymous_systems": meta.get("anonymous_systems") or [],
        "creator_summary": meta.get("creator_summary") or {},
    }
    public_manifest_path = Path(public_manifest_path) if public_manifest_path else comparison_dir / "expert_review" / "task3_case_based_blind_review_manifest.json"
    _write_json(public_manifest_path, public_manifest)
    return CaseBasedReviewAssets(output, owner_mapping_path, public_manifest_path)


def build_task3_case_based_expert_bundle(
    manifest_a: Dict[str, Any] | str | Path,
    manifest_b: Dict[str, Any] | str | Path,
    case_manifest: Dict[str, Any] | str | Path,
    task_meta: Optional[Dict[str, Any]] = None,
    *,
    output_path: str | Path | None = None,
    model_a_descriptor: Optional[Dict[str, Any]] = None,
    model_b_descriptor: Optional[Dict[str, Any]] = None,
) -> Path:
    assets = build_task3_case_based_blind_review_package(
        manifest_a,
        manifest_b,
        case_manifest,
        task_meta,
        model_a_descriptor=model_a_descriptor,
        model_b_descriptor=model_b_descriptor,
    )
    output = Path(output_path) if output_path else assets.public_manifest_path.parent / "expert_case_based_blind_review_bundle.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    items = [
        (assets.offline_html_path, "offline_review/task3_case_based_blind_review.html"),
        (assets.public_manifest_path, "expert_review/task3_case_based_blind_review_manifest.json"),
    ]
    case_path = Path(case_manifest) if not isinstance(case_manifest, dict) else None
    if case_path and case_path.exists():
        items.append((case_path, f"creator_inputs/{case_path.name}"))
    items.extend(_template_assets_for_expert_bundle())
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src, arc in items:
            if src.exists() and src.is_file():
                zf.write(src, arcname=arc)
        zf.writestr(
            "README.txt",
            "Case-based blind review package for Task 3.\nOpen offline_review/task3_case_based_blind_review.html and export the filled JSON or CSV.\n",
        )
    return output
