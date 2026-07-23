# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Generate a self-contained, offline workspace for remediation curators."""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from .paths import resolve_dataset_file
from .remediation import (
    RemediationError,
    _atomic_copy,
    _atomic_write,
    _prepare_input_protection,
    _queue_material,
    _separate_output_target,
    _sha256_bytes,
    _sha256_file,
    _trees_identical,
    _verify_queue_workspace,
)


CURATOR_ARTIFACT_VERSION = 1
CURATOR_HTML = "curator.html"
CURATOR_MANIFEST = "workspace_manifest.json"
_IMAGE_SUFFIX_RE = re.compile(r"\.(?:avif|bmp|gif|jpe?g|png|tiff?|webp)$", re.IGNORECASE)
_EDIT_STRING_KEYS = frozenset(
    {
        "disposition",
        "exclusion_reason",
        "reviewed_by",
        "notes",
        "sample_id",
        "paper_id",
        "stratum",
        "prompt",
        "system_instruction",
        "source_document_id",
        "creator_group_id",
        "gold_answer",
        "evidence_used",
        "visual_facts",
        "temporal_facts",
        "uncertainty",
        "missing_evidence",
        "criteria",
        "adjudicators",
    }
)
_EDIT_BOOL_KEYS = frozenset({"gold_enabled", "independent_attestation"})
_IMAGE_EDIT_KEYS = frozenset(
    {
        "image_path",
        "sha256",
        "page",
        "locator",
        "source_url",
        "license",
        "citation",
        "verified_by",
    }
)


def _blank_edit() -> dict[str, Any]:
    return {
        **{key: "" for key in _EDIT_STRING_KEYS},
        "stratum": "multimodal_hard",
        "images": [],
        "gold_enabled": False,
        "independent_attestation": False,
    }


def _validate_draft_edit(value: Any, task_id: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != (
        _EDIT_STRING_KEYS | _EDIT_BOOL_KEYS | {"images"}
    ):
        raise ValueError(f"draft task {task_id} has invalid edit fields")
    if any(not isinstance(value[key], str) for key in _EDIT_STRING_KEYS):
        raise ValueError(f"draft task {task_id} has a non-string edit field")
    if any(not isinstance(value[key], bool) for key in _EDIT_BOOL_KEYS):
        raise ValueError(f"draft task {task_id} has a non-boolean edit field")
    images = value["images"]
    if not isinstance(images, list):
        raise ValueError(f"draft task {task_id} images must be an array")
    for index, image in enumerate(images):
        if (
            not isinstance(image, dict)
            or set(image) != _IMAGE_EDIT_KEYS
            or any(not isinstance(image[key], str) for key in _IMAGE_EDIT_KEYS)
        ):
            raise ValueError(f"draft task {task_id} image {index} is malformed")
    return copy.deepcopy(value)


def _merge_draft_edits(
    current: Mapping[str, Any],
    envelope: Any,
    known_task_ids: set[str],
    queue_fingerprint: str,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Validate and atomically merge task-level draft edits without mutating inputs."""

    if not isinstance(envelope, dict) or set(envelope) != {
        "artifact_version",
        "queue_fingerprint",
        "edits",
    }:
        raise ValueError("draft envelope fields are invalid")
    if (
        type(envelope["artifact_version"]) is not int
        or envelope["artifact_version"] != CURATOR_ARTIFACT_VERSION
    ):
        raise ValueError("draft artifact_version is unsupported")
    if envelope["queue_fingerprint"] != queue_fingerprint:
        raise ValueError("draft belongs to another queue")
    incoming = envelope["edits"]
    if not isinstance(incoming, dict):
        raise ValueError("draft edits must be an object keyed by task_id")
    if any(not isinstance(task_id, str) for task_id in incoming):
        raise ValueError("draft edits keys must be task_id strings")
    unknown = sorted(set(incoming) - known_task_ids)
    if unknown:
        raise ValueError(f"draft references unknown task_id {unknown[0]}")

    validated = {
        task_id: _validate_draft_edit(value, task_id) for task_id, value in incoming.items()
    }
    merged = {
        task_id: _validate_draft_edit(current.get(task_id, _blank_edit()), task_id)
        for task_id in known_task_ids
    }
    blank = _blank_edit()
    summary = {"merged": 0, "equal": 0, "blank_ignored": 0}
    for task_id, incoming_edit in validated.items():
        current_edit = merged[task_id]
        if incoming_edit == blank:
            summary["blank_ignored"] += 1
        elif current_edit == blank:
            merged[task_id] = incoming_edit
            summary["merged"] += 1
        elif current_edit == incoming_edit:
            summary["equal"] += 1
        else:
            raise ValueError(f"draft merge conflict for task_id {task_id}")
    return merged, summary


def _embedded_json(value: Any) -> str:
    text = json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"))
    return (
        text.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _workspace_target(output_dir: str | Path, queue_root: Path) -> Path:
    return _separate_output_target(output_dir, protected_directories=(queue_root,))


def _source_inventory(material: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    dataset_root = Path(material["source"]["resolved"]["dataset_root"])
    references: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_by_hash: dict[str, Path] = {}
    suffix_by_hash: dict[str, str] = {}
    for task in material["tasks"]:
        for record in task["audited_image_hashes"]:
            digest = record["sha256"]
            relative = record["path"]
            parts = PurePosixPath(relative).parts
            try:
                source = resolve_dataset_file(dataset_root, Path(*parts))
            except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
                raise RemediationError(f"audited source image is unavailable: {relative}") from exc
            actual, size = _sha256_file(source)
            if actual != digest or size != record["size_bytes"]:
                raise RemediationError(f"audited source image changed: {relative}")
            previous = source_by_hash.get(digest)
            if previous is not None and _sha256_file(previous)[0] != actual:
                raise RemediationError(f"conflicting audited source images for SHA256 {digest}")
            source_by_hash[digest] = source
            suffix = Path(relative).suffix.lower()
            if not _IMAGE_SUFFIX_RE.fullmatch(suffix):
                suffix = ".bin"
            suffix_by_hash[digest] = min(suffix_by_hash.get(digest, suffix), suffix)
            reference = {
                "task_id": task["task_id"],
                "source_row_index": task["source_row_index"],
                "source_path": relative,
            }
            if reference not in references[digest]:
                references[digest].append(reference)

    inventory: list[dict[str, Any]] = []
    copies: dict[str, Path] = {}
    for digest in sorted(source_by_hash):
        relative = f"source_images/{digest}{suffix_by_hash[digest]}"
        source = source_by_hash[digest]
        _, size = _sha256_file(source)
        inventory.append(
            {
                "path": relative,
                "sha256": digest,
                "size_bytes": size,
                "source_references": sorted(
                    references[digest],
                    key=lambda item: (
                        item["source_row_index"],
                        item["source_path"],
                        item["task_id"],
                    ),
                ),
            }
        )
        copies[relative] = source
    return inventory, copies


def _browser_payload(
    material: Mapping[str, Any],
    verified_queue: Mapping[str, Any],
    inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    preview_by_hash = {entry["sha256"]: entry["path"] for entry in inventory}
    tasks = []
    for task, template in zip(material["tasks"], material["templates"], strict=True):
        tasks.append(
            {
                "task": {
                    **task,
                    "audited_image_hashes": [
                        {**record, "preview_path": preview_by_hash[record["sha256"]]}
                        for record in task["audited_image_hashes"]
                    ],
                },
                "binding": {
                    key: value
                    for key, value in template.items()
                    if key
                    not in {
                        "status",
                        "disposition",
                        "exclusion_reason",
                        "benchmark_row",
                        "provenance_row",
                        "reviewed_by",
                        "independent_attestation",
                        "notes",
                    }
                },
            }
        )
    return {
        "artifact_version": CURATOR_ARTIFACT_VERSION,
        "queue_fingerprint": verified_queue["queue_fingerprint"],
        "policy": verified_queue["policy"],
        "task_count": verified_queue["task_count"],
        "tasks": tasks,
    }


_STYLE = r"""
:root{color-scheme:light;--ink:#172026;--muted:#617079;--paper:#f5f1e8;--card:#fffdf7;--line:#d7cdbc;--accent:#9d3c27;--good:#276749;--bad:#9b2c2c;--warn:#ffd166}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.45 system-ui,sans-serif}header{position:sticky;top:0;z-index:2;padding:14px 4vw;background:#172026;color:white;box-shadow:0 2px 8px #0004}h1{font:700 22px Georgia,serif;margin:0 0 10px}.toolbar,.nav,.row{display:flex;gap:9px;align-items:center;flex-wrap:wrap}input,textarea,select,button{font:inherit}input,textarea,select{width:100%;padding:8px;border:1px solid var(--line);border-radius:5px;background:white}textarea{min-height:70px;resize:vertical}button{border:1px solid #806f5d;border-radius:5px;padding:8px 12px;background:#fff;cursor:pointer}button.primary{background:var(--accent);border-color:var(--accent);color:white}button.danger{color:var(--bad)}header input,header select{width:auto;min-width:150px}main{max-width:1400px;margin:20px auto;padding:0 3vw 80px}.grid{display:grid;grid-template-columns:minmax(280px,38%) 1fr;gap:18px}.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:16px;box-shadow:0 2px 5px #563c2312}.source{position:sticky;top:145px;max-height:calc(100vh - 165px);overflow:auto}h2,h3{font-family:Georgia,serif}h2{margin-top:0;font-size:20px}h3{border-bottom:1px solid var(--line);padding-bottom:5px}.meta{color:var(--muted);font-size:13px}.codes{display:flex;gap:5px;flex-wrap:wrap}.code{background:#eee4d5;padding:2px 6px;border-radius:10px;font-size:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2eee5;padding:10px;border-radius:5px;font-size:12px}.preview{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:9px}.preview img{width:100%;max-height:230px;object-fit:contain;background:#eee;border:1px solid var(--line)}label{display:block;font-weight:650;margin:11px 0 4px}.two{display:grid;grid-template-columns:1fr 1fr;gap:10px}.image-form{border:1px solid var(--line);padding:11px;margin:10px 0;border-radius:6px}.error{white-space:pre-wrap;color:var(--bad);font-weight:650}.ok{color:var(--good);font-weight:650}.warning{color:var(--warn);font-weight:700}.hidden{display:none!important}.progress{font-weight:700}.binding{font-family:monospace;font-size:11px;overflow-wrap:anywhere}.spacer{flex:1}@media(max-width:800px){header{position:static}.grid{grid-template-columns:1fr}.source{position:static;max-height:none}.two{grid-template-columns:1fr}header input,header select{width:100%}}
"""


_EXPERT_FORM_STYLE = r"""
.expert-guide{margin:0 0 20px;padding:16px 18px;border:1px solid #9bc3a8;border-left:5px solid var(--good);border-radius:8px;background:#f1f8f3}.expert-guide h3{margin:0 0 8px;border:0;padding:0}.expert-guide ol{margin:8px 0 0;padding-left:22px}.expert-guide li{margin:5px 0}.field-group{margin:13px 0}.field-group label{margin-bottom:5px;font-size:15px}.field-help{margin:5px 0 0;color:var(--muted);font-size:13px;line-height:1.4}.required-mark{color:var(--bad);font-weight:800}.decision-block,.confirmation-box{margin:16px 0;padding:15px;border-radius:8px}.decision-block{border:1px solid #c9b58d;background:#fbf6ea}.confirmation-box{border:1px solid #9bc3a8;background:#f5faf6}.confirmation-box h3{margin-top:0}.attestation-box{display:flex;align-items:flex-start;gap:9px;margin:14px 0;padding:12px;border:1px solid #9bc3a8;border-radius:6px;background:white}.attestation-box input{flex:0 0 auto;margin-top:4px}.technical-details,.source-details{margin:14px 0;border:1px solid var(--line);border-radius:6px;background:#faf7f0}.technical-details summary,.source-details summary{padding:10px 12px;cursor:pointer;font-weight:650}.technical-details>.binding,.source-details>pre{margin:0 12px 12px}.source-summary{margin:14px 0;padding:12px;border:1px solid var(--line);border-radius:7px;background:#faf7f0}.source-summary dt{margin-top:8px;color:var(--muted);font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.03em}.source-summary dd{margin:2px 0;white-space:pre-wrap;overflow-wrap:anywhere}.image-form{padding:16px;background:#fcfaf4}.image-form>.row:first-child{font:700 17px Georgia,serif}.image-form label{margin-top:14px}.image-form input:focus,.image-form textarea:focus,.field-group input:focus,.field-group textarea:focus,.field-group select:focus{outline:3px solid #d9ad78;outline-offset:1px;border-color:var(--accent)}#form>h3{margin-top:24px}.simple-intro{margin-top:-6px;color:var(--muted)}header .meta{color:#d7e0e4}.toolbar label{margin:0;color:inherit}.toolbar label input{margin-left:6px}.validation-heading{margin-top:22px}@media(max-width:850px){.expert-guide,.decision-block,.confirmation-box{padding:12px}.field-help{font-size:14px}.toolbar label input{display:block;margin:6px 0 0}}
"""

_STYLE += _EXPERT_FORM_STYLE


_SCRIPT = r"""
"use strict";
const DATA=JSON.parse(document.getElementById("workspace-data").textContent);
const KEY="vlm-curator:"+DATA.queue_fingerprint;
const MAX_DRAFT_BYTES=32*1024*1024;
const FIELDS=["artifact_version","queue_fingerprint","task_id","prepare_manifest_sha256","source_benchmark_sha256","source_provenance_sha256","source_audit_sha256","source_row_index","source_row_sha256","status","disposition","exclusion_reason","benchmark_row","provenance_row","reviewed_by","independent_attestation","notes"];
const EDIT_STRING_KEYS=["disposition","exclusion_reason","reviewed_by","notes","sample_id","paper_id","stratum","prompt","system_instruction","source_document_id","creator_group_id","gold_answer","evidence_used","visual_facts","temporal_facts","uncertainty","missing_evidence","criteria","adjudicators"];
const EDIT_BOOL_KEYS=["gold_enabled","independent_attestation"],IMAGE_KEYS=["image_path","sha256","page","locator","source_url","license","citation","verified_by"];
const $=id=>document.getElementById(id); const node=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n};
const plain=value=>value!==null&&typeof value==="object"&&!Array.isArray(value)&&Object.getPrototypeOf(value)===Object.prototype;
const blank=()=>({disposition:"",exclusion_reason:"",reviewed_by:"",notes:"",sample_id:"",paper_id:"",stratum:"multimodal_hard",prompt:"",system_instruction:"",source_document_id:"",creator_group_id:"",images:[],gold_enabled:false,independent_attestation:false,gold_answer:"",evidence_used:"",visual_facts:"",temporal_facts:"",uncertainty:"",missing_evidence:"",criteria:"",adjudicators:""});
const taskIds=new Set(DATA.tasks.map(item=>item.task.task_id));
function storageWarning(detail){const target=$("storage-warning");target.textContent="Хранилище браузера недоступно или повреждено. Работа продолжится в этой вкладке; регулярно экспортируйте черновик. "+detail;target.classList.remove("hidden")}
function storageGet(){try{return localStorage.getItem(KEY)}catch(_error){storageWarning("Не удалось прочитать localStorage.");return null}}
function storageSet(value){try{localStorage.setItem(KEY,value)}catch(_error){storageWarning("Не удалось сохранить данные в localStorage.")}}
function storageRemove(){try{localStorage.removeItem(KEY)}catch(_error){storageWarning("Не удалось очистить localStorage.")}}
function exactKeys(value,keys){const actual=Object.keys(value);return actual.length===keys.length&&keys.every(key=>Object.hasOwn(value,key))}
function validateEdit(value,taskId){const allKeys=[...EDIT_STRING_KEYS,...EDIT_BOOL_KEYS,"images"];if(!plain(value)||!exactKeys(value,allKeys))throw new Error(`task_id ${taskId}: недопустимый набор полей черновика`);const out=blank();for(const key of EDIT_STRING_KEYS){if(typeof value[key]!=="string")throw new Error(`task_id ${taskId}: поле ${key} должно быть строкой`);out[key]=value[key]}for(const key of EDIT_BOOL_KEYS){if(typeof value[key]!=="boolean")throw new Error(`task_id ${taskId}: поле ${key} должно быть логическим`);out[key]=value[key]}if(!Array.isArray(value.images))throw new Error(`task_id ${taskId}: images должно быть массивом`);out.images=value.images.map((image,index)=>{if(!plain(image)||!exactKeys(image,IMAGE_KEYS)||IMAGE_KEYS.some(key=>typeof image[key]!=="string"))throw new Error(`task_id ${taskId}: некорректное изображение ${index+1}`);const result={};for(const key of IMAGE_KEYS)result[key]=image[key];return result});return out}
let state={index:0,edits:{}};for(const taskId of taskIds)state.edits[taskId]=blank();const stored=storageGet();if(stored!==null){try{const saved=JSON.parse(stored);if(!plain(saved)||saved.queue_fingerprint!==DATA.queue_fingerprint||!Number.isInteger(saved.index)||!plain(saved.edits))throw new Error("структура сохранённого состояния не совпадает");const restoreCandidate={index:0,edits:{}};for(const taskId of taskIds)restoreCandidate.edits[taskId]=blank();for(const [taskId,value] of Object.entries(saved.edits)){if(!taskIds.has(taskId))throw new Error(`неизвестный task_id ${taskId}`);restoreCandidate.edits[taskId]=validateEdit(value,taskId)}if(saved.index>=0&&saved.index<DATA.task_count)restoreCandidate.index=saved.index;state=restoreCandidate}catch(_error){storageWarning("Сохранённое состояние пропущено: данные повреждены или имеют другую структуру.")}}
const imageRuntime=new Map();let pendingHashCount=0;function runtimeFor(image){let runtime=imageRuntime.get(image);if(!runtime){runtime={token:0,pending:false,url:null,name:"",size:0,error:""};imageRuntime.set(image,runtime)}return runtime}function revokeRuntimeUrl(runtime){if(runtime.url){URL.revokeObjectURL(runtime.url);runtime.url=null}}function beginHash(image){const runtime=runtimeFor(image);runtime.token+=1;revokeRuntimeUrl(runtime);runtime.name="";runtime.size=0;runtime.error="";if(!runtime.pending){runtime.pending=true;pendingHashCount+=1}return runtime.token}function currentHash(image,token,fileInput,file){const runtime=imageRuntime.get(image);return Boolean(runtime&&runtime.token===token&&fileInput.files&&fileInput.files[0]===file)}function finishHash(image,token){const runtime=imageRuntime.get(image);if(!runtime||runtime.token!==token)return null;if(runtime.pending){runtime.pending=false;pendingHashCount-=1}return runtime}function discardImageRuntime(image){const runtime=imageRuntime.get(image);if(!runtime)return;runtime.token+=1;if(runtime.pending){runtime.pending=false;pendingHashCount-=1}revokeRuntimeUrl(runtime);imageRuntime.delete(image)}function revokeAllPreviews(){for(const [image] of imageRuntime)discardImageRuntime(image);pendingHashCount=0}
const lines=value=>value.split(/\r?\n/).map(x=>x.trim()).filter(Boolean); const edit=()=>state.edits[DATA.tasks[state.index].task.task_id];
function save(){storageSet(JSON.stringify({queue_fingerprint:DATA.queue_fingerprint,index:state.index,edits:state.edits}));progress()}
function field(parent,label,key,type="input"){const l=node("label",label);const el=node(type);el.value=edit()[key]||"";el.addEventListener("input",()=>{edit()[key]=el.value;save()});parent.append(l,el);return el}
function progress(){let complete=0,retain=0,exclude=0,invalid=0;for(const item of DATA.tasks){const e=state.edits[item.task.task_id];const errors=validateOne(item,e);if(e.disposition)complete++;if(e.disposition==="retain")retain++;if(e.disposition==="exclude")exclude++;if(e.disposition&&errors.length)invalid++}$("progress").textContent=`Решено ${complete}/${DATA.task_count} · сохранить ${retain} · исключить ${exclude} · с ошибками ${invalid}`}
function matches(item){const e=state.edits[item.task.task_id],q=$("search").value.toLowerCase(),f=$("filter").value;const hay=JSON.stringify(item.task).toLowerCase();if(q&&!hay.includes(q))return false;const bad=e.disposition&&validateOne(item,e).length;if(f==="pending")return !e.disposition;if(f==="retain"||f==="exclude")return e.disposition===f;if(f==="invalid")return !!bad;if(f==="complete")return !!e.disposition&&!bad;return true}
function move(delta){let i=state.index;for(let n=0;n<DATA.tasks.length;n++){i=(i+delta+DATA.tasks.length)%DATA.tasks.length;if(matches(DATA.tasks[i])){state.index=i;save();render();return}}}
function sourcePanel(item){const root=$("source");root.replaceChildren(node("h2",`Задача ${item.task.source_row_index+1} из ${DATA.task_count}`),node("div",`task_id: ${item.task.task_id}`,"binding"),node("p","Исходные данные ниже доступны только для чтения и могут содержать ошибки. Не переносите их в replacement без независимой проверки. Сверьте queue fingerprint вверху страницы с handoff."));const codes=node("div",undefined,"codes");for(const value of [...item.task.critical_codes,...item.task.warning_codes])codes.append(node("span",value,"code"));const overlap=item.task.training_overlap_paper_ids.length?item.task.training_overlap_paper_ids.join(", "):item.task.critical_codes.includes("training_paper_overlap")?"audit обнаружил training_paper_overlap; canonical ID отсутствует в metadata задачи, сверить original_row и audit evidence":"не обнаружено";root.append(node("h3","Коды findings"),codes,node("div","Пересечение с training: "+overlap,"meta"));const previews=node("div",undefined,"preview");for(const image of item.task.audited_image_hashes){const box=node("div"),img=node("img");img.src=image.preview_path;img.alt=image.path;box.append(img,node("div",`${image.path}\n${image.sha256}`,"meta"));previews.append(box)}root.append(node("h3","Проверенные при audit исходные изображения"),previews,node("h3","Исходная original_row (только чтение)"),node("pre",JSON.stringify(item.task.original_row,null,2)),node("h3","Устаревшие provenance-записи (только чтение)"),node("pre",JSON.stringify(item.task.legacy_provenance_rows,null,2)))}
function imageForm(parent,image,index){const box=node("section",undefined,"image-form"),title=node("div",`Изображение ${index+1}`,"row"),remove=node("button","Удалить","danger");remove.type="button";remove.addEventListener("click",()=>{discardImageRuntime(image);edit().images.splice(index,1);save();render()});title.append(remove);box.append(title,node("p","Укажите canonical image_path самостоятельно. После экспорта отдельно скопируйте выбранный файл в $Curated по пути assets/images/...; имя файла не подставляется автоматически."));let shaInput;for(const [label,key] of [["image_path (assets/images/...)","image_path"],["SHA256 в нижнем регистре","sha256"],["Страница","page"],["Точный locator (рисунок, панель, таблица)","locator"],["URL источника","source_url"],["Лицензия","license"],["Библиографическая citation","citation"],["verified_by (по одному ASCII ID в строке)","verified_by"]]){const l=node("label",label),el=key==="citation"?node("textarea"):node("input");el.value=image[key]||"";el.addEventListener("input",()=>{image[key]=el.value;save()});if(key==="sha256")shaInput=el;box.append(l,el)}const fileLabel=node("label","Выбрать проверенный файл и вычислить SHA256"),fileInput=node("input");fileInput.type="file";fileInput.accept="image/*";const fileStatus=node("div","Файл не выбран. SHA256 можно ввести вручную.","meta"),localPreview=node("img");localPreview.alt="Локальный preview выбранного файла";const existingRuntime=imageRuntime.get(image);if(existingRuntime&&existingRuntime.pending){fileStatus.textContent="Вычисляется SHA256 выбранного файла…";fileStatus.className="warning";localPreview.className="hidden"}else if(existingRuntime&&existingRuntime.error){fileStatus.textContent=existingRuntime.error;fileStatus.className="error";localPreview.className="hidden"}else if(existingRuntime&&existingRuntime.url){localPreview.src=existingRuntime.url;fileStatus.textContent=`Выбран файл: ${existingRuntime.name}, ${existingRuntime.size} байт`}else localPreview.className="hidden";fileInput.addEventListener("change",async()=>{const file=fileInput.files&&fileInput.files[0];if(!file)return;const token=beginHash(image);localPreview.removeAttribute("src");localPreview.className="hidden";fileStatus.textContent="Вычисляется SHA256 выбранного файла…";fileStatus.className="warning";if(!globalThis.crypto||!crypto.subtle||typeof crypto.subtle.digest!=="function"){const runtime=finishHash(image,token);if(runtime){runtime.error="Web Crypto недоступен. Введите SHA256 вручную.";fileStatus.textContent=runtime.error;fileStatus.className="error"}return}try{const buffer=await file.arrayBuffer();if(!currentHash(image,token,fileInput,file))return;const digest=await crypto.subtle.digest("SHA-256",buffer);if(!currentHash(image,token,fileInput,file))return;const runtime=finishHash(image,token);if(!runtime)return;image.sha256=Array.from(new Uint8Array(digest),value=>value.toString(16).padStart(2,"0")).join("");shaInput.value=image.sha256;save();runtime.name=file.name;runtime.size=file.size;try{runtime.url=URL.createObjectURL(file);localPreview.src=runtime.url;localPreview.className=""}catch(_previewError){runtime.url=null;localPreview.className="hidden"}fileStatus.textContent=`SHA256 вычислен. Выбран файл: ${file.name}, ${file.size} байт. Скопируйте эти bytes в $Curated по введённому image_path.`;fileStatus.className="ok"}catch(_error){if(!currentHash(image,token,fileInput,file))return;const runtime=finishHash(image,token);if(runtime){runtime.error="Не удалось прочитать файл или вычислить SHA256. Введите SHA256 вручную.";fileStatus.textContent=runtime.error;fileStatus.className="error"}}});fileLabel.append(fileInput);box.append(fileLabel,fileStatus,localPreview);parent.append(box)}
function formPanel(item){const e=edit(),root=$("form");root.replaceChildren();root.append(node("h2","Решение экспертов"),node("p","Исходная строка может быть ошибочной. Сохранение требует независимо проверенный полный replacement; исключение требует конкретную проверяемую причину. Сверьте queue fingerprint с handoff. Предварительная проверка браузера не заменяет curate-assemble."),node("div","Неизменяемая привязка к очереди","meta"),node("div",JSON.stringify(item.binding),"binding"));const d=node("select");for(const [v,t] of [["","Выберите решение"],["retain","Сохранить с полной заменой"],["exclude","Исключить"]]){const o=node("option",t);o.value=v;o.selected=e.disposition===v;d.append(o)}d.addEventListener("change",()=>{e.disposition=d.value;save();render()});root.append(node("label","Решение disposition"),d);field(root,"reviewed_by (по одному ASCII ID в строке)","reviewed_by","textarea");const attestLabel=node("label"),attest=node("input");attest.type="checkbox";attest.style.width="auto";attest.checked=e.independent_attestation;attest.addEventListener("change",()=>{e.independent_attestation=attest.checked;save();render()});attestLabel.append(attest,document.createTextNode(" Подтверждаю: два указанных эксперта независимо проверили это решение"));root.append(attestLabel);field(root,"Примечания notes","notes","textarea");if(e.disposition==="exclude"){field(root,"Конкретная причина исключения exclusion_reason","exclusion_reason","textarea")}if(e.disposition==="retain"){const copy=node("button","Скопировать только sample_id и paper_id из источника");copy.type="button";copy.addEventListener("click",()=>{e.sample_id=String(item.task.original_row.sample_id||"");e.paper_id=String(item.task.original_row.paper_id||"");save();render()});root.append(node("p","Поля replacement изначально пусты. Кнопка ниже явно копирует только идентификаторы; их всё равно нужно проверить."),copy);const two=node("div",undefined,"two");field(two,"sample_id","sample_id");field(two,"Канонический paper_id","paper_id");root.append(two);const sl=node("label","Страта stratum"),s=node("select");for(const v of ["multimodal_hard","temporal_hard","easy_control"]){const o=node("option",v);o.value=v;o.selected=e.stratum===v;s.append(o)}s.addEventListener("change",()=>{e.stratum=s.value;save()});root.append(sl,s);field(root,"Самодостаточный запрос пользователю prompt","prompt","textarea");field(root,"Системная инструкция (необязательно)","system_instruction","textarea");const ids=node("div",undefined,"two");field(ids,"source_document_id","source_document_id");field(ids,"creator_group_id","creator_group_id");root.append(ids,node("h3","Изображения replacement"));e.images.forEach((image,index)=>imageForm(root,image,index));const add=node("button","Добавить изображение");add.type="button";add.addEventListener("click",()=>{e.images.push({image_path:"",sha256:"",page:"",locator:"",source_url:"",license:"",citation:"",verified_by:""});save();render()});root.append(add);const goldLabel=node("label"),gold=node("input");gold.type="checkbox";gold.checked=e.gold_enabled;gold.style.width="auto";gold.addEventListener("change",()=>{e.gold_enabled=gold.checked;save();render()});goldLabel.append(gold,document.createTextNode(" Добавить adjudicated gold_answer и rubric"));root.append(node("h3","Необязательный gold_answer"),goldLabel);if(e.gold_enabled){field(root,"Эталонный ответ gold_answer","gold_answer","textarea");field(root,"Использованные свидетельства (по одному в строке)","evidence_used","textarea");field(root,"Визуальные факты (по одному в строке)","visual_facts","textarea");field(root,"Временные факты (по одному в строке)","temporal_facts","textarea");field(root,"Неопределённость (необязательно)","uncertainty","textarea");field(root,"Недостающие свидетельства (необязательно)","missing_evidence","textarea");field(root,"Критерии rubric (по одному в строке)","criteria","textarea");field(root,"adjudicators (по одному ASCII ID в строке)","adjudicators","textarea")}}const errors=validateOne(item,e),status=node("div",errors.length?errors.join("\n"):e.disposition?"Предварительная проверка решения пройдена.":"Решение ещё не заполнено.",errors.length?"error":e.disposition?"ok":"meta");root.append(node("h3","Предварительная проверка в браузере"),status)}
function normalizedIdentity(value){return value.normalize("NFKC").trim().replace(/\s+/gu," ").toLowerCase()}function ids(value,label,errors){const values=lines(value),normalized=values.map(normalizedIdentity);if(values.length<2)errors.push(`${label}: нужны минимум два идентификатора`);if(normalized.some(x=>!/^[A-Za-z0-9](?:[A-Za-z0-9._:@/+ -]*[A-Za-z0-9])?$/.test(x)))errors.push(`${label}: после NFKC-нормализации используйте ASCII-идентификаторы без пробелов по краям`);if(new Set(normalized).size!==normalized.length)errors.push(`${label}: идентификаторы должны быть различными после NFKC, нормализации пробелов и регистра`);return values}
function validAssetPath(value){if(typeof value!=="string"||!value.startsWith("assets/images/")||value.includes("\\")||value.includes(":"))return false;const parts=value.split("/");if(parts.length<3||parts.some(part=>!part||part==="."||part===".."||part!==part.replace(/[ .]+$/,"")))return false;for(const part of parts){if(/[<>"|?*\x00-\x1f]/.test(part))return false;const base=part.split(".",1)[0].trimEnd().replace(/[¹²³]/g,d=>({"¹":"1","²":"2","³":"3"})[d]).toLowerCase();if(/^(aux|clock\$|con|conin\$|conout\$|nul|prn|com[1-9]|lpt[1-9])$/.test(base))return false}return true}
function validateOne(item,e){const errors=[];if(!["retain","exclude"].includes(e.disposition))return ["Выберите: сохранить replacement или исключить."];ids(e.reviewed_by,"reviewed_by",errors);if(!e.independent_attestation)errors.push("Подтвердите независимую проверку решения двумя указанными экспертами.");if(e.disposition==="exclude"){if(!e.exclusion_reason.trim())errors.push("Укажите конкретную причину исключения.");return errors}for(const [k,n] of [["sample_id","sample_id"],["paper_id","paper_id"],["source_document_id","source_document_id"],["creator_group_id","creator_group_id"]])if(!/^[\x21-\x7e](?:[\x20-\x7e]*[\x21-\x7e])?$/.test(e[k]))errors.push(`${n}: требуется ASCII-значение без пробелов по краям.`);if(!e.prompt.trim()||e.prompt!==e.prompt.trim())errors.push("prompt обязателен и не должен содержать пробелы по краям.");if(e.paper_id!==e.paper_id.toLowerCase()||!/^(?:doi:10\.\d{4,9}\/[^\s]+|arxiv:(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[a-z]{2})?\/\d{7})|paper:\S(?:[\x20-\x7e]*\S)?)$/.test(e.paper_id))errors.push("paper_id должен быть каноническим в нижнем регистре: doi:/arxiv: без пробелов либо paper: с внутренними пробелами.");if(!e.images.length)errors.push("Добавьте минимум одно изображение.");e.images.forEach((im,i)=>{const p=`Изображение ${i+1}`;if(!validAssetPath(im.image_path))errors.push(`${p}: image_path должен быть нормализованным Windows-safe путём внутри assets/images/.`);if(!/^[0-9a-f]{64}$/.test(im.sha256))errors.push(`${p}: требуется lowercase SHA256.`);for(const k of ["page","locator","license","citation"])if(!String(im[k]||"").trim())errors.push(`${p}: заполните ${k}.`);try{const u=new URL(im.source_url);if(!["http:","https:"].includes(u.protocol))throw 0}catch(_error){errors.push(`${p}: source_url должен быть HTTP(S) URL.`)}ids(im.verified_by,`${p} verified_by`,errors)});if(e.gold_enabled||DATA.policy.require_gold){if(!e.gold_answer.trim()||!lines(e.evidence_used).length||!lines(e.visual_facts).length||!lines(e.temporal_facts).length||!lines(e.criteria).length)errors.push("Полностью заполните gold_answer, evidence, visual/temporal facts и критерии rubric.");ids(e.adjudicators,"adjudicators",errors)}return errors}
function decision(item,e){const images=e.images.map(im=>({image_path:im.image_path,sha256:im.sha256,page:im.page.trim(),locator:im.locator,source_url:im.source_url,license:im.license,citation:im.citation,verified_by:lines(im.verified_by)}));const messages=[];if(e.system_instruction.trim())messages.push({role:"system",content:[{type:"text",text:e.system_instruction}]});messages.push({role:"user",content:[{type:"text",text:e.prompt},...images.map(()=>({type:"image"}))]});let benchmark={sample_id:e.sample_id,paper_id:e.paper_id,stratum:e.stratum,primary_endpoint:DATA.policy.primary_strata.includes(e.stratum),model_task_prompt:e.prompt,messages,images:images.map(x=>x.image_path),split_provenance:{paper_holdout:true,source_holdout:true,creator_holdout:true,training_overlap_checked:true,source_document_id:e.source_document_id,creator_group_id:e.creator_group_id}};if(e.gold_enabled||DATA.policy.require_gold){benchmark.gold_answer={answer:e.gold_answer,evidence_used:lines(e.evidence_used),visual_facts:lines(e.visual_facts),temporal_facts:lines(e.temporal_facts),uncertainty:e.uncertainty.trim()||null,missing_evidence:e.missing_evidence.trim()||null};benchmark.rubric={criteria:lines(e.criteria),adjudicators:lines(e.adjudicators)}}const result={...item.binding,status:"complete",disposition:e.disposition,exclusion_reason:e.disposition==="exclude"?e.exclusion_reason:null,benchmark_row:e.disposition==="retain"?benchmark:null,provenance_row:e.disposition==="retain"?{sample_id:e.sample_id,paper_id:e.paper_id,images}:null,reviewed_by:lines(e.reviewed_by),independent_attestation:e.independent_attestation,notes:e.notes};const ordered={};for(const key of FIELDS)ordered[key]=result[key];return ordered}
function download(name,text){const a=node("a");a.href=URL.createObjectURL(new Blob([text],{type:"application/json"}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}
function exportFinal(){if(pendingHashCount>0){setFirstInvalidNavigation(null);$("global-error").textContent=`Финальный экспорт заблокирован: дождитесь вычисления SHA256 (${pendingHashCount}).`;return}const report=collectExportProblems();if(report.taskErrors.length){setFirstInvalidNavigation(report.firstInvalidIndex);$("global-error").textContent=["Финальный экспорт невозможен: все retain требуют полностью заполненных полей replacement; недействительные решения нельзя экспортировать.",`Ошибки по задачам: ${report.taskErrors.length}; недействительных retain: ${report.invalidRetainCount}; недействительных exclude: ${report.invalidExcludeCount}; без решения: ${report.invalidUndecidedCount}.`,...report.taskErrors.slice(0,30),...(report.taskErrors.length>30?[`... ещё ${report.taskErrors.length-30}`]:[])].join("\n");return}setFirstInvalidNavigation(null);$("global-error").textContent="";download("completed_decisions.jsonl",report.decisions.map(x=>JSON.stringify(x)).join("\n")+"\n")}
function sameEdit(left,right){return JSON.stringify(left)===JSON.stringify(right)}
function importDraft(file){if(file.size>MAX_DRAFT_BYTES){$("global-error").textContent="Импорт отклонён: размер черновика превышает 32 МиБ.";return}const reader=new FileReader();reader.onerror=()=>{$("global-error").textContent="Импорт отклонён: FileReader не смог прочитать черновик."};reader.onabort=()=>{$("global-error").textContent="Импорт отклонён: чтение черновика отменено."};reader.onload=()=>{try{if(typeof reader.result!=="string")throw new Error("черновик не является текстом");const envelope=JSON.parse(reader.result);if(!plain(envelope)||!exactKeys(envelope,["artifact_version","queue_fingerprint","edits"]))throw new Error("неверные поля envelope черновика");if(envelope.artifact_version!==DATA.artifact_version)throw new Error("неподдерживаемый artifact_version черновика");if(envelope.queue_fingerprint!==DATA.queue_fingerprint)throw new Error("черновик относится к другой очереди");if(!plain(envelope.edits))throw new Error("edits должен быть объектом с ключами task_id");const incoming={};for(const [taskId,value] of Object.entries(envelope.edits)){if(!taskIds.has(taskId))throw new Error(`неизвестный task_id ${taskId}`);incoming[taskId]=validateEdit(value,taskId)}const candidate={};for(const taskId of taskIds)candidate[taskId]=validateEdit(state.edits[taskId],taskId);const empty=blank(),summary={merged:0,equal:0,blank_ignored:0};for(const [taskId,value] of Object.entries(incoming)){const current=candidate[taskId];if(sameEdit(value,empty))summary.blank_ignored++;else if(sameEdit(current,empty)){candidate[taskId]=value;summary.merged++}else if(sameEdit(current,value))summary.equal++;else throw new Error(`конфликт merge для task_id ${taskId}; изменения не применены`)}revokeAllPreviews();state.edits=candidate;save();render();$("global-error").textContent=`Merge черновика завершён: добавлено ${summary.merged}, уже совпадало ${summary.equal}, пустых пропущено ${summary.blank_ignored}.`}catch(error){$("global-error").textContent="Импорт отклонён: "+(error instanceof SyntaxError?"некорректный JSON":error.message)}};try{reader.readAsText(file)}catch(_error){$("global-error").textContent="Импорт отклонён: не удалось начать чтение файла."}}
function render(){const item=DATA.tasks[state.index];sourcePanel(item);formPanel(item);$("position").textContent=`${state.index+1} / ${DATA.task_count}`;progress()}
$("prev").addEventListener("click",()=>move(-1));$("next").addEventListener("click",()=>move(1));$("search").addEventListener("input",render);$("filter").addEventListener("change",render);$("export-final").addEventListener("click",exportFinal);$("export-draft").addEventListener("click",()=>download("curator_draft.json",JSON.stringify({artifact_version:DATA.artifact_version,queue_fingerprint:DATA.queue_fingerprint,edits:state.edits})+"\n"));$("import-draft").addEventListener("change",event=>event.target.files[0]&&importDraft(event.target.files[0]));$("clear").addEventListener("click",()=>{if(confirm("Удалить все локально сохранённые решения для этой очереди?")){revokeAllPreviews();storageRemove();state={index:0,edits:{}};for(const taskId of taskIds)state.edits[taskId]=blank();save();render()}});window.addEventListener("beforeunload",revokeAllPreviews);render();
"""


_RECOVERY_SCRIPT = r"""
const recoveryToolbar=$("export-final").parentElement;
const resetInvalidRetainsButton=node("button","Сбросить неполные retain","danger");
resetInvalidRetainsButton.id="reset-invalid-retains";
resetInvalidRetainsButton.type="button";
const firstErrorButton=node("button","Перейти к первой ошибке");
firstErrorButton.id="go-first-error";
firstErrorButton.type="button";
firstErrorButton.hidden=true;
recoveryToolbar.insertBefore(resetInvalidRetainsButton,$("export-final"));
recoveryToolbar.insertBefore(firstErrorButton,$("export-final"));
let firstInvalidIndex=null;
function setFirstInvalidNavigation(index){firstInvalidIndex=index;firstErrorButton.hidden=index===null}
function resetInvalidRetains(){const invalidRetains=[];for(const item of DATA.tasks){const taskId=item.task.task_id,e=state.edits[taskId];if(e.disposition==="retain"&&validateOne(item,e).length>0)invalidRetains.push({taskId,e})}if(!invalidRetains.length){$("global-error").textContent="Неполных retain для сброса нет.";return}if(!confirm(`Сбросить ${invalidRetains.length} неполных retain? Их временные previews будут удалены, а exclude останутся без изменений.`))return;for(const {taskId,e} of invalidRetains){for(const image of e.images)discardImageRuntime(image);state.edits[taskId]=blank()}setFirstInvalidNavigation(null);save();render();$("global-error").textContent=`Сброшено неполных retain: ${invalidRetains.length}. Исключения не изменены.`}
function collectExportProblems(){const taskErrors=[],decisions=[],sampleIds=new Map();let invalidRetainCount=0,invalidExcludeCount=0,invalidUndecidedCount=0,firstInvalidIndex=null;DATA.tasks.forEach((item,index)=>{const e=state.edits[item.task.task_id],bad=validateOne(item,e);if(e.disposition==="retain"){if(sampleIds.has(e.sample_id))bad.push(`sample_id уже использован в задаче ${sampleIds.get(e.sample_id)+1}.`);else sampleIds.set(e.sample_id,index)}if(bad.length){if(firstInvalidIndex===null)firstInvalidIndex=index;if(e.disposition==="retain")invalidRetainCount+=1;else if(e.disposition==="exclude")invalidExcludeCount+=1;else invalidUndecidedCount+=1;taskErrors.push(`Задача ${index+1} (${item.task.task_id}): ${bad.join(" ")}`)}else decisions.push(decision(item,e))});return{taskErrors,decisions,invalidRetainCount,invalidExcludeCount,invalidUndecidedCount,firstInvalidIndex}}
function goToFirstError(){const report=collectExportProblems();if(report.firstInvalidIndex===null){setFirstInvalidNavigation(null);return}setFirstInvalidNavigation(report.firstInvalidIndex);state.index=firstInvalidIndex;save();render()}
resetInvalidRetainsButton.addEventListener("click",resetInvalidRetains);
firstErrorButton.addEventListener("click",goToFirstError);
"""

_SCRIPT += _RECOVERY_SCRIPT


_EXPERT_FORM_SCRIPT = r"""
const EXPERT_FIELD_GUIDE=Object.freeze({
  reviewed_by:{label:"Идентификаторы двух экспертов",help:"Укажите реальные служебные ID: по одному на строке, минимум два разных ID латиницей. Не используйте reviewer-slot-1/2/3.",placeholder:"expert-01\nexpert-02",required:true},
  notes:{label:"Комментарий к решению",help:"Необязательное пояснение для владельца релиза. Например: что именно было проверено или исправлено.",placeholder:"Необязательно"},
  exclusion_reason:{label:"Почему запись нужно исключить?",help:"Опишите конкретную проверяемую причину. Например: подтверждено пересечение с обучающими данными или невозможно установить источник изображения.",placeholder:"Конкретная причина исключения",required:true},
  sample_id:{label:"Уникальный ID примера",help:"Уникальное короткое имя этой записи. Оно не должно повторяться в других сохранённых записях.",placeholder:"cap150-example-001",required:true},
  paper_id:{label:"ID научной статьи",help:"Канонический ID строчными буквами: doi:10...; arxiv:...; либо paper:... Если исходный ID неверен, укажите независимо подтверждённый.",placeholder:"doi:10.1234/example",required:true},
  prompt:{label:"Вопрос к модели",help:"Напишите самодостаточный научный вопрос по выбранным изображениям. Не включайте ответ, сравнение со старым ответом или ссылки на скрытый контекст.",placeholder:"Какой вывод следует из показанного графика и какие визуальные признаки это подтверждают?",required:true},
  system_instruction:{label:"Дополнительная инструкция модели",help:"Необязательное ограничение формата ответа. Не добавляйте сюда факты, которые модель должна вывести сама.",placeholder:"Отвечайте только по информации на изображении."},
  source_document_id:{label:"ID исходного документа",help:"Стабильный ID конкретной статьи, PDF или версии документа, проверенной на отсутствие в обучающих источниках.",placeholder:"doi:10.1234/example",required:true},
  creator_group_id:{label:"ID автора или группы создателей",help:"Стабильный ID автора либо группы, по которому проверено отсутствие пересечения с обучающими данными.",placeholder:"orcid:0000-0000-0000-0000",required:true},
  gold_answer:{label:"Эталонный ответ",help:"Ответ, независимо согласованный экспертами. Заполняется только если для исследования нужен gold answer.",placeholder:"Проверенный эталонный ответ",required:true},
  evidence_used:{label:"Свидетельства для эталонного ответа",help:"По одному подтверждающему фрагменту на строке.",placeholder:"Figure 2, panel A: ...",required:true},
  visual_facts:{label:"Проверенные визуальные факты",help:"По одному наблюдаемому на изображении факту на строке.",placeholder:"Красная линия достигает максимума после синей.",required:true},
  temporal_facts:{label:"Проверенные временные факты",help:"По одному временному отношению или событию на строке. Если временная ось отсутствует, явно укажите это.",placeholder:"Пик наблюдается после отметки 20 ms.",required:true},
  uncertainty:{label:"Неопределённость",help:"Необязательно: что нельзя уверенно установить по доступным данным.",placeholder:"Необязательно"},
  missing_evidence:{label:"Недостающие свидетельства",help:"Необязательно: каких данных не хватает для полного ответа.",placeholder:"Необязательно"},
  criteria:{label:"Критерии оценки ответа",help:"По одному понятному критерию на строке.",placeholder:"Верно называет основной тренд.\nСсылается на наблюдаемые признаки.",required:true},
  adjudicators:{label:"Эксперты эталонного ответа",help:"Минимум два разных реальных ASCII ID, по одному на строке.",placeholder:"expert-01\nexpert-02",required:true}
});
let expertFieldSerial=0;
field=function(parent,legacyLabel,key,type="input"){
  const spec=EXPERT_FIELD_GUIDE[key]||{label:legacyLabel,help:"Заполните поле после независимой проверки.",required:true};
  const wrap=node("div",undefined,"field-group");
  wrap.dataset.fieldGroup=key;
  const label=node("label");
  label.append(document.createTextNode(spec.label));
  if(spec.required)label.append(document.createTextNode(" "),node("span","*","required-mark"));
  const control=node(type);
  control.value=edit()[key]||"";
  control.autocomplete="off";
  control.dataset.field=key;
  if(spec.placeholder)control.placeholder=spec.placeholder;
  const id=`expert-field-${key}-${++expertFieldSerial}`;
  const helpId=`${id}-help`;
  control.id=id;
  label.htmlFor=id;
  control.setAttribute("aria-describedby",helpId);
  if(spec.required)control.setAttribute("aria-required","true");
  control.addEventListener("input",()=>{edit()[key]=control.value;save()});
  const help=node("div",spec.help,"field-help");
  help.id=helpId;
  wrap.append(label,control,help);
  parent.append(wrap);
  return control;
};

const EXPERT_IMAGE_GUIDE=Object.freeze({
  "image_path (assets/images/...)":{label:"Путь изображения в датасете",help:"Укажите новый нормализованный путь, начинающийся с assets/images/. После экспорта те же bytes нужно поместить по этому пути.",placeholder:"assets/images/cap150-example-001/figure-2a.png",required:true},
  "SHA256 в нижнем регистре":{label:"Контрольная сумма изображения (SHA256)",help:"Выберите точный локальный файл ниже: форма вычислит SHA256 автоматически. Не хешируйте имя файла или URL.",placeholder:"64 шестнадцатеричных символа",required:true},
  "Страница":{label:"Страница документа",help:"Страница PDF или другое точное обозначение места в документе.",placeholder:"3",required:true},
  "Точный locator (рисунок, панель, таблица)":{label:"Где находится изображение",help:"Укажите номер рисунка, панель или таблицу так, чтобы другой эксперт мог быстро найти источник.",placeholder:"Figure 2, panel A",required:true},
  "URL источника":{label:"Прямая ссылка на источник",help:"Полный HTTP(S) URL без логина, query-параметров и #fragment. Ссылка должна вести к проверенному источнику.",placeholder:"https://example.org/article/figure-2a.png",required:true},
  "Лицензия":{label:"Лицензия использования",help:"Точное обозначение подтверждённой лицензии. Не делайте вывод только по имени сайта.",placeholder:"CC-BY-4.0",required:true},
  "Библиографическая citation":{label:"Библиографическая ссылка",help:"Полная ссылка на статью и конкретный рисунок или панель.",placeholder:"Author et al. (2025), Figure 2A, DOI: ...",required:true},
  "verified_by (по одному ASCII ID в строке)":{label:"Кто проверил это изображение",help:"Минимум два разных реальных ASCII ID, по одному на строке. Эксперты подтверждают bytes, источник, locator и лицензию.",placeholder:"expert-01\nexpert-02",required:true},
  "Выбрать проверенный файл и вычислить SHA256":{label:"Выберите точный файл изображения",help:"Файл не загружается в сеть и не сохраняется в HTML. После экспорта отдельно поместите эти же bytes в curated_dataset.",required:true}
});

function guidedLabel(label,control,spec){
  const id=`expert-image-field-${++expertFieldSerial}`;
  const helpId=`${id}-help`;
  const required=spec.required?" *":"";
  if(label.contains(control))label.replaceChildren(document.createTextNode(spec.label+required+" "),control);
  else{label.textContent=spec.label+required;label.htmlFor=id}
  control.id=id;
  control.setAttribute("aria-describedby",helpId);
  if(spec.placeholder)control.placeholder=spec.placeholder;
  if(spec.required)control.setAttribute("aria-required","true");
  const help=node("div",spec.help,"field-help");
  help.id=helpId;
  if(label.contains(control))label.after(help);else control.after(help);
}

function enhanceImageForm(box,image){
  const intro=Array.from(box.children).find(child=>child.tagName==="P");
  if(intro){intro.textContent="Используйте только новое независимо проверенное изображение. Заполните все поля ниже и затем выберите точный локальный файл.";intro.className="simple-intro"}
  const remove=box.querySelector("button.danger");
  if(remove)remove.textContent="Удалить изображение";
  for(const label of Array.from(box.querySelectorAll("label"))){
    const original=label.textContent.trim();
    const spec=EXPERT_IMAGE_GUIDE[original];
    if(!spec)continue;
    let control=label.querySelector("input,textarea,select")||label.nextElementSibling;
    if(!control||!["INPUT","TEXTAREA","SELECT"].includes(control.tagName))continue;
    if(original.startsWith("verified_by")&&control.tagName!=="TEXTAREA"){
      const textarea=node("textarea");
      textarea.value=image.verified_by||"";
      textarea.addEventListener("input",()=>{image.verified_by=textarea.value;save()});
      control.replaceWith(textarea);
      control=textarea;
    }
    guidedLabel(label,control,spec);
  }
}

function expertGuide(){
  const guide=node("section",undefined,"expert-guide");
  guide.append(node("h3","Как заполнить эту запись"));
  const list=node("ol");
  for(const text of [
    "Слева изучите исходную запись, замечания аудита и изображения.",
    "Выберите: оставить запись после полной проверки или исключить её.",
    "Если запись остаётся, проверьте статью, вопрос, изображения, источник, лицензию и отсутствие пересечений с обучающими данными.",
    "В последнем блоке укажите двух независимых экспертов и подтвердите совместное решение."
  ])list.append(node("li",text));
  guide.append(list,node("p","Поля со знаком * обязательны. Данные сохраняются локально в этом браузере.","meta"));
  return guide;
}

function humanizeValidation(message){
  return message
    .replaceAll("reviewed_by","идентификаторы экспертов")
    .replaceAll("verified_by","проверившие изображение")
    .replaceAll("source_document_id","ID исходного документа")
    .replaceAll("creator_group_id","ID автора или группы")
    .replaceAll("paper_id","ID статьи")
    .replaceAll("prompt","вопрос к модели")
    .replaceAll("replacement","новая запись")
    .replaceAll("retain","оставленная запись")
    .replaceAll("exclude","исключённая запись");
}

const technicalValidateOne=validateOne;
validateOne=function(item,value){return technicalValidateOne(item,value).map(humanizeValidation)};

const technicalProgress=progress;
progress=function(){technicalProgress();$("progress").textContent=$("progress").textContent.replace("сохранить","оставить").replace("exclude","исключить")};

const technicalSourcePanel=sourcePanel;
sourcePanel=function(item){
  technicalSourcePanel(item);
  const root=$("source"),row=item.task.original_row||{};
  const intro=Array.from(root.children).find(child=>child.tagName==="P");
  if(intro)intro.textContent="Это исходные материалы только для сравнения. Они могут содержать ошибки: подтверждайте сведения по независимым источникам.";
  const summary=node("dl",undefined,"source-summary");
  const add=(title,value)=>{if(value===undefined||value===null||value==="")return;summary.append(node("dt",title),node("dd",typeof value==="string"?value:JSON.stringify(value,null,2)))};
  add("ID примера",row.sample_id);
  add("ID статьи",row.paper_id);
  add("Исходный вопрос",row.model_task_prompt||row.prompt);
  add("Указанные изображения",row.images);
  if(summary.children.length&&intro)intro.after(summary);
  for(const heading of Array.from(root.querySelectorAll("h3"))){
    const names={"Коды findings":"Почему запись требует проверки","Проверенные при audit исходные изображения":"Исходные изображения из аудита — только для проверки","Исходная original_row (только чтение)":"Полная исходная запись","Устаревшие provenance-записи (только чтение)":"Старые сведения об источниках"};
    if(names[heading.textContent])heading.textContent=names[heading.textContent];
    const pre=heading.nextElementSibling;
    if(pre&&pre.tagName==="PRE"){
      const details=node("details",undefined,"source-details"),summaryNode=node("summary",heading.textContent);
      heading.replaceWith(details);
      details.append(summaryNode,pre);
    }
  }
  const overlap=Array.from(root.querySelectorAll(".meta")).find(value=>value.textContent.startsWith("Пересечение с training:"));
  if(overlap)overlap.textContent=overlap.textContent.replace("Пересечение с training:","Пересечение с обучающими данными:");
};

const technicalFormPanel=formPanel;
formPanel=function(item){
  technicalFormPanel(item);
  const root=$("form"),title=root.querySelector("h2");
  title.textContent="Решение по текущей записи";
  const intro=Array.from(root.children).find(child=>child.tagName==="P");
  if(intro){intro.textContent="Сначала выберите решение. Форма покажет только те поля, которые нужны для этого варианта.";intro.className="simple-intro"}
  title.after(expertGuide());

  const labels=()=>Array.from(root.querySelectorAll("label"));
  const decisionLabel=labels().find(label=>label.textContent.includes("Решение disposition"));
  if(decisionLabel){
    const select=decisionLabel.nextElementSibling,block=node("section",undefined,"decision-block");
    decisionLabel.textContent="1. Что сделать с этой записью? *";
    select.id=`expert-decision-${item.task.task_id}`;
    decisionLabel.htmlFor=select.id;
    select.setAttribute("aria-required","true");
    for(const option of select.options){if(option.value==="")option.textContent="Выберите решение";if(option.value==="retain")option.textContent="Оставить после полной проверки";if(option.value==="exclude")option.textContent="Исключить из датасета"}
    decisionLabel.before(block);
    block.append(decisionLabel,select,node("div","Оставляйте запись только если все научные поля и изображения можно независимо подтвердить. Иначе выберите исключение и укажите причину.","field-help"));
  }

  const stratumLabel=labels().find(label=>label.textContent.includes("Страта stratum"));
  if(stratumLabel){
    const select=stratumLabel.nextElementSibling;
    stratumLabel.textContent="Тип задания *";
    for(const option of select.options){if(option.value==="multimodal_hard")option.textContent="Сложное мультимодальное";if(option.value==="temporal_hard")option.textContent="Сложное временное";if(option.value==="easy_control")option.textContent="Простой контрольный пример"}
    select.after(node("div","Выберите тип по содержанию нового вопроса: анализ нескольких видов данных, временной динамики или простой контроль.","field-help"));
  }

  const imageBoxes=Array.from(root.querySelectorAll(".image-form"));
  imageBoxes.forEach((imageBox,index)=>enhanceImageForm(imageBox,edit().images[index]));
  for(const heading of root.querySelectorAll("h3")){
    if(heading.textContent==="Изображения replacement")heading.textContent="3. Проверенные изображения";
    if(heading.textContent==="Необязательный gold_answer")heading.textContent="Дополнительно: эталонный ответ";
    if(heading.textContent==="Предварительная проверка в браузере"){heading.textContent="Проверка текущей записи";heading.classList.add("validation-heading")}
  }
  for(const button of root.querySelectorAll("button")){
    if(button.textContent==="Скопировать только sample_id и paper_id из источника")button.textContent="Подставить исходные ID для проверки";
    if(button.textContent==="Добавить изображение")button.textContent="Добавить ещё изображение";
  }
  const copyButton=Array.from(root.querySelectorAll("button")).find(button=>button.textContent==="Подставить исходные ID для проверки");
  if(copyButton){
    const copyIntro=copyButton.previousElementSibling;
    const replacementHeading=node("h3","2. Проверенная новая запись");
    if(copyIntro&&copyIntro.tagName==="P")copyIntro.before(replacementHeading);else copyButton.before(replacementHeading);
  }
  for(const paragraph of root.querySelectorAll("p"))if(paragraph.textContent.startsWith("Поля replacement"))paragraph.textContent="Можно подставить исходные ID как черновик, но оба значения всё равно нужно сверить с независимым источником.";

  const goldLabel=labels().find(label=>label.textContent.includes("Добавить adjudicated gold_answer"));
  if(goldLabel){const checkbox=goldLabel.querySelector("input");goldLabel.replaceChildren(checkbox,document.createTextNode(" Добавить независимо согласованный эталонный ответ"));goldLabel.after(node("div","Обычно не требуется. Включайте только если протокол требует автоматических метрик по gold answer.","field-help"))}

  const attestationLabel=labels().find(label=>label.textContent.includes("два указанных эксперта независимо проверили"));
  const reviewers=root.querySelector('[data-field-group="reviewed_by"]');
  const notes=root.querySelector('[data-field-group="notes"]');
  const validationHeading=Array.from(root.querySelectorAll("h3")).find(heading=>heading.textContent==="Проверка текущей записи");
  if(attestationLabel&&reviewers&&notes){
    const checkbox=attestationLabel.querySelector("input"),confirmation=node("section",undefined,"confirmation-box");
    attestationLabel.replaceChildren(checkbox,document.createTextNode(" Подтверждаем, что два указанных эксперта независимо проверили это решение *"));
    attestationLabel.className="attestation-box";
    const attestHelp=node("div","Ставьте отметку только после двух реальных независимых проверок. Форма не может подтвердить личности экспертов автоматически.","field-help");
    confirmation.append(node("h3","Последний шаг: подтверждение экспертов"),node("p","Этот блок обязателен и для оставленной, и для исключённой записи.","simple-intro"),reviewers,attestationLabel,attestHelp,notes);
    if(validationHeading)validationHeading.before(confirmation);else root.append(confirmation);
  }

  const bindingLabel=Array.from(root.querySelectorAll(".meta")).find(value=>value.textContent==="Неизменяемая привязка к очереди");
  const bindings=Array.from(root.querySelectorAll(".binding"));
  if(bindings.length){const details=node("details",undefined,"technical-details");details.append(node("summary","Техническая привязка к очереди — обычно не требуется"));if(bindingLabel)details.append(bindingLabel);details.append(...bindings);root.append(details)}
};

document.title="Экспертная проверка научных примеров";
const headerTitle=document.querySelector("header h1");
if(headerTitle)headerTitle.textContent="Экспертная проверка научных примеров";
const searchInput=$("search");
searchInput.placeholder="Найти задачу по ID или содержимому";
const filterLabels={all:"Все задачи",pending:"Без решения",retain:"Оставленные",exclude:"Исключённые",invalid:"Требуют исправления",complete:"Заполнены без ошибок"};
for(const option of $("filter").options)option.textContent=filterLabels[option.value]||option.textContent;
$("export-draft").textContent="Скачать черновик";
$("export-final").textContent="Скачать готовые решения";
$("clear").textContent="Очистить всю локальную работу";
resetInvalidRetainsButton.textContent="Сбросить незавершённые сохранения";
const importInput=$("import-draft"),importLabel=importInput.parentElement;
importLabel.replaceChildren(document.createTextNode("Загрузить черновик "),importInput);
render();
"""

_SCRIPT += _EXPERT_FORM_SCRIPT


def _html(payload: Mapping[str, Any]) -> bytes:
    document = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src 'self' blob:; script-src 'nonce-curator-v1'; style-src 'nonce-curator-v1'; connect-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'">
<title>Офлайн-курация VLM benchmark</title><style nonce="curator-v1">{_STYLE}</style></head>
<body><header><h1>Офлайн-курация benchmark</h1><div>queue fingerprint: <code>{payload["queue_fingerprint"]}</code></div><p>Сверьте fingerprint с handoff. Финальный completed_decisions.jsonl обязательно проверьте командой curate-assemble; проверка в браузере только предварительная.</p><p class="meta">Программа проверяет queue bindings и разные declared ID, но не может доказать личность или независимость людей. Release owner хранит внешний подписанный или датированный review record.</p><div id="storage-warning" class="warning hidden"></div><div class="toolbar"><input id="search" type="search" placeholder="Поиск по содержимому задач"><select id="filter"><option value="all">Все задачи</option><option value="pending">Не заполнены</option><option value="retain">Сохранить</option><option value="exclude">Исключить</option><option value="invalid">С ошибками</option><option value="complete">Готовы предварительно</option></select><span id="progress" class="progress"></span></div><div class="toolbar"><button id="prev">Предыдущая</button><span id="position"></span><button id="next">Следующая</button><span class="spacer"></span><button id="export-draft">Экспортировать черновик</button><label>Объединить черновик <input id="import-draft" type="file" accept="application/json"></label><button id="clear" class="danger">Очистить сохранённую работу</button><button id="export-final" class="primary">Экспортировать completed_decisions.jsonl</button></div><div id="global-error" class="error"></div></header>
<main><div class="grid"><aside id="source" class="card source"></aside><section id="form" class="card"></section></div></main>
<script id="workspace-data" type="application/json" nonce="curator-v1">{_embedded_json(payload)}</script><script nonce="curator-v1">{_SCRIPT}</script></body></html>"""
    return document.encode("utf-8")


def generate_curator_workspace(
    config: Mapping[str, Any],
    prepare_manifest: str | Path,
    queue_manifest: str | Path,
    output_dir: str | Path,
    *,
    benchmark_schema: str | Path,
    provenance_schema: str | Path,
) -> dict[str, Any]:
    """Verify an immutable queue and publish a deterministic offline curator workspace."""

    material = _queue_material(config, prepare_manifest, benchmark_schema, provenance_schema)
    queue_root, verified_queue = _verify_queue_workspace(material, queue_manifest)
    inventory, copies = _source_inventory(material)
    protected_directories, protected_files = _prepare_input_protection(material)
    target = _separate_output_target(
        output_dir,
        protected_directories=(queue_root, *protected_directories),
        protected_files=protected_files,
    )
    html = _html(_browser_payload(material, verified_queue, inventory))
    manifest = {
        "artifact_version": CURATOR_ARTIFACT_VERSION,
        "queue_fingerprint": verified_queue["queue_fingerprint"],
        "task_count": verified_queue["task_count"],
        "html_file": CURATOR_HTML,
        "html_sha256": _sha256_bytes(html),
        "source_images": inventory,
    }
    manifest_bytes = (
        json.dumps(manifest, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    staging = target.with_name(f".{target.name}.curator.{os.getpid()}.tmp")
    if staging.exists() or staging.is_symlink():
        raise RemediationError(f"curator staging path already exists: {staging}")
    try:
        staging.mkdir()
        _atomic_write(staging / CURATOR_HTML, html)
        for relative, source in sorted(copies.items()):
            _atomic_copy(source, staging.joinpath(*relative.split("/")))
        inventory_by_path = {entry["path"]: entry for entry in inventory}
        for relative in copies:
            digest, size = _sha256_file(staging.joinpath(*relative.split("/")))
            expected = inventory_by_path[relative]
            if digest != expected["sha256"] or size != expected["size_bytes"]:
                raise RemediationError(f"copied source image changed unexpectedly: {relative}")
        _atomic_write(staging / CURATOR_MANIFEST, manifest_bytes)
        if target.exists():
            if _trees_identical(staging, target):
                return manifest
            raise RemediationError("output_dir already contains a different curator workspace")
        if target.is_symlink():
            raise RemediationError("output_dir is a symlink")
        try:
            os.replace(staging, target)
        except OSError as exc:
            if target.is_dir() and _trees_identical(staging, target):
                return manifest
            raise RemediationError(f"cannot publish curator workspace atomically: {exc}") from exc
    finally:
        if staging.exists() and staging.is_dir():
            shutil.rmtree(staging)
    return manifest


__all__ = ["generate_curator_workspace"]
