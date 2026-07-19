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
    _output_target,
    _queue_material,
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
    raw = Path(output_dir)
    try:
        candidate = raw.resolve(strict=False)
        queue = queue_root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RemediationError(f"cannot resolve curator workspace paths: {exc}") from exc
    if candidate == queue or queue in candidate.parents or candidate in queue.parents:
        raise RemediationError("output_dir must be separate from and must not contain the queue")
    return _output_target(raw)


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


_SCRIPT = r"""
"use strict";
const DATA=JSON.parse(document.getElementById("workspace-data").textContent);
const KEY="vlm-curator:"+DATA.queue_fingerprint;
const MAX_DRAFT_BYTES=32*1024*1024;
const FIELDS=["artifact_version","queue_fingerprint","task_id","prepare_manifest_sha256","source_benchmark_sha256","source_provenance_sha256","source_audit_sha256","source_row_index","source_row_sha256","status","disposition","exclusion_reason","benchmark_row","provenance_row","reviewed_by","notes"];
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
function sourcePanel(item){const root=$("source");root.replaceChildren(node("h2",`Задача ${item.task.source_row_index+1} из ${DATA.task_count}`),node("div",`task_id: ${item.task.task_id}`,"binding"),node("p","Исходные данные ниже доступны только для чтения и могут содержать ошибки. Не переносите их в replacement без независимой проверки. Сверьте queue fingerprint вверху страницы с handoff."));const codes=node("div",undefined,"codes");for(const value of [...item.task.critical_codes,...item.task.warning_codes])codes.append(node("span",value,"code"));root.append(node("h3","Коды findings"),codes,node("div","Пересечение с training: "+(item.task.training_overlap_paper_ids.join(", ")||"не обнаружено"),"meta"));const previews=node("div",undefined,"preview");for(const image of item.task.audited_image_hashes){const box=node("div"),img=node("img");img.src=image.preview_path;img.alt=image.path;box.append(img,node("div",`${image.path}\n${image.sha256}`,"meta"));previews.append(box)}root.append(node("h3","Проверенные при audit исходные изображения"),previews,node("h3","Исходная original_row (только чтение)"),node("pre",JSON.stringify(item.task.original_row,null,2)),node("h3","Устаревшие provenance-записи (только чтение)"),node("pre",JSON.stringify(item.task.legacy_provenance_rows,null,2)))}
function imageForm(parent,image,index){const box=node("section",undefined,"image-form"),title=node("div",`Изображение ${index+1}`,"row"),remove=node("button","Удалить","danger");remove.type="button";remove.addEventListener("click",()=>{discardImageRuntime(image);edit().images.splice(index,1);save();render()});title.append(remove);box.append(title,node("p","Укажите canonical image_path самостоятельно. После экспорта отдельно скопируйте выбранный файл в $Curated по пути assets/images/...; имя файла не подставляется автоматически."));let shaInput;for(const [label,key] of [["image_path (assets/images/...)","image_path"],["SHA256 в нижнем регистре","sha256"],["Страница","page"],["Точный locator (рисунок, панель, таблица)","locator"],["URL источника","source_url"],["Лицензия","license"],["Библиографическая citation","citation"],["verified_by (по одному ASCII ID в строке)","verified_by"]]){const l=node("label",label),el=key==="citation"?node("textarea"):node("input");el.value=image[key]||"";el.addEventListener("input",()=>{image[key]=el.value;save()});if(key==="sha256")shaInput=el;box.append(l,el)}const fileLabel=node("label","Выбрать проверенный файл и вычислить SHA256"),fileInput=node("input");fileInput.type="file";fileInput.accept="image/*";const fileStatus=node("div","Файл не выбран. SHA256 можно ввести вручную.","meta"),localPreview=node("img");localPreview.alt="Локальный preview выбранного файла";const existingRuntime=imageRuntime.get(image);if(existingRuntime&&existingRuntime.pending){fileStatus.textContent="Вычисляется SHA256 выбранного файла…";fileStatus.className="warning";localPreview.className="hidden"}else if(existingRuntime&&existingRuntime.error){fileStatus.textContent=existingRuntime.error;fileStatus.className="error";localPreview.className="hidden"}else if(existingRuntime&&existingRuntime.url){localPreview.src=existingRuntime.url;fileStatus.textContent=`Выбран файл: ${existingRuntime.name}, ${existingRuntime.size} байт`}else localPreview.className="hidden";fileInput.addEventListener("change",async()=>{const file=fileInput.files&&fileInput.files[0];if(!file)return;const token=beginHash(image);localPreview.removeAttribute("src");localPreview.className="hidden";fileStatus.textContent="Вычисляется SHA256 выбранного файла…";fileStatus.className="warning";if(!globalThis.crypto||!crypto.subtle||typeof crypto.subtle.digest!=="function"){const runtime=finishHash(image,token);if(runtime){runtime.error="Web Crypto недоступен. Введите SHA256 вручную.";fileStatus.textContent=runtime.error;fileStatus.className="error"}return}try{const buffer=await file.arrayBuffer();if(!currentHash(image,token,fileInput,file))return;const digest=await crypto.subtle.digest("SHA-256",buffer);if(!currentHash(image,token,fileInput,file))return;const runtime=finishHash(image,token);if(!runtime)return;image.sha256=Array.from(new Uint8Array(digest),value=>value.toString(16).padStart(2,"0")).join("");shaInput.value=image.sha256;save();runtime.name=file.name;runtime.size=file.size;try{runtime.url=URL.createObjectURL(file);localPreview.src=runtime.url;localPreview.className=""}catch(_previewError){runtime.url=null;localPreview.className="hidden"}fileStatus.textContent=`SHA256 вычислен. Выбран файл: ${file.name}, ${file.size} байт. Скопируйте эти bytes в $Curated по введённому image_path.`;fileStatus.className="ok"}catch(_error){if(!currentHash(image,token,fileInput,file))return;const runtime=finishHash(image,token);if(runtime){runtime.error="Не удалось прочитать файл или вычислить SHA256. Введите SHA256 вручную.";fileStatus.textContent=runtime.error;fileStatus.className="error"}}});fileLabel.append(fileInput);box.append(fileLabel,fileStatus,localPreview);parent.append(box)}
function formPanel(item){const e=edit(),root=$("form");root.replaceChildren();root.append(node("h2","Решение экспертов"),node("p","Исходная строка может быть ошибочной. Сохранение требует независимо проверенный полный replacement; исключение требует конкретную проверяемую причину. Сверьте queue fingerprint с handoff. Предварительная проверка браузера не заменяет curate-assemble."),node("div","Неизменяемая привязка к очереди","meta"),node("div",JSON.stringify(item.binding),"binding"));const d=node("select");for(const [v,t] of [["","Выберите решение"],["retain","Сохранить с полной заменой"],["exclude","Исключить"]]){const o=node("option",t);o.value=v;o.selected=e.disposition===v;d.append(o)}d.addEventListener("change",()=>{e.disposition=d.value;save();render()});root.append(node("label","Решение disposition"),d);field(root,"reviewed_by (по одному ASCII ID в строке)","reviewed_by","textarea");const attestLabel=node("label"),attest=node("input");attest.type="checkbox";attest.style.width="auto";attest.checked=e.independent_attestation;attest.addEventListener("change",()=>{e.independent_attestation=attest.checked;save();render()});attestLabel.append(attest,document.createTextNode(" Подтверждаю: два указанных эксперта независимо проверили это решение"));root.append(attestLabel);field(root,"Примечания notes","notes","textarea");if(e.disposition==="exclude"){field(root,"Конкретная причина исключения exclusion_reason","exclusion_reason","textarea")}if(e.disposition==="retain"){const copy=node("button","Скопировать только sample_id и paper_id из источника");copy.type="button";copy.addEventListener("click",()=>{e.sample_id=String(item.task.original_row.sample_id||"");e.paper_id=String(item.task.original_row.paper_id||"");save();render()});root.append(node("p","Поля replacement изначально пусты. Кнопка ниже явно копирует только идентификаторы; их всё равно нужно проверить."),copy);const two=node("div",undefined,"two");field(two,"sample_id","sample_id");field(two,"Канонический paper_id","paper_id");root.append(two);const sl=node("label","Страта stratum"),s=node("select");for(const v of ["multimodal_hard","temporal_hard","easy_control"]){const o=node("option",v);o.value=v;o.selected=e.stratum===v;s.append(o)}s.addEventListener("change",()=>{e.stratum=s.value;save()});root.append(sl,s);field(root,"Самодостаточный запрос пользователю prompt","prompt","textarea");field(root,"Системная инструкция (необязательно)","system_instruction","textarea");const ids=node("div",undefined,"two");field(ids,"source_document_id","source_document_id");field(ids,"creator_group_id","creator_group_id");root.append(ids,node("h3","Изображения replacement"));e.images.forEach((image,index)=>imageForm(root,image,index));const add=node("button","Добавить изображение");add.type="button";add.addEventListener("click",()=>{e.images.push({image_path:"",sha256:"",page:"",locator:"",source_url:"",license:"",citation:"",verified_by:""});save();render()});root.append(add);const goldLabel=node("label"),gold=node("input");gold.type="checkbox";gold.checked=e.gold_enabled;gold.style.width="auto";gold.addEventListener("change",()=>{e.gold_enabled=gold.checked;save();render()});goldLabel.append(gold,document.createTextNode(" Добавить adjudicated gold_answer и rubric"));root.append(node("h3","Необязательный gold_answer"),goldLabel);if(e.gold_enabled){field(root,"Эталонный ответ gold_answer","gold_answer","textarea");field(root,"Использованные свидетельства (по одному в строке)","evidence_used","textarea");field(root,"Визуальные факты (по одному в строке)","visual_facts","textarea");field(root,"Временные факты (по одному в строке)","temporal_facts","textarea");field(root,"Неопределённость (необязательно)","uncertainty","textarea");field(root,"Недостающие свидетельства (необязательно)","missing_evidence","textarea");field(root,"Критерии rubric (по одному в строке)","criteria","textarea");field(root,"adjudicators (по одному ASCII ID в строке)","adjudicators","textarea")}}const errors=validateOne(item,e),status=node("div",errors.length?errors.join("\n"):e.disposition?"Предварительная проверка решения пройдена.":"Решение ещё не заполнено.",errors.length?"error":e.disposition?"ok":"meta");root.append(node("h3","Предварительная проверка в браузере"),status)}
function normalizedIdentity(value){return value.normalize("NFKC").trim().replace(/\s+/gu," ").toLowerCase()}function ids(value,label,errors){const values=lines(value),normalized=values.map(normalizedIdentity);if(values.length<2)errors.push(`${label}: нужны минимум два идентификатора`);if(normalized.some(x=>!/^[A-Za-z0-9](?:[A-Za-z0-9._:@/+ -]*[A-Za-z0-9])?$/.test(x)))errors.push(`${label}: после NFKC-нормализации используйте ASCII-идентификаторы без пробелов по краям`);if(new Set(normalized).size!==normalized.length)errors.push(`${label}: идентификаторы должны быть различными после NFKC, нормализации пробелов и регистра`);return values}
function validAssetPath(value){if(typeof value!=="string"||!value.startsWith("assets/images/")||value.includes("\\")||value.includes(":"))return false;const parts=value.split("/");if(parts.length<3||parts.some(part=>!part||part==="."||part===".."||part!==part.replace(/[ .]+$/,"")))return false;for(const part of parts){if(/[<>"|?*\x00-\x1f]/.test(part))return false;const base=part.split(".",1)[0].trimEnd().replace(/[¹²³]/g,d=>({"¹":"1","²":"2","³":"3"})[d]).toLowerCase();if(/^(aux|clock\$|con|conin\$|conout\$|nul|prn|com[1-9]|lpt[1-9])$/.test(base))return false}return true}
function validateOne(item,e){const errors=[];if(!["retain","exclude"].includes(e.disposition))return ["Выберите: сохранить replacement или исключить."];ids(e.reviewed_by,"reviewed_by",errors);if(!e.independent_attestation)errors.push("Подтвердите независимую проверку решения двумя указанными экспертами.");if(e.disposition==="exclude"){if(!e.exclusion_reason.trim())errors.push("Укажите конкретную причину исключения.");return errors}for(const [k,n] of [["sample_id","sample_id"],["paper_id","paper_id"],["source_document_id","source_document_id"],["creator_group_id","creator_group_id"]])if(!/^[\x21-\x7e](?:[\x20-\x7e]*[\x21-\x7e])?$/.test(e[k]))errors.push(`${n}: требуется ASCII-значение без пробелов по краям.`);if(!e.prompt.trim()||e.prompt!==e.prompt.trim())errors.push("prompt обязателен и не должен содержать пробелы по краям.");if(e.paper_id!==e.paper_id.toLowerCase()||!/^(?:doi:10\.\d{4,9}\/[^\s]+|arxiv:(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[a-z]{2})?\/\d{7})|paper:\S(?:[\x20-\x7e]*\S)?)$/.test(e.paper_id))errors.push("paper_id должен быть каноническим в нижнем регистре: doi:/arxiv: без пробелов либо paper: с внутренними пробелами.");if(!e.images.length)errors.push("Добавьте минимум одно изображение.");e.images.forEach((im,i)=>{const p=`Изображение ${i+1}`;if(!validAssetPath(im.image_path))errors.push(`${p}: image_path должен быть нормализованным Windows-safe путём внутри assets/images/.`);if(!/^[0-9a-f]{64}$/.test(im.sha256))errors.push(`${p}: требуется lowercase SHA256.`);for(const k of ["page","locator","license","citation"])if(!String(im[k]||"").trim())errors.push(`${p}: заполните ${k}.`);try{const u=new URL(im.source_url);if(!["http:","https:"].includes(u.protocol))throw 0}catch(_error){errors.push(`${p}: source_url должен быть HTTP(S) URL.`)}ids(im.verified_by,`${p} verified_by`,errors)});if(e.gold_enabled||DATA.policy.require_gold){if(!e.gold_answer.trim()||!lines(e.evidence_used).length||!lines(e.visual_facts).length||!lines(e.temporal_facts).length||!lines(e.criteria).length)errors.push("Полностью заполните gold_answer, evidence, visual/temporal facts и критерии rubric.");ids(e.adjudicators,"adjudicators",errors)}return errors}
function decision(item,e){const images=e.images.map(im=>({image_path:im.image_path,sha256:im.sha256,page:im.page.trim(),locator:im.locator,source_url:im.source_url,license:im.license,citation:im.citation,verified_by:lines(im.verified_by)}));const messages=[];if(e.system_instruction.trim())messages.push({role:"system",content:[{type:"text",text:e.system_instruction}]});messages.push({role:"user",content:[{type:"text",text:e.prompt},...images.map(()=>({type:"image"}))]});let benchmark={sample_id:e.sample_id,paper_id:e.paper_id,stratum:e.stratum,primary_endpoint:DATA.policy.primary_strata.includes(e.stratum),model_task_prompt:e.prompt,messages,images:images.map(x=>x.image_path),split_provenance:{paper_holdout:true,source_holdout:true,creator_holdout:true,training_overlap_checked:true,source_document_id:e.source_document_id,creator_group_id:e.creator_group_id}};if(e.gold_enabled||DATA.policy.require_gold){benchmark.gold_answer={answer:e.gold_answer,evidence_used:lines(e.evidence_used),visual_facts:lines(e.visual_facts),temporal_facts:lines(e.temporal_facts),uncertainty:e.uncertainty.trim()||null,missing_evidence:e.missing_evidence.trim()||null};benchmark.rubric={criteria:lines(e.criteria),adjudicators:lines(e.adjudicators)}}const result={...item.binding,status:"complete",disposition:e.disposition,exclusion_reason:e.disposition==="exclude"?e.exclusion_reason:null,benchmark_row:e.disposition==="retain"?benchmark:null,provenance_row:e.disposition==="retain"?{sample_id:e.sample_id,paper_id:e.paper_id,images}:null,reviewed_by:lines(e.reviewed_by),notes:e.notes};const ordered={};for(const key of FIELDS)ordered[key]=result[key];return ordered}
function download(name,text){const a=node("a");a.href=URL.createObjectURL(new Blob([text],{type:"application/json"}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}
function exportFinal(){if(pendingHashCount>0){$("global-error").textContent=`Финальный экспорт заблокирован: дождитесь вычисления SHA256 (${pendingHashCount}).`;return}const errors=[];const decisions=[],sampleIds=new Map();DATA.tasks.forEach((item,i)=>{const e=state.edits[item.task.task_id],bad=validateOne(item,e);if(e.disposition==="retain"){if(sampleIds.has(e.sample_id))bad.push(`sample_id уже использован в задаче ${sampleIds.get(e.sample_id)+1}.`);else sampleIds.set(e.sample_id,i)}if(bad.length)errors.push(`Задача ${i+1} (${item.task.task_id}): ${bad.join(" ")}`);else decisions.push(decision(item,e))});if(decisions.length!==DATA.task_count)errors.unshift(`Нужно завершить все ${DATA.task_count} задач.`);if(errors.length){$("global-error").textContent=errors.slice(0,30).join("\n")+(errors.length>30?`\n... ещё ${errors.length-30}`:"");return}$("global-error").textContent="";download("completed_decisions.jsonl",decisions.map(x=>JSON.stringify(x)).join("\n")+"\n")}
function sameEdit(left,right){return JSON.stringify(left)===JSON.stringify(right)}
function importDraft(file){if(file.size>MAX_DRAFT_BYTES){$("global-error").textContent="Импорт отклонён: размер черновика превышает 32 МиБ.";return}const reader=new FileReader();reader.onerror=()=>{$("global-error").textContent="Импорт отклонён: FileReader не смог прочитать черновик."};reader.onabort=()=>{$("global-error").textContent="Импорт отклонён: чтение черновика отменено."};reader.onload=()=>{try{if(typeof reader.result!=="string")throw new Error("черновик не является текстом");const envelope=JSON.parse(reader.result);if(!plain(envelope)||!exactKeys(envelope,["artifact_version","queue_fingerprint","edits"]))throw new Error("неверные поля envelope черновика");if(envelope.artifact_version!==DATA.artifact_version)throw new Error("неподдерживаемый artifact_version черновика");if(envelope.queue_fingerprint!==DATA.queue_fingerprint)throw new Error("черновик относится к другой очереди");if(!plain(envelope.edits))throw new Error("edits должен быть объектом с ключами task_id");const incoming={};for(const [taskId,value] of Object.entries(envelope.edits)){if(!taskIds.has(taskId))throw new Error(`неизвестный task_id ${taskId}`);incoming[taskId]=validateEdit(value,taskId)}const candidate={};for(const taskId of taskIds)candidate[taskId]=validateEdit(state.edits[taskId],taskId);const empty=blank(),summary={merged:0,equal:0,blank_ignored:0};for(const [taskId,value] of Object.entries(incoming)){const current=candidate[taskId];if(sameEdit(value,empty))summary.blank_ignored++;else if(sameEdit(current,empty)){candidate[taskId]=value;summary.merged++}else if(sameEdit(current,value))summary.equal++;else throw new Error(`конфликт merge для task_id ${taskId}; изменения не применены`)}revokeAllPreviews();state.edits=candidate;save();render();$("global-error").textContent=`Merge черновика завершён: добавлено ${summary.merged}, уже совпадало ${summary.equal}, пустых пропущено ${summary.blank_ignored}.`}catch(error){$("global-error").textContent="Импорт отклонён: "+(error instanceof SyntaxError?"некорректный JSON":error.message)}};try{reader.readAsText(file)}catch(_error){$("global-error").textContent="Импорт отклонён: не удалось начать чтение файла."}}
function render(){const item=DATA.tasks[state.index];sourcePanel(item);formPanel(item);$("position").textContent=`${state.index+1} / ${DATA.task_count}`;progress()}
$("prev").addEventListener("click",()=>move(-1));$("next").addEventListener("click",()=>move(1));$("search").addEventListener("input",render);$("filter").addEventListener("change",render);$("export-final").addEventListener("click",exportFinal);$("export-draft").addEventListener("click",()=>download("curator_draft.json",JSON.stringify({artifact_version:DATA.artifact_version,queue_fingerprint:DATA.queue_fingerprint,edits:state.edits})+"\n"));$("import-draft").addEventListener("change",event=>event.target.files[0]&&importDraft(event.target.files[0]));$("clear").addEventListener("click",()=>{if(confirm("Удалить все локально сохранённые решения для этой очереди?")){revokeAllPreviews();storageRemove();state={index:0,edits:{}};for(const taskId of taskIds)state.edits[taskId]=blank();save();render()}});window.addEventListener("beforeunload",revokeAllPreviews);render();
"""


def _html(payload: Mapping[str, Any]) -> bytes:
    document = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src 'self' blob:; script-src 'nonce-curator-v1'; style-src 'nonce-curator-v1'; connect-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'">
<title>Офлайн-курация VLM benchmark</title><style nonce="curator-v1">{_STYLE}</style></head>
<body><header><h1>Офлайн-курация benchmark</h1><div>queue fingerprint: <code>{payload["queue_fingerprint"]}</code></div><p>Сверьте fingerprint с handoff. Финальный completed_decisions.jsonl обязательно проверьте командой curate-assemble; проверка в браузере только предварительная.</p><div id="storage-warning" class="warning hidden"></div><div class="toolbar"><input id="search" type="search" placeholder="Поиск по содержимому задач"><select id="filter"><option value="all">Все задачи</option><option value="pending">Не заполнены</option><option value="retain">Сохранить</option><option value="exclude">Исключить</option><option value="invalid">С ошибками</option><option value="complete">Готовы предварительно</option></select><span id="progress" class="progress"></span></div><div class="toolbar"><button id="prev">Предыдущая</button><span id="position"></span><button id="next">Следующая</button><span class="spacer"></span><button id="export-draft">Экспортировать черновик</button><label>Объединить черновик <input id="import-draft" type="file" accept="application/json"></label><button id="clear" class="danger">Очистить сохранённую работу</button><button id="export-final" class="primary">Экспортировать completed_decisions.jsonl</button></div><div id="global-error" class="error"></div></header>
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
    target = _workspace_target(output_dir, queue_root)
    inventory, copies = _source_inventory(material)
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
