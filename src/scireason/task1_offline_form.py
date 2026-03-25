from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _default_state() -> dict[str, Any]:
    return {
        "_schema": "trajectory_form_state_v3",
        "expert": {"last_name": "", "first_name": "", "patronymic": "-"},
        "topic": "",
        "cutoff_year": 2020,
        "domain_query": "",
        "domain_language": "ru",
        "domain_qid": "Q336",
        "papers": [{"id": "", "year": 0, "title": ""}],
        "steps": [{
            "step_id": 1,
            "claim": "",
            "sources": [{"type": "text", "source": "", "page": "", "locator": "", "snippet_or_summary": ""}],
            "conditions": {"system": "unknown", "environment": "unknown", "protocol": "unknown", "notes": ""},
            "inference": "",
            "next_question": "",
        }],
        "edges": [],
        "selected_step_index": 0,
        "filename": "",
        "github": {"repo": "", "branch": "main", "path": "", "message": ""},
    }


_HTML_TEMPLATE = r'''<!doctype html>
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
      --text: #0f172a;
      --muted: #475569;
      --accent: #2563eb;
      --accent-soft: #dbeafe;
      --success: #166534;
      --success-soft: #dcfce7;
      --warning: #92400e;
      --warning-soft: #fef3c7;
      --danger: #b91c1c;
      --danger-soft: #fee2e2;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background: linear-gradient(180deg, #eff6ff 0%, var(--bg) 180px);
    }
    .container { max-width: 1200px; margin: 0 auto; padding: 24px; }
    .hero {
      background: linear-gradient(135deg, rgba(37,99,235,0.12), rgba(37,99,235,0.03));
      border: 1px solid rgba(37,99,235,0.16);
      border-radius: 20px;
      padding: 24px;
      margin-bottom: 18px;
      box-shadow: 0 20px 50px rgba(15, 23, 42, 0.08);
    }
    .hero h1 { margin: 0 0 8px; font-size: 28px; }
    .hero p { margin: 0; color: var(--muted); line-height: 1.5; }
    .status {
      margin-top: 12px;
      display: inline-flex;
      gap: 8px;
      align-items: center;
      padding: 10px 12px;
      border-radius: 999px;
      background: rgba(255,255,255,0.7);
      border: 1px solid rgba(148, 163, 184, 0.35);
      color: var(--muted);
      font-size: 14px;
    }
    .toolbar, .toolbar-secondary {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 14px;
    }
    button, .button-like {
      border: 0;
      border-radius: 12px;
      padding: 11px 16px;
      cursor: pointer;
      font-weight: 600;
      transition: transform .12s ease, box-shadow .12s ease, opacity .12s ease;
      box-shadow: 0 8px 18px rgba(15, 23, 42, 0.08);
      background: #ffffff;
      color: var(--text);
      border: 1px solid rgba(148, 163, 184, 0.28);
    }
    button:hover, .button-like:hover { transform: translateY(-1px); }
    button.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
    button.success { background: var(--success); color: #fff; border-color: var(--success); }
    button.warning { background: #b45309; color: #fff; border-color: #b45309; }
    button.danger { background: var(--danger); color: #fff; border-color: var(--danger); }
    button.ghost { background: #fff; color: var(--muted); }
    input[type=file] { display: none; }
    .card {
      background: var(--card);
      border: 1px solid rgba(148, 163, 184, 0.25);
      border-radius: 18px;
      padding: 18px;
      margin-bottom: 16px;
      box-shadow: 0 14px 35px rgba(15, 23, 42, 0.05);
    }
    .card h2, .card h3, .card h4 { margin-top: 0; }
    .section-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-bottom: 12px;
    }
    .section-head p { margin: 4px 0 0; color: var(--muted); }
    .grid { display: grid; gap: 12px; }
    .grid.cols-2 { grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }
    .grid.cols-3 { grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }
    label.field { display: flex; flex-direction: column; gap: 6px; font-weight: 600; font-size: 14px; }
    label.field span.meta { color: var(--muted); font-weight: 500; }
    input[type=text], input[type=number], select, textarea {
      width: 100%;
      border-radius: 12px;
      border: 1px solid rgba(148, 163, 184, 0.4);
      padding: 11px 12px;
      font: inherit;
      color: var(--text);
      background: #fff;
    }
    textarea { min-height: 96px; resize: vertical; }
    input[readonly] { background: #f8fafc; }
    .muted { color: var(--muted); }
    .hint { color: var(--muted); font-size: 13px; }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
    }
    .pill.info { background: #e0f2fe; color: #075985; }
    .pill.success { background: var(--success-soft); color: var(--success); }
    .pill.warning { background: var(--warning-soft); color: var(--warning); }
    .callout {
      border-radius: 14px;
      padding: 12px 14px;
      margin-top: 10px;
      line-height: 1.45;
      border: 1px solid transparent;
    }
    .callout.info { background: #eff6ff; color: #1d4ed8; border-color: #bfdbfe; }
    .callout.success { background: var(--success-soft); color: var(--success); border-color: #86efac; }
    .callout.warning { background: var(--warning-soft); color: var(--warning); border-color: #fcd34d; }
    .callout.error { background: var(--danger-soft); color: var(--danger); border-color: #fca5a5; }
    .paper-row, .source-card, .step-card {
      border: 1px solid rgba(148, 163, 184, 0.28);
      border-radius: 16px;
      padding: 14px;
      background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,0.98));
      margin-bottom: 12px;
    }
    .step-card { padding: 16px; }
    .step-header, .subsection-header {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      margin-bottom: 10px;
    }
    .step-title { font-size: 18px; font-weight: 700; }
    .step-claim-preview { color: var(--muted); font-size: 13px; }
    .mini-actions { display: flex; flex-wrap: wrap; gap: 8px; }
    .mini-actions button { padding: 8px 11px; border-radius: 10px; font-size: 13px; }
    .edge-table-wrap { overflow-x: auto; }
    table.edge-table {
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      min-width: 720px;
    }
    .edge-table th, .edge-table td {
      border-bottom: 1px solid rgba(148,163,184,.2);
      padding: 10px 12px;
      vertical-align: top;
      background: white;
    }
    .edge-table th {
      position: sticky;
      top: 0;
      background: #f8fafc;
      z-index: 1;
      font-size: 13px;
      text-align: left;
    }
    .edge-table td.center { text-align: center; }
    .edge-check { width: 18px; height: 18px; }
    .results-list { display: grid; gap: 8px; margin-top: 10px; }
    .result-option {
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid rgba(148,163,184,.25);
      background: #fff;
      cursor: pointer;
      text-align: left;
    }
    .result-option:hover { border-color: #93c5fd; box-shadow: 0 6px 15px rgba(37,99,235,.08); }
    .result-option strong { display: block; }
    .footer-note { color: var(--muted); font-size: 13px; margin-top: 10px; }
    .hidden { display: none !important; }
    code {
      padding: 2px 5px;
      border-radius: 6px;
      background: #e2e8f0;
      font-size: 0.95em;
    }
    @media (max-width: 760px) {
      .container { padding: 16px; }
      .hero { padding: 18px; }
      .section-head { flex-direction: column; align-items: flex-start; }
    }
  </style>
</head>
<body>
  <div class="container">
    <section class="hero">
      <h1>Task 1 — автономная форма Reasoning Trajectories</h1>
      <p>Эту HTML-форму можно скачать из ноутбука и заполнять локально на устройстве. Она хранит черновик в браузере, умеет экспортировать текущую форму в <code>.yaml</code> и выгружать / загружать черновик как <code>.json</code>.</p>
      <div class="status" id="autosaveStatus">Черновик ещё не сохранён</div>
    </section>

    <section class="toolbar">
      <button class="primary" id="exportYamlBtn">Скачать YAML</button>
      <button class="success" id="saveDraftBtn">Скачать черновик JSON</button>
      <label class="button-like" for="loadDraftInput">Загрузить черновик JSON</label>
      <input type="file" id="loadDraftInput" accept=".json,application/json">
      <button class="ghost" id="restoreBtn">Восстановить из автосохранения</button>
      <button class="warning" id="clearDraftBtn">Сбросить автосохранение</button>
    </section>

    <section class="card">
      <div class="section-head">
        <div>
          <h2>Эксперт и основная информация</h2>
          <p>Эти поля нужны для формирования <code>submission_id</code> и итогового артефакта.</p>
        </div>
        <span class="pill info">artifact v2</span>
      </div>
      <div class="grid cols-3">
        <label class="field"><span>Фамилия*</span><input type="text" id="expertLast" placeholder="Иванов"></label>
        <label class="field"><span>Имя*</span><input type="text" id="expertFirst" placeholder="Иван"></label>
        <label class="field"><span>Отчество*</span><input type="text" id="expertPat" placeholder="Иванович или -"></label>
      </div>
      <div class="grid cols-2" style="margin-top:12px;">
        <label class="field"><span>Topic*</span><input type="text" id="topicInput" placeholder="Коротко: о чём траектория"></label>
        <label class="field"><span>Cutoff year*</span><input type="number" id="cutoffYearInput" min="1800" max="2100" step="1"></label>
      </div>
      <div class="grid cols-2" style="margin-top:12px;">
        <label class="field"><span>filename</span><input type="text" id="filenameInput" placeholder="Оставьте пустым — имя сформируется автоматически"></label>
        <label class="field"><span>Предпросмотр ID</span><input type="text" id="submissionPreview" readonly></label>
      </div>
    </section>

    <section class="card">
      <div class="section-head">
        <div>
          <h2>Домен (Wikidata QID)</h2>
          <p>Можно ввести QID вручную или выполнить поиск через Wikidata прямо из формы.</p>
        </div>
        <span class="pill success" id="domainBadge">domain ready</span>
      </div>
      <div class="grid cols-3">
        <label class="field" style="grid-column: span 2;"><span>Wikidata search</span><input type="text" id="domainQuery" placeholder="Science / Наука / Lithium-ion battery"></label>
        <label class="field"><span>Lang</span><select id="domainLanguage"><option value="ru">Русский (ru)</option><option value="en">English (en)</option></select></label>
      </div>
      <div class="toolbar-secondary" style="margin-top:12px; margin-bottom:0;">
        <button class="ghost" id="domainSearchBtn">Найти в Wikidata</button>
      </div>
      <div class="results-list" id="wikidataResults"></div>
      <div class="grid cols-2" style="margin-top:12px;">
        <label class="field"><span>Domain (QID)*</span><input type="text" id="domainQid" placeholder="Q336"></label>
        <label class="field"><span>Обязательные conditions для выбранного домена</span><input type="text" id="requiredConditions" readonly></label>
      </div>
      <div class="callout info" id="wikidataStatus">По умолчанию используется домен <code>Q336</code>, если ничего не выбрано.</div>
    </section>

    <section class="card">
      <div class="section-head">
        <div>
          <h2>Публикации</h2>
          <p>Для каждой публикации заполните <code>id</code>, <code>year</code> и <code>title</code>.</p>
        </div>
        <div class="mini-actions">
          <button class="ghost" id="addPaperBtn">Добавить публикацию</button>
        </div>
      </div>
      <div id="papersContainer"></div>
    </section>

    <section class="card">
      <div class="section-head">
        <div>
          <h2>Шаги reasoning trajectory</h2>
          <p>Каждый шаг включает claim, набор источников, conditions, inference и next question.</p>
        </div>
        <div class="mini-actions">
          <button class="ghost" id="addStepBtn">Добавить шаг</button>
        </div>
      </div>
      <div id="stepsContainer"></div>
    </section>

    <section class="card">
      <div class="section-head">
        <div>
          <h2>Связи между шагами</h2>
          <p>Отмечайте направленные связи между шагами; петли <code>i → i</code> недоступны.</p>
        </div>
        <span class="pill warning" id="edgeCountPill">0 edges</span>
      </div>
      <div class="edge-table-wrap" id="edgesMatrixWrap"></div>
    </section>

    <section class="card">
      <div class="section-head">
        <div>
          <h2>Что именно сохраняется</h2>
          <p>Автосохранение хранится в <code>localStorage</code> браузера, а финальный YAML скачивается как локальный файл.</p>
        </div>
      </div>
      <div class="callout info">Загрузка / выгрузка HTML-файла работает через браузерные <code>Blob</code>-объекты и ссылку из <code>URL.createObjectURL()</code>; черновики формы сохраняются в <code>localStorage</code>; импорт JSON-черновика читается через <code>FileReader</code>.</div>
      <div class="footer-note">Совет: после завершения разметки скачайте итоговый YAML и при необходимости отдельно сохраните JSON-черновик как резервную копию.</div>
    </section>

    <div id="globalMessage"></div>
  </div>

  <script>
  const INITIAL_STATE = __INITIAL_STATE__;
  const DOMAIN_CONFIGS = __DOMAIN_CONFIGS__;
  const STORAGE_KEY = '__STORAGE_KEY__';
  const FORM_SCHEMA = 'trajectory_form_state_v3';
  const DEFAULT_DOMAIN_QID = (DOMAIN_CONFIGS.find((x) => x.wikidata_qid) || {}).wikidata_qid || 'Q336';

  const dom = {
    autosaveStatus: document.getElementById('autosaveStatus'),
    globalMessage: document.getElementById('globalMessage'),
    exportYamlBtn: document.getElementById('exportYamlBtn'),
    saveDraftBtn: document.getElementById('saveDraftBtn'),
    loadDraftInput: document.getElementById('loadDraftInput'),
    restoreBtn: document.getElementById('restoreBtn'),
    clearDraftBtn: document.getElementById('clearDraftBtn'),
    expertLast: document.getElementById('expertLast'),
    expertFirst: document.getElementById('expertFirst'),
    expertPat: document.getElementById('expertPat'),
    topicInput: document.getElementById('topicInput'),
    cutoffYearInput: document.getElementById('cutoffYearInput'),
    filenameInput: document.getElementById('filenameInput'),
    submissionPreview: document.getElementById('submissionPreview'),
    domainQuery: document.getElementById('domainQuery'),
    domainLanguage: document.getElementById('domainLanguage'),
    domainSearchBtn: document.getElementById('domainSearchBtn'),
    wikidataResults: document.getElementById('wikidataResults'),
    wikidataStatus: document.getElementById('wikidataStatus'),
    domainQid: document.getElementById('domainQid'),
    requiredConditions: document.getElementById('requiredConditions'),
    domainBadge: document.getElementById('domainBadge'),
    papersContainer: document.getElementById('papersContainer'),
    addPaperBtn: document.getElementById('addPaperBtn'),
    stepsContainer: document.getElementById('stepsContainer'),
    addStepBtn: document.getElementById('addStepBtn'),
    edgesMatrixWrap: document.getElementById('edgesMatrixWrap'),
    edgeCountPill: document.getElementById('edgeCountPill'),
  };

  const CYR = {
    'а':'a','б':'b','в':'v','г':'g','д':'d','е':'e','ё':'e','ж':'zh','з':'z','и':'i','й':'i','к':'k','л':'l','м':'m','н':'n','о':'o','п':'p','р':'r','с':'s','т':'t','у':'u','ф':'f','х':'kh','ц':'ts','ч':'ch','ш':'sh','щ':'shch','ы':'y','э':'e','ю':'yu','я':'ya','ь':'','ъ':''
  };

  function deepClone(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function defaultSource() {
    return { type: 'text', source: '', page: '', locator: '', snippet_or_summary: '' };
  }

  function defaultStep(index) {
    return {
      step_id: index,
      claim: '',
      sources: [defaultSource()],
      conditions: { system: 'unknown', environment: 'unknown', protocol: 'unknown', notes: '' },
      inference: '',
      next_question: '',
    };
  }

  function defaultPaper() {
    return { id: '', year: 0, title: '' };
  }

  function defaultState() {
    return {
      _schema: FORM_SCHEMA,
      expert: { last_name: '', first_name: '', patronymic: '-' },
      topic: '',
      cutoff_year: 2020,
      domain_query: '',
      domain_language: 'ru',
      domain_qid: DEFAULT_DOMAIN_QID,
      papers: [defaultPaper()],
      steps: [defaultStep(1)],
      edges: [],
      selected_step_index: 0,
      filename: '',
      github: { repo: '', branch: 'main', path: '', message: '' },
    };
  }

  function slugify(text) {
    const prepared = transliterate(text || '').toLowerCase();
    return prepared.replace(/[^a-z0-9]+/g, '_').replace(/_+/g, '_').replace(/^_+|_+$/g, '').slice(0, 60) || 'trajectory';
  }

  function transliterate(text) {
    const src = String(text || '');
    let out = '';
    for (const ch of src) {
      const low = ch.toLowerCase();
      let repl = CYR[low];
      if (typeof repl === 'undefined') {
        out += ch;
        continue;
      }
      if (ch !== low && repl) {
        repl = repl.charAt(0).toUpperCase() + repl.slice(1);
      }
      out += repl;
    }
    return out;
  }

  function cleanTextPreview(text, limit = 48) {
    const value = String(text || '').replace(/\s+/g, ' ').trim();
    if (value.length <= limit) return value;
    return value.slice(0, Math.max(1, limit - 1)).trimEnd() + '…';
  }

  function nowUtc() {
    return new Date().toISOString().replace(/\.\d{3}Z$/, 'Z');
  }

  function domainConfigFor(qid) {
    return DOMAIN_CONFIGS.find((item) => String(item.wikidata_qid || '').trim() === String(qid || '').trim()) || null;
  }

  function requiredConditionKeys(qid) {
    const cfg = domainConfigFor(qid);
    const keys = (cfg && Array.isArray(cfg.required_conditions)) ? cfg.required_conditions.slice() : [];
    return keys;
  }

  function normalizeState(raw) {
    const base = defaultState();
    const state = Object.assign({}, base, raw || {});
    state.expert = Object.assign({}, base.expert, state.expert || {});
    state.github = Object.assign({}, base.github, state.github || {});
    state.domain_language = ['ru', 'en'].includes(state.domain_language) ? state.domain_language : 'ru';
    state.domain_qid = String(state.domain_qid || DEFAULT_DOMAIN_QID || 'Q336').trim();
    state.papers = Array.isArray(state.papers) && state.papers.length ? state.papers.map((paper) => ({
      id: String((paper || {}).id || ''),
      year: Number((paper || {}).year || 0),
      title: String((paper || {}).title || ''),
    })) : [defaultPaper()];
    state.steps = Array.isArray(state.steps) && state.steps.length ? state.steps.map((step, idx) => normalizeStep(step, idx + 1, state.domain_qid)) : [defaultStep(1)];
    state.edges = normalizeEdges(state.edges, state.steps.length);
    state.selected_step_index = Number.isInteger(state.selected_step_index) ? state.selected_step_index : 0;
    state._schema = FORM_SCHEMA;
    return state;
  }

  function normalizeStep(step, index, domainQid) {
    const base = defaultStep(index);
    const normalized = Object.assign({}, base, step || {});
    normalized.step_id = index;
    normalized.claim = String(normalized.claim || '');
    normalized.inference = String(normalized.inference || '');
    normalized.next_question = String(normalized.next_question || '');
    normalized.sources = Array.isArray(normalized.sources) && normalized.sources.length ? normalized.sources.map((src) => ({
      type: ['text', 'image', 'table'].includes(String((src || {}).type || '').trim()) ? String(src.type).trim() : 'text',
      source: String((src || {}).source || ''),
      page: String((src || {}).page || ''),
      locator: String((src || {}).locator || ''),
      snippet_or_summary: String((src || {}).snippet_or_summary || ''),
    })) : [defaultSource()];
    normalized.conditions = Object.assign({}, base.conditions, normalized.conditions || {});
    for (const key of requiredConditionKeys(domainQid)) {
      if (!(key in normalized.conditions)) normalized.conditions[key] = '';
    }
    return normalized;
  }

  function normalizeEdges(edges, stepCount) {
    const out = [];
    const seen = new Set();
    if (!Array.isArray(edges)) return out;
    for (const pair of edges) {
      if (!Array.isArray(pair) || pair.length < 2) continue;
      const from = Number(pair[0]);
      const to = Number(pair[1]);
      if (!Number.isFinite(from) || !Number.isFinite(to) || from === to) continue;
      if (from < 1 || to < 1 || from > stepCount || to > stepCount) continue;
      const key = `${from}->${to}`;
      if (seen.has(key)) continue;
      seen.add(key);
      out.push([from, to]);
    }
    return out;
  }

  let state = normalizeState(INITIAL_STATE);

  function syncTopFieldsToState() {
    state.expert.last_name = dom.expertLast.value;
    state.expert.first_name = dom.expertFirst.value;
    state.expert.patronymic = dom.expertPat.value;
    state.topic = dom.topicInput.value;
    state.cutoff_year = Number(dom.cutoffYearInput.value || 0);
    state.filename = dom.filenameInput.value;
    state.domain_query = dom.domainQuery.value;
    state.domain_language = dom.domainLanguage.value;
    state.domain_qid = String(dom.domainQid.value || DEFAULT_DOMAIN_QID || 'Q336').trim();
  }

  function syncStateToTopFields() {
    dom.expertLast.value = state.expert.last_name || '';
    dom.expertFirst.value = state.expert.first_name || '';
    dom.expertPat.value = state.expert.patronymic || '-';
    dom.topicInput.value = state.topic || '';
    dom.cutoffYearInput.value = String(state.cutoff_year || 2020);
    dom.filenameInput.value = state.filename || '';
    dom.domainQuery.value = state.domain_query || '';
    dom.domainLanguage.value = state.domain_language || 'ru';
    dom.domainQid.value = state.domain_qid || DEFAULT_DOMAIN_QID || 'Q336';
  }

  function renderPapers() {
    dom.papersContainer.innerHTML = '';
    state.papers.forEach((paper, index) => {
      const row = document.createElement('div');
      row.className = 'paper-row';
      row.innerHTML = `
        <div class="subsection-header">
          <div><strong>Публикация ${index + 1}</strong></div>
          <div class="mini-actions">
            <button class="danger" data-remove-paper="${index}">Удалить</button>
          </div>
        </div>
        <div class="grid cols-3">
          <label class="field"><span>id*</span><input type="text" data-paper-field="id" data-paper-index="${index}" value="${escapeAttr(paper.id)}" placeholder="doi:... / arxiv:... / openalex:... / url"></label>
          <label class="field"><span>year*</span><input type="number" min="0" step="1" data-paper-field="year" data-paper-index="${index}" value="${escapeAttr(paper.year || '')}"></label>
          <label class="field"><span>title*</span><input type="text" data-paper-field="title" data-paper-index="${index}" value="${escapeAttr(paper.title)}" placeholder="Название статьи"></label>
        </div>`;
      dom.papersContainer.appendChild(row);
    });
  }

  function renderSteps() {
    dom.stepsContainer.innerHTML = '';
    state.steps.forEach((step, index) => {
      const requiredKeys = requiredConditionKeys(state.domain_qid);
      const conditionKeys = ['system', 'environment', 'protocol', 'notes', ...requiredKeys.filter((key) => !['system', 'environment', 'protocol', 'notes'].includes(key))];
      const wrap = document.createElement('div');
      wrap.className = 'step-card';
      const sourcesHtml = step.sources.map((source, sourceIndex) => `
        <div class="source-card">
          <div class="subsection-header">
            <div><strong>Источник ${sourceIndex + 1}</strong></div>
            <div class="mini-actions"><button class="danger" data-remove-source="${index}:${sourceIndex}">Удалить источник</button></div>
          </div>
          <div class="grid cols-3">
            <label class="field"><span>type</span>
              <select data-source-field="type" data-step-index="${index}" data-source-index="${sourceIndex}">
                <option value="text" ${source.type === 'text' ? 'selected' : ''}>Text</option>
                <option value="image" ${source.type === 'image' ? 'selected' : ''}>Image/Figure</option>
                <option value="table" ${source.type === 'table' ? 'selected' : ''}>Table</option>
              </select>
            </label>
            <label class="field"><span>page</span><input type="text" data-source-field="page" data-step-index="${index}" data-source-index="${sourceIndex}" value="${escapeAttr(source.page)}" placeholder="например, 1"></label>
            <label class="field"><span>locator</span><input type="text" data-source-field="locator" data-step-index="${index}" data-source-index="${sourceIndex}" value="${escapeAttr(source.locator)}" placeholder="Figure 3 / Table 2"></label>
          </div>
          <label class="field" style="margin-top:10px;"><span>source*</span><input type="text" data-source-field="source" data-step-index="${index}" data-source-index="${sourceIndex}" value="${escapeAttr(source.source)}" placeholder="doi:... / arxiv:... / openalex:... / url"></label>
          <label class="field" style="margin-top:10px;"><span>snippet*</span><textarea data-source-field="snippet_or_summary" data-step-index="${index}" data-source-index="${sourceIndex}" placeholder="Цитата/выжимка/описание">${escapeHtml(source.snippet_or_summary)}</textarea></label>
        </div>`).join('');
      const conditionsHtml = conditionKeys.map((key) => `
        <label class="field"><span>${escapeHtml(key)}${['system','environment','protocol'].includes(key) ? '*' : ''}</span><input type="text" data-condition-key="${escapeAttr(key)}" data-step-index="${index}" value="${escapeAttr((step.conditions || {})[key] || '')}" placeholder="${key === 'notes' ? '' : 'unknown'}"></label>`).join('');
      wrap.innerHTML = `
        <div class="step-header">
          <div>
            <div class="step-title">Шаг ${index + 1}</div>
            <div class="step-claim-preview">${escapeHtml(cleanTextPreview(step.claim || 'Утверждение пока не заполнено', 90))}</div>
          </div>
          <div class="mini-actions">
            <button class="ghost" data-add-source="${index}">Добавить источник</button>
            <button class="danger" data-remove-step="${index}">Удалить шаг</button>
          </div>
        </div>
        <label class="field"><span>claim*</span><textarea data-step-field="claim" data-step-index="${index}" placeholder="Утверждение шага">${escapeHtml(step.claim)}</textarea></label>
        <div style="margin-top:12px;">
          <div class="subsection-header"><div><strong>Sources</strong></div></div>
          ${sourcesHtml}
        </div>
        <div style="margin-top:12px;">
          <div class="subsection-header"><div><strong>Conditions</strong></div><div class="hint">Если в статье не указано — пишите <code>unknown</code>.</div></div>
          <div class="grid cols-2">${conditionsHtml}</div>
        </div>
        <div class="grid cols-2" style="margin-top:12px;">
          <label class="field"><span>inference*</span><textarea data-step-field="inference" data-step-index="${index}" placeholder="Логический вывод">${escapeHtml(step.inference)}</textarea></label>
          <label class="field"><span>next_question*</span><textarea data-step-field="next_question" data-step-index="${index}" placeholder="Вопрос для следующего шага">${escapeHtml(step.next_question)}</textarea></label>
        </div>`;
      dom.stepsContainer.appendChild(wrap);
    });
  }

  function renderEdgesMatrix() {
    const rows = [];
    const stepCount = state.steps.length;
    if (!stepCount) {
      dom.edgesMatrixWrap.innerHTML = '<div class="hint">Сначала добавьте хотя бы один шаг.</div>';
      dom.edgeCountPill.textContent = '0 edges';
      return;
    }
    const headers = state.steps.map((step, index) => `<th>${index + 1}: ${escapeHtml(cleanTextPreview(step.claim || '', 32) || 'шаг')}</th>`).join('');
    for (let from = 1; from <= stepCount; from += 1) {
      const cells = [];
      for (let to = 1; to <= stepCount; to += 1) {
        if (from === to) {
          cells.push('<td class="center">—</td>');
          continue;
        }
        const checked = state.edges.some((pair) => pair[0] === from && pair[1] === to);
        cells.push(`<td class="center"><input class="edge-check" type="checkbox" data-edge-from="${from}" data-edge-to="${to}" ${checked ? 'checked' : ''}></td>`);
      }
      rows.push(`<tr><th>${from}: ${escapeHtml(cleanTextPreview(state.steps[from - 1].claim || '', 38) || 'шаг')}</th>${cells.join('')}</tr>`);
    }
    dom.edgesMatrixWrap.innerHTML = `<table class="edge-table"><thead><tr><th>from \\ to</th>${headers}</tr></thead><tbody>${rows.join('')}</tbody></table>`;
    dom.edgeCountPill.textContent = `${state.edges.length} edges`;
  }

  function refreshMeta() {
    const expertSlug = slugify([state.expert.last_name, state.expert.first_name, state.expert.patronymic].filter(Boolean).join(' '));
    dom.submissionPreview.value = `${expertSlug}__pending_hash`;
    const required = requiredConditionKeys(state.domain_qid);
    dom.requiredConditions.value = required.length ? required.join(', ') : 'только базовые поля system/environment/protocol/notes';
    const cfg = domainConfigFor(state.domain_qid);
    dom.domainBadge.textContent = cfg ? `${cfg.title || cfg.domain_id || state.domain_qid}` : `Custom domain ${state.domain_qid || DEFAULT_DOMAIN_QID}`;
  }

  function renderAll(save = false) {
    syncStateToTopFields();
    renderPapers();
    renderSteps();
    renderEdgesMatrix();
    refreshMeta();
    if (save) saveAutosave('render');
  }

  function setMessage(kind, html) {
    if (!html) {
      dom.globalMessage.innerHTML = '';
      return;
    }
    dom.globalMessage.innerHTML = `<div class="callout ${kind}" style="margin-bottom:18px;">${html}</div>`;
  }

  function setAutosaveStatus(text, kind = 'info') {
    dom.autosaveStatus.textContent = text;
    dom.autosaveStatus.style.color = kind === 'error' ? '#b91c1c' : kind === 'success' ? '#166534' : '#475569';
  }

  function saveAutosave(reason = 'autosave') {
    try {
      syncTopFieldsToState();
      const payload = deepClone(state);
      payload._schema = FORM_SCHEMA;
      payload._saved_at = nowUtc();
      payload._saved_reason = reason;
      localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
      setAutosaveStatus(`Черновик автосохранён (${payload._saved_at})`, 'success');
    } catch (err) {
      setAutosaveStatus(`Не удалось сохранить черновик: ${err}`, 'error');
    }
  }

  function restoreAutosave(showMessage = true) {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) {
        if (showMessage) setMessage('warning', 'Автосохранённый черновик пока не найден.');
        return false;
      }
      state = normalizeState(JSON.parse(raw));
      renderAll(false);
      setAutosaveStatus(`Черновик восстановлен (${state._saved_at || 'без даты'})`, 'success');
      if (showMessage) setMessage('success', 'Форма восстановлена из локального автосохранения браузера.');
      return true;
    } catch (err) {
      setMessage('error', `Не удалось восстановить черновик: ${escapeHtml(String(err))}`);
      return false;
    }
  }

  function addPaper() {
    state.papers.push(defaultPaper());
    renderAll(true);
  }

  function removePaper(index) {
    state.papers.splice(index, 1);
    if (!state.papers.length) state.papers.push(defaultPaper());
    renderAll(true);
  }

  function addStep() {
    state.steps.push(defaultStep(state.steps.length + 1));
    state.steps = state.steps.map((step, idx) => normalizeStep(step, idx + 1, state.domain_qid));
    state.edges = normalizeEdges(state.edges, state.steps.length);
    renderAll(true);
  }

  function removeStep(index) {
    state.steps.splice(index, 1);
    if (!state.steps.length) state.steps.push(defaultStep(1));
    state.steps = state.steps.map((step, idx) => normalizeStep(step, idx + 1, state.domain_qid));
    state.edges = normalizeEdges(state.edges.filter((pair) => pair[0] !== index + 1 && pair[1] !== index + 1).map((pair) => [pair[0] > index + 1 ? pair[0] - 1 : pair[0], pair[1] > index + 1 ? pair[1] - 1 : pair[1]]), state.steps.length);
    renderAll(true);
  }

  function addSource(stepIndex) {
    state.steps[stepIndex].sources.push(defaultSource());
    renderAll(true);
  }

  function removeSource(stepIndex, sourceIndex) {
    state.steps[stepIndex].sources.splice(sourceIndex, 1);
    if (!state.steps[stepIndex].sources.length) state.steps[stepIndex].sources.push(defaultSource());
    renderAll(true);
  }

  async function computeSubmissionMeta(doc) {
    const expertSlug = slugify([doc.expert.last_name, doc.expert.first_name, doc.expert.patronymic].filter(Boolean).join(' '));
    const canonical = JSON.stringify(doc);
    const bytes = new TextEncoder().encode(canonical);
    const hashBuffer = await crypto.subtle.digest('SHA-256', bytes);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const fullHash = hashArray.map((b) => b.toString(16).padStart(2, '0')).join('');
    const shortHash = fullHash.slice(0, 12);
    return {
      artifact_hash: shortHash,
      submission_id: `${expertSlug || 'expert'}__${shortHash}`,
      latin_slug: expertSlug || 'expert',
      latin_full_name: transliterate([doc.expert.last_name, doc.expert.first_name, doc.expert.patronymic].filter(Boolean).join(' ')).trim(),
    };
  }

  function currentDocBase() {
    syncTopFieldsToState();
    state.steps = state.steps.map((step, idx) => normalizeStep(step, idx + 1, state.domain_qid));
    state.edges = normalizeEdges(state.edges, state.steps.length);
    return {
      artifact_version: 2,
      topic: String(state.topic || '').trim(),
      domain: String(state.domain_qid || DEFAULT_DOMAIN_QID || 'Q336').trim(),
      cutoff_year: Number(state.cutoff_year || 0),
      papers: state.papers.map((paper) => ({ id: String(paper.id || '').trim(), year: Number(paper.year || 0), title: String(paper.title || '').trim() })),
      steps: state.steps.map((step, idx) => ({
        step_id: idx + 1,
        claim: String(step.claim || '').trim(),
        sources: step.sources.map((src) => ({
          type: String(src.type || 'text').trim() || 'text',
          source: String(src.source || '').trim(),
          snippet_or_summary: String(src.snippet_or_summary || '').trim(),
          ...(String(src.page || '').trim() ? { page: String(src.page || '').trim() } : {}),
          ...(String(src.locator || '').trim() ? { locator: String(src.locator || '').trim() } : {}),
        })),
        conditions: Object.fromEntries(Object.entries(step.conditions || {}).map(([key, value]) => [String(key), String(value || '')])),
        inference: String(step.inference || '').trim(),
        next_question: String(step.next_question || '').trim(),
      })),
      edges: normalizeEdges(state.edges, state.steps.length),
      generated_at: nowUtc(),
      expert: {
        last_name: String(state.expert.last_name || '').trim(),
        first_name: String(state.expert.first_name || '').trim(),
        patronymic: String(state.expert.patronymic || '').trim(),
      },
    };
  }

  function validateDoc(doc) {
    const errors = [];
    if (!doc.expert.last_name || !doc.expert.first_name || !doc.expert.patronymic) errors.push('Заполните ФИО эксперта.');
    if (!doc.topic) errors.push('Заполните поле Topic.');
    if (!doc.domain) errors.push('Укажите Domain (QID).');
    if (!Number.isFinite(doc.cutoff_year) || doc.cutoff_year <= 0) errors.push('Cutoff year должен быть положительным числом.');
    const validPapers = doc.papers.filter((paper) => paper.id || paper.title || paper.year);
    if (!validPapers.length) errors.push('Добавьте хотя бы одну публикацию.');
    validPapers.forEach((paper, index) => {
      if (!paper.id || !paper.title || !paper.year) {
        errors.push(`papers[${index + 1}]: заполните id, year и title.`);
      }
    });
    doc.steps.forEach((step, index) => {
      if (!step.claim) errors.push(`steps[${index + 1}]: заполните claim.`);
      if (!step.inference) errors.push(`steps[${index + 1}]: заполните inference.`);
      if (!step.next_question) errors.push(`steps[${index + 1}]: заполните next_question.`);
      if (!Array.isArray(step.sources) || !step.sources.length) {
        errors.push(`steps[${index + 1}]: добавьте хотя бы один источник.`);
      }
      (step.sources || []).forEach((src, srcIndex) => {
        if (!src.source || !src.snippet_or_summary) errors.push(`steps[${index + 1}].sources[${srcIndex + 1}]: заполните source и snippet_or_summary.`);
      });
      ['system', 'environment', 'protocol'].forEach((key) => {
        if (!(key in (step.conditions || {}))) errors.push(`steps[${index + 1}].conditions.${key}: поле отсутствует.`);
      });
    });
    return errors;
  }

  function yamlScalar(value) {
    if (value === null || typeof value === 'undefined') return '""';
    if (typeof value === 'number') return Number.isFinite(value) ? String(value) : '0';
    if (typeof value === 'boolean') return value ? 'true' : 'false';
    const text = String(value);
    if (text === '') return '""';
    if (/^[A-Za-z0-9_:\/.+\-]+$/.test(text) && !['true', 'false', 'null'].includes(text.toLowerCase())) return text;
    return JSON.stringify(text);
  }

  function toYaml(value, indent = 0) {
    const pad = '  '.repeat(indent);
    if (Array.isArray(value)) {
      if (!value.length) return '[]';
      return value.map((item) => {
        if (item && typeof item === 'object') {
          const nested = toYaml(item, indent + 1);
          const lines = String(nested).split('\n');
          return `${pad}- ${lines[0]}\n${lines.slice(1).map((line) => `${pad}  ${line}`).join('\n')}`;
        }
        return `${pad}- ${yamlScalar(item)}`;
      }).join('\n');
    }
    if (value && typeof value === 'object') {
      const entries = Object.entries(value);
      if (!entries.length) return '{}';
      return entries.map(([key, item]) => {
        if (item && typeof item === 'object') {
          const nested = toYaml(item, indent + 1);
          if (Array.isArray(item)) return `${pad}${key}:\n${nested}`;
          return `${pad}${key}:\n${nested}`;
        }
        return `${pad}${key}: ${yamlScalar(item)}`;
      }).join('\n');
    }
    return `${pad}${yamlScalar(value)}`;
  }

  function downloadBlob(filename, blob) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  async function exportYaml() {
    const doc = currentDocBase();
    const errors = validateDoc(doc);
    if (errors.length) {
      setMessage('error', `<strong>Форма заполнена не полностью.</strong><br>${errors.map((item) => escapeHtml(item)).join('<br>')}`);
      return;
    }
    const meta = await computeSubmissionMeta(doc);
    doc.expert.full_name = [doc.expert.last_name, doc.expert.first_name, doc.expert.patronymic].filter(Boolean).join(' ').trim();
    doc.expert.latin_full_name = meta.latin_full_name;
    doc.expert.latin_slug = meta.latin_slug;
    doc.artifact_hash = meta.artifact_hash;
    doc.submission_id = meta.submission_id;
    dom.submissionPreview.value = doc.submission_id;
    const yamlText = toYaml(doc) + '\n';
    const filename = (String(state.filename || '').trim() || `${doc.submission_id}.yaml`).replace(/\s+/g, '_');
    downloadBlob(filename.toLowerCase().endsWith('.yaml') || filename.toLowerCase().endsWith('.yml') ? filename : `${filename}.yaml`, new Blob([yamlText], { type: 'text/yaml;charset=utf-8' }));
    saveAutosave('export_yaml');
    setMessage('success', `YAML собран и скачан: <code>${escapeHtml(filename)}</code>`);
  }

  function saveDraftJson() {
    syncTopFieldsToState();
    const payload = deepClone(state);
    payload._schema = FORM_SCHEMA;
    payload._saved_at = nowUtc();
    payload._saved_reason = 'manual_json_export';
    const topicSlug = slugify(payload.topic || 'trajectory_form');
    downloadBlob(`${topicSlug || 'trajectory_form'}__draft.json`, new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json;charset=utf-8' }));
    saveAutosave('manual_json_export');
    setMessage('success', 'JSON-черновик скачан на устройство.');
  }

  async function loadDraftFromFile(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const parsed = JSON.parse(String(reader.result || '{}'));
        state = normalizeState(parsed);
        renderAll(false);
        saveAutosave('import_json');
        setMessage('success', `Черновик <code>${escapeHtml(file.name)}</code> загружен в форму.`);
      } catch (err) {
        setMessage('error', `Не удалось прочитать JSON-черновик: ${escapeHtml(String(err))}`);
      } finally {
        dom.loadDraftInput.value = '';
      }
    };
    reader.readAsText(file);
  }

  async function searchWikidata() {
    const query = String(dom.domainQuery.value || '').trim();
    const language = String(dom.domainLanguage.value || 'ru').trim() || 'ru';
    if (!query) {
      setMessage('warning', 'Введите текст для поиска в Wikidata.');
      return;
    }
    dom.wikidataStatus.innerHTML = 'Ищу…';
    dom.wikidataResults.innerHTML = '';
    try {
      const params = new URLSearchParams({
        action: 'wbsearchentities',
        search: query,
        language,
        format: 'json',
        limit: '10',
        origin: '*',
      });
      const resp = await fetch(`https://www.wikidata.org/w/api.php?${params.toString()}`, { headers: { 'Accept': 'application/json' } });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const payload = await resp.json();
      const results = Array.isArray(payload.search) ? payload.search : [];
      if (!results.length) {
        dom.wikidataStatus.innerHTML = '<span class="muted">Ничего не найдено.</span>';
        return;
      }
      dom.wikidataStatus.innerHTML = `<span class="muted">Найдено: ${results.length}</span>`;
      results.forEach((item) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'result-option';
        const qid = String(item.id || '').trim();
        button.innerHTML = `<strong>${escapeHtml(qid)} — ${escapeHtml(String(item.label || ''))}</strong><span class="muted">${escapeHtml(String(item.description || ''))}</span>`;
        button.addEventListener('click', () => {
          dom.domainQid.value = qid;
          state.domain_qid = qid;
          saveAutosave('wikidata_pick');
          renderAll(false);
          setMessage('success', `Выбран домен <code>${escapeHtml(qid)}</code>.`);
        });
        dom.wikidataResults.appendChild(button);
      });
    } catch (err) {
      dom.wikidataStatus.innerHTML = `<span style="color:#b91c1c">Ошибка поиска: ${escapeHtml(String(err))}</span>`;
    }
  }

  function escapeHtml(value) {
    return String(value || '').replace(/[&<>"']/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
  }

  function escapeAttr(value) {
    return escapeHtml(value);
  }

  document.addEventListener('input', (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    if (target.matches('[data-paper-field]')) {
      const index = Number(target.getAttribute('data-paper-index'));
      const field = target.getAttribute('data-paper-field');
      if (!Number.isFinite(index) || !field) return;
      state.papers[index][field] = field === 'year' ? Number(target.value || 0) : target.value;
      saveAutosave(`paper:${field}`);
      return;
    }
    if (target.matches('[data-step-field]')) {
      const index = Number(target.getAttribute('data-step-index'));
      const field = target.getAttribute('data-step-field');
      if (!Number.isFinite(index) || !field) return;
      state.steps[index][field] = target.value;
      renderEdgesMatrix();
      saveAutosave(`step:${field}`);
      return;
    }
    if (target.matches('[data-source-field]')) {
      const stepIndex = Number(target.getAttribute('data-step-index'));
      const sourceIndex = Number(target.getAttribute('data-source-index'));
      const field = target.getAttribute('data-source-field');
      if (!Number.isFinite(stepIndex) || !Number.isFinite(sourceIndex) || !field) return;
      state.steps[stepIndex].sources[sourceIndex][field] = target.value;
      saveAutosave(`source:${field}`);
      return;
    }
    if (target.matches('[data-condition-key]')) {
      const stepIndex = Number(target.getAttribute('data-step-index'));
      const key = target.getAttribute('data-condition-key');
      if (!Number.isFinite(stepIndex) || !key) return;
      state.steps[stepIndex].conditions[key] = target.value;
      saveAutosave(`condition:${key}`);
      return;
    }
    if ([dom.expertLast, dom.expertFirst, dom.expertPat, dom.topicInput, dom.cutoffYearInput, dom.filenameInput, dom.domainQuery, dom.domainLanguage, dom.domainQid].includes(target)) {
      syncTopFieldsToState();
      state.steps = state.steps.map((step, idx) => normalizeStep(step, idx + 1, state.domain_qid));
      refreshMeta();
      if (target === dom.domainQid) {
        renderSteps();
      }
      saveAutosave(`field:${target.id}`);
    }
  });

  document.addEventListener('change', (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    if (target.matches('.edge-check')) {
      const from = Number(target.getAttribute('data-edge-from'));
      const to = Number(target.getAttribute('data-edge-to'));
      const key = `${from}->${to}`;
      const exists = state.edges.some((pair) => pair[0] === from && pair[1] === to);
      if (target.checked && !exists) {
        state.edges.push([from, to]);
      } else if (!target.checked && exists) {
        state.edges = state.edges.filter((pair) => !(pair[0] === from && pair[1] === to));
      }
      state.edges = normalizeEdges(state.edges, state.steps.length);
      renderEdgesMatrix();
      saveAutosave(`edge:${key}`);
      return;
    }
    if (target === dom.loadDraftInput && target.files && target.files[0]) {
      loadDraftFromFile(target.files[0]);
    }
  });

  document.addEventListener('click', (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    const removePaperIndex = target.getAttribute('data-remove-paper');
    if (removePaperIndex !== null) {
      removePaper(Number(removePaperIndex));
      return;
    }
    const removeStepIndex = target.getAttribute('data-remove-step');
    if (removeStepIndex !== null) {
      removeStep(Number(removeStepIndex));
      return;
    }
    const addSourceIndex = target.getAttribute('data-add-source');
    if (addSourceIndex !== null) {
      addSource(Number(addSourceIndex));
      return;
    }
    const removeSourceRef = target.getAttribute('data-remove-source');
    if (removeSourceRef !== null) {
      const [stepIndex, sourceIndex] = removeSourceRef.split(':').map((part) => Number(part));
      removeSource(stepIndex, sourceIndex);
    }
  });

  dom.addPaperBtn.addEventListener('click', addPaper);
  dom.addStepBtn.addEventListener('click', addStep);
  dom.exportYamlBtn.addEventListener('click', exportYaml);
  dom.saveDraftBtn.addEventListener('click', saveDraftJson);
  dom.restoreBtn.addEventListener('click', () => restoreAutosave(true));
  dom.clearDraftBtn.addEventListener('click', () => {
    localStorage.removeItem(STORAGE_KEY);
    setAutosaveStatus('Автосохранение удалено', 'info');
    setMessage('warning', 'Локальный черновик удалён из браузера. Текущие поля на странице не очищены.');
  });
  dom.domainSearchBtn.addEventListener('click', searchWikidata);

  renderAll(false);
  restoreAutosave(false);
  setAutosaveStatus('Форма готова. Автосохранение включено.', 'info');
  </script>
</body>
</html>
'''


def build_task1_offline_form(output_path: str | Path, initial_state: dict[str, Any] | None = None, domain_configs: list[dict[str, Any]] | None = None) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    state = initial_state or _default_state()
    configs = domain_configs or [
        {
            "domain_id": "science",
            "title": "General scientific literature",
            "wikidata_qid": "Q336",
            "required_conditions": [],
            "source_path": "embedded",
        }
    ]

    html_text = (
        _HTML_TEMPLATE
        .replace('__PAGE_TITLE__', 'Task 1 offline form')
        .replace('__INITIAL_STATE__', json.dumps(state, ensure_ascii=False))
        .replace('__DOMAIN_CONFIGS__', json.dumps(configs, ensure_ascii=False))
        .replace('__STORAGE_KEY__', 'task1-offline-form-v1')
    )
    output.write_text(html_text, encoding='utf-8')
    return output


if __name__ == '__main__':
    build_task1_offline_form(Path('task1_offline_form.html'))
