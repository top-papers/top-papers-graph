# -*- coding: utf-8 -*-
"""
Билдер диагностического набора Task 3 (A/B). Утилита: читает config.yaml
(списки papers и cases) и собирает из него:
  - task3_ab_case_manifest.filled.json (корень) — итоговый манифест;
  - web_form/task3_ab_creator_offline_form_ru.html — GUI-форму, предзагруженную манифестом
    (с серверным мостом для web_form/serve.py; без авто-стартовых кейсов);
  - validation/VALIDATION_LOG.txt — отчёт валидации по логике cell 8 блокнота.

Метаданные эксперимента (topic, creator_id, cutoff_year, review_goal) здесь НЕ
заполняются — они задаются в GUI (форма/бот). По умолчанию config пуст → пустой набор.

Зависимости: PyYAML (см. scripts/requirements.txt). Сеть не используется.
Запуск из корня репозитория:  python3 scripts/build.py
"""
import json, os, re

try:
    import yaml
except ImportError:
    raise SystemExit('Нужен PyYAML: pip install -r scripts/requirements.txt')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(BASE, 'config.yaml')
WEBFORM = os.path.join(BASE, 'web_form')
VALID = os.path.join(BASE, 'validation')
NB = os.path.join(BASE, 'notebook', 'expert_task3_yaml_bundle_for_ab_test.ipynb')
MANIFEST = os.path.join(BASE, 'task3_ab_case_manifest.filled.json')
os.makedirs(WEBFORM, exist_ok=True)
os.makedirs(VALID, exist_ok=True)

# ───────────────────────── схема / enum (как в cell 8) ─────────────────────────
MATCH_KEYS = ['candidate_signature', 'candidate_source_contains', 'candidate_predicate_contains',
              'candidate_target_contains', 'hypothesis_title_contains', 'premise_contains',
              'mechanism_contains', 'time_scope_contains', 'rank_hint']
EMPTY_MATCH = {k: '' for k in MATCH_KEYS}
VF = {'evidence', 'visual_fact', 'temporal', 'overall'}
VR = {'missed_visual_fact', 'wrong_evidence_linkage', 'needs_time_fix', 'hallucinated_visual_inference'}
VS = {'multimodal_hard', 'temporal_hard', 'easy_control'}
VE = {'figure', 'table', 'figure_or_table', 'formula', 'page', 'mixed'}


def aslist(v):
    if v is None:
        return []
    return list(v) if isinstance(v, (list, tuple)) else [v]


# ───────────────────────── чтение конфига ─────────────────────────
if not os.path.exists(CONFIG):
    raise SystemExit('Нет config.yaml в корне репозитория — создайте его (см. шаблон в README).')
cfg = yaml.safe_load(open(CONFIG, encoding='utf-8')) or {}
papers = {p['id']: p for p in (cfg.get('papers') or []) if isinstance(p, dict) and p.get('id')}
spec_cases = cfg.get('cases') or []

# ───────────────────────── сборка кейсов ─────────────────────────
cases = []
for i, s in enumerate(spec_cases, 1):
    s = s or {}
    pid = s.get('paper_id', '') or ''
    paper = papers.get(pid, {})
    stratum = s.get('stratum', 'multimodal_hard')
    match = dict(EMPTY_MATCH)
    match.update(s.get('match') or {})
    cases.append({
        'case_id': s.get('case_id') or 'case_%02d' % i,
        'enabled': bool(s.get('enabled', True)),
        'primary_endpoint': bool(s.get('primary_endpoint', stratum != 'easy_control')),
        'stratum': stratum,
        'paper_title': s.get('paper_title', paper.get('title', '')) or '',
        'paper_id': pid,
        'year': s.get('year', paper.get('year', '')) or '',
        'evidence_kind': s.get('evidence_kind', 'figure_or_table'),
        'page_hint': s.get('page_hint', '') or '',
        'creator_prompt': s.get('creator_prompt', '') or '',
        'creator_rationale': s.get('creator_rationale', '') or '',
        'review_focus': aslist(s.get('review_focus')),
        'expected_error_modes': aslist(s.get('expected_error_modes')),
        'match': match,
        'notes': s.get('notes', '') or '',
    })

manifest = {
    'schema_version': 'task3-ab-creator-v1',
    'experiment_meta': {'topic': '', 'submission_id': '', 'creator_id': '',
                        'cutoff_year': '', 'review_goal': ''},
    'cases': cases,
}
json.dump(manifest, open(MANIFEST, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

# ───────────────────────── валидация (логика cell 8) ─────────────────────────
def reqmiss(c):
    z = [f for f in ['paper_title', 'creator_prompt', 'creator_rationale'] if not str(c.get(f, '')).strip()]
    if not isinstance(c.get('review_focus'), list) or not c['review_focus']:
        z.append('review_focus')
    if not any(str((c.get('match') or {}).get(k, '')).strip() for k in MATCH_KEYS):
        z.append('match.*')
    return z

from collections import Counter
log = []
allok = True
ids = set()
for c in cases:
    pr = []
    mm = reqmiss(c)
    if mm:
        pr.append('required:' + ','.join(mm))
    if c['case_id'] in ids:
        pr.append('dup_id')
    ids.add(c['case_id'])
    if c['evidence_kind'] not in VE:
        pr.append('bad_evidence_kind')
    if c['stratum'] not in VS:
        pr.append('bad_stratum')
    if any(x not in VF for x in c['review_focus']):
        pr.append('bad_review_focus')
    if any(x not in VR for x in c['expected_error_modes']):
        pr.append('bad_error_mode')
    if len(c['review_focus']) > 3:
        pr.append('review_focus>3')
    if len(c['expected_error_modes']) > 3:
        pr.append('error_modes>3')
    wc = len(c['creator_prompt'].split())
    if wc > 40:
        pr.append('prompt>40w')
    if pr:
        allok = False
    log.append('%s %-15s %-9s %-6s w=%2d %s' % (
        c['case_id'], c['stratum'], c['page_hint'] or '-',
        'READY' if not pr else 'NOT', wc, ('PROB:' + ';'.join(pr)) if pr else 'ok'))

n = len(cases)
cnt = Counter(c['stratum'] for c in cases)
dist = ['%-15s %d/%d=%d%%' % (k, cnt[k], n, round(100 * cnt[k] / n)) for k in sorted(cnt)] if n else ['(кейсов нет)']
open(os.path.join(VALID, 'VALIDATION_LOG.txt'), 'w', encoding='utf-8').write(
    'total=%d unique_ids=%d all_ready=%s\n\n' % (n, len(ids), allok) +
    ('\n'.join(log) + '\n' if log else '(кейсов нет)\n') +
    '\n--- распределение страт ---\n' + '\n'.join(dist) + '\n')

# ───────────────────────── веб-форма из шаблона cell 8 ─────────────────────────
# Серверный мост для web_form/serve.py: кнопка «Сохранить на сервер» (POST /api/manifest)
# + подгрузка манифеста с диска (GET /api/manifest). В офлайне fetch падает в catch.
BRIDGE = r'''
/* --- server bridge (added by scripts/build.py for web_form/serve.py) --- */
(function(){
  function onServer(){ return location.protocol === 'http:' || location.protocol === 'https:'; }
  try {
    var bar = document.querySelector('.toolbar');
    if (bar) {
      var btn = document.createElement('button');
      btn.id = 'saveServer'; btn.className = 'primary';
      btn.textContent = 'Сохранить на сервер';
      btn.onclick = function(){
        fetch('/api/manifest', {method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify(state, null, 2)})
          .then(function(r){ return r.json().then(function(j){ return {ok:r.ok, j:j}; }); })
          .then(function(res){
            if (res.ok && res.j && res.j.ok) alert('Сохранено в репозиторий: ' + res.j.cases + ' кейсов.');
            else alert('Ошибка сохранения: ' + ((res.j && res.j.error) || JSON.stringify(res.j)));
          })
          .catch(function(e){ alert('Сервер недоступен (офлайн-режим). Используйте «Скачать заполненный JSON». ' + e); });
      };
      bar.appendChild(btn);
    }
  } catch(e){}
  if (onServer()) {
    fetch('/api/manifest').then(function(r){ return r.ok ? r.json() : null; })
      .then(function(m){
        if (m && Array.isArray(m.cases) && m.cases.length) { Object.assign(state, m); render(); save(); }
      })
      .catch(function(){});
  }
})();
'''

cell8 = ''.join(json.load(open(NB, encoding='utf-8'))['cells'][8]['source'])
mk = "html_template = r'''"
i = cell8.index(mk) + len(mk)
j = cell8.index("'''", i)
tmpl = cell8[i:j]
app = json.dumps(manifest, ensure_ascii=False).replace('</', '<\\/')
html_out = tmpl.replace('__APP_DATA__', app)

# вырезаем авто-добавление 3 стартовых кейсов — набор берётся только из config/манифеста
html_out, k_auto = re.subn(
    r"if \(!\(state\.cases \|\| \[\]\)\.length\) \{\s*addCase\('multimodal_hard'\);\s*"
    r"addCase\('temporal_hard'\);\s*addCase\('easy_control'\);\s*\}",
    '/* стартовые кейсы не создаём — берутся из config/манифеста */',
    html_out)
if k_auto == 0:
    raise SystemExit('не найден блок авто-создания кейсов в шаблоне формы — форма не будет пустой')

# серверный мост перед последним </script>
k = html_out.rfind('</script>')
if k == -1:
    raise SystemExit('не найден </script> в шаблоне формы — серверный мост не вставлен')
html_out = html_out[:k] + BRIDGE + html_out[k:]
open(os.path.join(WEBFORM, 'task3_ab_creator_offline_form_ru.html'), 'w', encoding='utf-8').write(html_out)

print('build OK: cases=%d all_ready=%s | %s' % (n, allok, ', '.join(dist)))
