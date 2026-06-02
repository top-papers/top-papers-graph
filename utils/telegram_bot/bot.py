# -*- coding: utf-8 -*-
"""
Telegram-бот создателя диагностического набора Task 3 (A/B, top-papers-graph).

Назначение: то же, что и офлайн веб-форма task3_ab_creator_offline_form_ru.html —
дать эксперту собрать/отредактировать кейсы и выгрузить task3_ab_case_manifest.filled.json.
Схема и правило готовности повторяют cell 8 блокнота:
кейс «Готов к экспорту», когда непусты paper_title, creator_prompt, creator_rationale,
review_focus (>=1) и хотя бы одно поле match.*

Зависимости: python-telegram-bot>=20 (см. requirements.txt).
Токен бота: из telegram_bot/.env (TELEGRAM_BOT_TOKEN=...; см. .env.example).
"""
import json, os, copy, logging
from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, ContextTypes

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
HERE = os.path.dirname(os.path.abspath(__file__))

# ---------- настройки ----------
# Токен бота — единственный секрет — берётся из .env (TELEGRAM_BOT_TOKEN) либо из
# переменной окружения. Никаких конфигов для данных: всё остальное задаётся в чате.
GOOGLE_FORM_URL = 'https://forms.gle/h5RwEA8DsZh9pBAt8'
DATA_DIR = os.path.join(HERE, 'tg_data')
SEED = os.path.join(HERE, '..', 'task3_ab_case_manifest.filled.json')
os.makedirs(DATA_DIR, exist_ok=True)

def env_token():
    """Читает TELEGRAM_BOT_TOKEN из telegram_bot/.env, иначе из переменной окружения."""
    p = os.path.join(HERE, '.env')
    if os.path.exists(p):
        for line in open(p, encoding='utf-8'):
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            if k.strip() == 'TELEGRAM_BOT_TOKEN':
                return v.strip().strip('"').strip("'")
    return os.environ.get('TELEGRAM_BOT_TOKEN', '').strip()

# ---------- схема / enum ----------
EVIDENCE = {'figure', 'table', 'figure_or_table', 'formula', 'page', 'mixed'}
STRATA = {'multimodal_hard', 'temporal_hard', 'easy_control'}
FOCUS = {'evidence', 'visual_fact', 'temporal', 'overall'}
ERRORS = {'missed_visual_fact', 'wrong_evidence_linkage', 'needs_time_fix', 'hallucinated_visual_inference'}
MATCH_KEYS = ['candidate_signature', 'candidate_source_contains', 'candidate_predicate_contains',
              'candidate_target_contains', 'hypothesis_title_contains', 'premise_contains',
              'mechanism_contains', 'time_scope_contains', 'rank_hint']
SCALAR_FIELDS = ['paper_title', 'paper_id', 'year', 'evidence_kind', 'page_hint',
                 'creator_prompt', 'creator_rationale', 'stratum', 'notes']

EMPTY_MATCH = {k: '' for k in MATCH_KEYS}

def empty_manifest():
    return {
        'schema_version': 'task3-ab-creator-v1',
        'experiment_meta': {
            'topic': '',
            'submission_id': '',
            'creator_id': '',
            'cutoff_year': '',
            'review_goal': '',
        },
        'cases': [],
    }

def new_case(stratum='multimodal_hard', idx=1):
    preset_focus = {'multimodal_hard': ['evidence', 'visual_fact'],
                    'temporal_hard': ['temporal', 'evidence'],
                    'easy_control': ['overall']}[stratum]
    preset_err = {'multimodal_hard': ['missed_visual_fact', 'wrong_evidence_linkage'],
                  'temporal_hard': ['needs_time_fix', 'wrong_evidence_linkage'],
                  'easy_control': []}[stratum]
    return {
        'case_id': 'case_%02d' % idx, 'enabled': True,
        'primary_endpoint': stratum != 'easy_control', 'stratum': stratum,
        'paper_title': '', 'paper_id': '', 'year': '', 'evidence_kind': 'figure_or_table',
        'page_hint': '', 'creator_prompt': '', 'creator_rationale': '',
        'review_focus': preset_focus, 'expected_error_modes': preset_err,
        'match': dict(EMPTY_MATCH), 'notes': '',
    }

# ---------- хранение состояния по чату ----------
def state_path(chat_id):
    return os.path.join(DATA_DIR, '%s.json' % chat_id)

def load_state(chat_id):
    p = state_path(chat_id)
    if os.path.exists(p):
        return json.load(open(p, encoding='utf-8'))
    if os.path.exists(SEED):
        return json.load(open(SEED, encoding='utf-8'))
    return empty_manifest()

def save_state(chat_id, st):
    json.dump(st, open(state_path(chat_id), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

def find_case(st, cid):
    for c in st['cases']:
        if c.get('case_id') == cid:
            return c
    return None

# ---------- готовность (повтор requiredMissing из cell 8) ----------
def required_missing(c):
    miss = []
    if not str(c.get('paper_title', '')).strip(): miss.append('paper_title')
    if not str(c.get('creator_prompt', '')).strip(): miss.append('creator_prompt')
    if not str(c.get('creator_rationale', '')).strip(): miss.append('creator_rationale')
    rf = c.get('review_focus')
    if not isinstance(rf, list) or not rf: miss.append('review_focus')
    m = c.get('match', {}) or {}
    if not any(str(m.get(k, '')).strip() for k in MATCH_KEYS): miss.append('match.*')
    return miss

# ---------- хэндлеры ----------
HELP = (
    'Команды бота-создателя набора Task 3:\n'
    '/meta — показать метаданные; /setmeta <поле> <значение>\n'
    '   поля: topic, submission_id, creator_id, cutoff_year, review_goal\n'
    '/list — список кейсов и статус готовности\n'
    '/show <case_id> — показать кейс\n'
    '/add <stratum> — добавить кейс (multimodal_hard|temporal_hard|easy_control)\n'
    '/del <case_id> — удалить кейс\n'
    '/set <case_id> <поле> <значение> — задать поле\n'
    '   поля: ' + ', '.join(SCALAR_FIELDS) + ', match.<ключ>\n'
    '/focus <case_id> <v1,v2> — review_focus (evidence,visual_fact,temporal,overall)\n'
    '/errors <case_id> <v1,v2> — expected_error_modes\n'
    '/validate — проверить готовность всех кейсов и пропорции\n'
    '/export — выгрузить task3_ab_case_manifest.filled.json\n'
    '/form — ссылка на Google Форму для ручной отправки\n'
)

async def cmd_start(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = load_state(u.effective_chat.id); save_state(u.effective_chat.id, st)
    await u.message.reply_text('Форма создателя набора Task 3 (A/B). Загружено кейсов: %d.\n\n%s'
                               % (len(st['cases']), HELP))

async def cmd_help(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text(HELP)

async def cmd_meta(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = load_state(u.effective_chat.id)
    m = st['experiment_meta']
    await u.message.reply_text('Метаданные:\n' + '\n'.join('%s: %s' % (k, m.get(k, '')) for k in
                               ['topic', 'submission_id', 'creator_id', 'cutoff_year', 'review_goal']))

async def cmd_setmeta(u: Update, c: ContextTypes.DEFAULT_TYPE):
    if len(c.args) < 2:
        return await u.message.reply_text('Формат: /setmeta <поле> <значение>')
    field, val = c.args[0], ' '.join(c.args[1:])
    st = load_state(u.effective_chat.id)
    if field not in st['experiment_meta']:
        return await u.message.reply_text('Неизвестное поле метаданных.')
    st['experiment_meta'][field] = val
    save_state(u.effective_chat.id, st)
    await u.message.reply_text('OK: %s = %s' % (field, val))

async def cmd_list(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = load_state(u.effective_chat.id)
    if not st['cases']:
        return await u.message.reply_text('Кейсов нет. Добавьте: /add multimodal_hard')
    out = []
    for cs in st['cases']:
        miss = required_missing(cs)
        out.append('%s [%s] %s — %s' % (cs['case_id'], cs['stratum'],
                   'ГОТОВ' if not miss else 'нужно: ' + ','.join(miss),
                   (cs.get('paper_title') or '')[:40]))
    await u.message.reply_text('\n'.join(out))

def render_case(cs):
    m = cs.get('match', {})
    mset = {k: v for k, v in m.items() if str(v).strip()}
    return (
        '%s [%s]\n' % (cs['case_id'], cs['stratum']) +
        'paper_title: %s\n' % cs.get('paper_title', '') +
        'paper_id: %s   year: %s\n' % (cs.get('paper_id', ''), cs.get('year', '')) +
        'evidence_kind: %s   page_hint: %s\n' % (cs.get('evidence_kind', ''), cs.get('page_hint', '')) +
        'creator_prompt: %s\n' % cs.get('creator_prompt', '') +
        'creator_rationale: %s\n' % cs.get('creator_rationale', '') +
        'review_focus: %s\n' % ', '.join(cs.get('review_focus', [])) +
        'expected_error_modes: %s\n' % ', '.join(cs.get('expected_error_modes', [])) +
        'match: %s\n' % (', '.join('%s=%s' % (k, v) for k, v in mset.items()) or '(пусто)') +
        'notes: %s\n' % (cs.get('notes', '') or '') +
        'статус: %s' % ('ГОТОВ' if not required_missing(cs) else 'нужно: ' + ','.join(required_missing(cs)))
    )

async def cmd_show(u: Update, c: ContextTypes.DEFAULT_TYPE):
    if not c.args:
        return await u.message.reply_text('Формат: /show <case_id>')
    st = load_state(u.effective_chat.id)
    cs = find_case(st, c.args[0])
    if not cs:
        return await u.message.reply_text('Нет кейса %s' % c.args[0])
    await u.message.reply_text(render_case(cs))

async def cmd_add(u: Update, c: ContextTypes.DEFAULT_TYPE):
    stratum = c.args[0] if c.args else 'multimodal_hard'
    if stratum not in STRATA:
        return await u.message.reply_text('stratum: ' + ', '.join(STRATA))
    st = load_state(u.effective_chat.id)
    nums = [int(x['case_id'].split('_')[-1]) for x in st['cases'] if x['case_id'].split('_')[-1].isdigit()]
    cs = new_case(stratum, (max(nums) + 1) if nums else 1)
    st['cases'].append(cs)
    save_state(u.effective_chat.id, st)
    await u.message.reply_text('Добавлен %s [%s]' % (cs['case_id'], stratum))

async def cmd_del(u: Update, c: ContextTypes.DEFAULT_TYPE):
    if not c.args:
        return await u.message.reply_text('Формат: /del <case_id>')
    st = load_state(u.effective_chat.id)
    before = len(st['cases'])
    st['cases'] = [x for x in st['cases'] if x['case_id'] != c.args[0]]
    save_state(u.effective_chat.id, st)
    await u.message.reply_text('Удалено' if len(st['cases']) < before else 'Не найдено')

async def cmd_set(u: Update, c: ContextTypes.DEFAULT_TYPE):
    if len(c.args) < 3:
        return await u.message.reply_text('Формат: /set <case_id> <поле> <значение>')
    cid, field, val = c.args[0], c.args[1], ' '.join(c.args[2:])
    st = load_state(u.effective_chat.id)
    cs = find_case(st, cid)
    if not cs:
        return await u.message.reply_text('Нет кейса %s' % cid)
    if field.startswith('match.'):
        key = field.split('.', 1)[1]
        if key not in MATCH_KEYS:
            return await u.message.reply_text('match-ключи: ' + ', '.join(MATCH_KEYS))
        cs.setdefault('match', dict(EMPTY_MATCH))[key] = val
    elif field in SCALAR_FIELDS:
        if field == 'evidence_kind' and val not in EVIDENCE:
            return await u.message.reply_text('evidence_kind: ' + ', '.join(EVIDENCE))
        if field == 'stratum' and val not in STRATA:
            return await u.message.reply_text('stratum: ' + ', '.join(STRATA))
        cs[field] = val
    else:
        return await u.message.reply_text('Неизвестное поле. См. /help')
    save_state(u.effective_chat.id, st)
    await u.message.reply_text('OK. Статус: %s' %
        ('ГОТОВ' if not required_missing(cs) else 'нужно: ' + ','.join(required_missing(cs))))

async def _set_list(u, c, field, valid):
    if len(c.args) < 2:
        return await u.message.reply_text('Формат: /%s <case_id> <v1,v2>' % field.split('_')[0])
    cid = c.args[0]
    vals = [x.strip() for x in ' '.join(c.args[1:]).replace(' ', ',').split(',') if x.strip()]
    bad = [v for v in vals if v not in valid]
    if bad:
        return await u.message.reply_text('Недопустимо: %s. Можно: %s' % (','.join(bad), ', '.join(valid)))
    st = load_state(u.effective_chat.id)
    cs = find_case(st, cid)
    if not cs:
        return await u.message.reply_text('Нет кейса %s' % cid)
    cs[field] = vals
    save_state(u.effective_chat.id, st)
    await u.message.reply_text('OK: %s = %s' % (field, ', '.join(vals)))

async def cmd_focus(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await _set_list(u, c, 'review_focus', FOCUS)

async def cmd_errors(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await _set_list(u, c, 'expected_error_modes', ERRORS)

async def cmd_validate(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = load_state(u.effective_chat.id)
    cases = st['cases']
    n = len(cases)
    not_ready = [(cs['case_id'], required_missing(cs)) for cs in cases if required_missing(cs)]
    from collections import Counter
    cnt = Counter(cs['stratum'] for cs in cases)
    targets = {'multimodal_hard': 60, 'temporal_hard': 25, 'easy_control': 15}
    mix = []
    for k, t in targets.items():
        pct = round(100 * cnt.get(k, 0) / n) if n else 0
        mix.append('%s %d/%d=%d%% (цель %d%%, %s)' % (k, cnt.get(k, 0), n, pct, t,
                   'ok' if abs(pct - t) <= 12 else 'вне допуска'))
    msg = 'Всего кейсов: %d\n' % n
    msg += 'Готовы все: %s\n' % ('ДА' if not not_ready else 'НЕТ')
    if not_ready:
        msg += '\n'.join('  %s нужно: %s' % (cid, ','.join(m)) for cid, m in not_ready) + '\n'
    msg += '\nПропорции:\n' + '\n'.join('  ' + x for x in mix)
    if n < 8:
        msg += '\n\nВнимание: для экспорта рекомендуется >= 8 кейсов.'
    await u.message.reply_text(msg)

async def cmd_export(u: Update, c: ContextTypes.DEFAULT_TYPE):
    st = load_state(u.effective_chat.id)
    out = os.path.join(DATA_DIR, '%s_task3_ab_case_manifest.filled.json' % u.effective_chat.id)
    json.dump(st, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    not_ready = [cs['case_id'] for cs in st['cases'] if required_missing(cs)]
    caption = 'task3_ab_case_manifest.filled.json'
    if not_ready:
        caption += '\nВнимание: не готовы кейсы: ' + ', '.join(not_ready)
    with open(out, 'rb') as fh:
        await u.message.reply_document(document=InputFile(fh, filename='task3_ab_case_manifest.filled.json'),
                                       caption=caption)

async def cmd_form(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text('Отправка данных — ручной шаг. Google Форма:\n%s' % GOOGLE_FORM_URL)

def main():
    token = env_token()
    if not token:
        raise SystemExit('Нет токена бота: создайте telegram_bot/.env со строкой '
                         'TELEGRAM_BOT_TOKEN=... (см. .env.example) или задайте переменную окружения.')
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler('start', cmd_start))
    app.add_handler(CommandHandler('help', cmd_help))
    app.add_handler(CommandHandler('meta', cmd_meta))
    app.add_handler(CommandHandler('setmeta', cmd_setmeta))
    app.add_handler(CommandHandler('list', cmd_list))
    app.add_handler(CommandHandler('show', cmd_show))
    app.add_handler(CommandHandler('add', cmd_add))
    app.add_handler(CommandHandler('del', cmd_del))
    app.add_handler(CommandHandler('set', cmd_set))
    app.add_handler(CommandHandler('focus', cmd_focus))
    app.add_handler(CommandHandler('errors', cmd_errors))
    app.add_handler(CommandHandler('validate', cmd_validate))
    app.add_handler(CommandHandler('export', cmd_export))
    app.add_handler(CommandHandler('form', cmd_form))
    logging.info('Bot started.')
    app.run_polling()

if __name__ == '__main__':
    main()
