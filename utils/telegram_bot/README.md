# Telegram-бот создателя набора Task 3 (A/B)

Альтернатива офлайн веб-форме `task3_ab_creator_offline_form_ru.html`. Бот ведёт
эксперта по кейсам, проверяет готовность по той же логике, что и форма
(из cell 8 блокнота), и выгружает `task3_ab_case_manifest.filled.json`.

## Установка
```bash
cd telegram_bot
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # затем впишите в .env токен бота
```

## Токен бота (`.env`)
Единственный секрет — токен от @BotFather. Берётся из файла `.env` (или из
переменной окружения `TELEGRAM_BOT_TOKEN`). Никаких других конфигов нет.

```
# telegram_bot/.env
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
```

Файл `.env` в репозиторий **не коммитится** (см. корневой `.gitignore`). Всё
остальное — параметры набора и метаданные — заполняется прямо в чате командами
(см. ниже), а служебные настройки зашиты в `bot.py` как дефолты:
- Google Форма — `https://forms.gle/h5RwEA8DsZh9pBAt8` (команда `/form`);
- рабочая папка состояния по чатам — `telegram_bot/tg_data/`;
- стартовый набор при первом запуске — `../task3_ab_case_manifest.filled.json`
  (по умолчанию пуст; наполняется через `config.yaml` + `scripts/build.py` или прямо в чате).

## Запуск
```bash
python3 bot.py            # токен читается из .env
# или без файла:
TELEGRAM_BOT_TOKEN=... python3 bot.py
```
Без токена бот завершится с понятным сообщением (не трейсбеком).

## Команды
- `/start`, `/help`
- `/meta`, `/setmeta <поле> <значение>` — topic, submission_id, creator_id, cutoff_year, review_goal
- `/list` — кейсы и статус готовности
- `/show <case_id>`
- `/add <stratum>` — multimodal_hard | temporal_hard | easy_control
- `/del <case_id>`
- `/set <case_id> <поле> <значение>` — paper_title, paper_id, year, evidence_kind, page_hint,
  creator_prompt, creator_rationale, stratum, notes, а также `match.<ключ>`
- `/focus <case_id> <v1,v2>` — review_focus (evidence, visual_fact, temporal, overall)
- `/errors <case_id> <v1,v2>` — expected_error_modes
- `/validate` — готовность всех кейсов + проверка пропорций 60/25/15
- `/export` — прислать `task3_ab_case_manifest.filled.json`
- `/form` — ссылка на Google Форму (отправка — ручной шаг, бот её не делает)

## Правило готовности (как в форме)
Кейс «Готов к экспорту», когда непусты: `paper_title`, `creator_prompt`,
`creator_rationale`, `review_focus` (≥1) и **хотя бы одно поле `match.*`**.
