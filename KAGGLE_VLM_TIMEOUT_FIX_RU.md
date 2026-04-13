# Kaggle VLM timeout fix

Что исправлено:
- добавлен preload local VLM worker с отдельным startup timeout;
- page-level timeout увеличен для Kaggle по умолчанию до 300 секунд;
- startup timeout для cold start модели увеличен до 900 секунд;
- page-level `max_new_tokens` теперь берётся из `settings.vlm_max_new_tokens`;
- в Kaggle notebook/runner по умолчанию используется `VLM_MAX_NEW_TOKENS=192`;
- в Kaggle notebook/runner по умолчанию используется более умеренный `VLM_MAX_PIXELS=768*28*28`.

Зачем это нужно:
- 7B/8B VLM на Kaggle может долго стартовать и не укладываться в прежний таймаут 120 секунд;
- генерация на каждой странице с `max_new_tokens=512` была слишком тяжёлой и повышала риск таймаута.
