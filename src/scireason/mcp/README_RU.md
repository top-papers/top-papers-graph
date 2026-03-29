# MCP

## Что сделано

В папке `mcp/` собран локальный MCP server на `FastMCP`.

Цель была простая:

- не держать все тулзы в одном `server.py`
- дать общий wrapper для регистрации, чтобы достаточно было написать функцию с декоратором и она уже была бы MCP tool
- показать 2 способа подключения тулз: авто и вручную

## Структура

- `server.py` - entrypoint
- `runtime.py` - bootstrap сервера
- `registry.py` - автообнаружение и отбор тулз
- `decorators.py` - декоратор `@scireason_mcp_tool(...)`
- `manual.py` - примеры ручной регистрации
- `toolsets/` - модули с авто-тулзами
- `specs.py` - структуры метаданных
- `config.py` - env-конфиг

## Как это работает

1. `server.py` создает сервер через `create_mcp_server()`.
2. `runtime.py` читает env-конфиг.
3. `registry.py` импортирует модули из `toolsets/`.
4. Функции с `@scireason_mcp_tool(...)` превращаются в MCP tools.
5. Из `manual.py` отдельно добавляются `doctor_tool` и `health_tool`.

## Регистрация тулз

Есть 2 режима.

### 1. Авто

Функция лежит в `toolsets/` и помечена декоратором.

Минимальные требования:

- type hints на аргументы и return
- docstring
- json-совместимый результат

### 2. Ручная

Регистрация идет напрямую через `mcp.tool(...)`.

Пример лежит в `manual.py`.

## Env

- `MCP_TOOLSETS=demo,papers`
- `MCP_TOOLS=search_papers_tool`
- `MCP_DISABLE_TOOLS=demo_echo_tool`
- `MCP_EXTRA_MODULES=package.module1,package.module2`
- `MCP_READ_ONLY=1`
- `MCP_TRANSPORT=stdio`
- `MCP_MOUNT_PATH=/mcp`

Текущий основной режим: `stdio`.

## Дальше

- `streamable-http`
- внешний доступ через тунели (но надо посмотреть с ngrok)
- более понятный набор интеграций

С `https` уже были пробы, но кажется сделать чтобы оно работало бы из коробки на локальной машине сложновато
