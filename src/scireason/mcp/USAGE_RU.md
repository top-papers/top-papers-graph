# Usage

## 1. Быстрый запуск для проверки

Рекомендуемый путь для локальной проверки:

```bash
cd /home/karaluv/proga/top-papers-graph
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[mcp]'
```

Проверка, что сервер стартует:

```bash
PYTHONPATH=src python -m scireason.mcp.server
```

(Потом вырубить он там зависнет, Ctrl+C для выхода, т.к. все равно в config придется прописывать)

## 2. Пример через Copilot CLI

Нужна папка `~/.copilot`:

```bash
mkdir -p ~/.copilot
```

Конфиг MCP server:

Файл: `~/.copilot/mcp-config.json`

У меня это так было, у вас ваши пути должны быть и ваш питон где все стоит

```json
{
  "mcpServers": {
    "scireason-local": {
      "type": "local",
      "command": "/bin/bash",
      "args": [
        "-lc",
        "cd /home/karaluv/proga/top-papers-graph && PYTHONPATH=src MCP_TOOLSETS=demo,papers /home/karaluv/proga/top-papers-graph/.venv/bin/python -m scireason.mcp.server"
      ],
      "tools": ["*"]
    }
  }
}
```

Запуск:

```bash
copilot
```

Проверка в Copilot CLI:

1. `/mcp show`
2. `/mcp show scireason-local`
3. `Use the scireason-local MCP server and call doctor_tool.`
4. `Use search_papers_tool from scireason-local to find papers about graph rag, limit 3.`

## 3. Пример добавления функции

Если нужна новая авто-тулза, ее не надо прописывать в `server.py`.

Пример:

```python
from __future__ import annotations

from ..decorators import scireason_mcp_tool


@scireason_mcp_tool(toolset="demo")
def demo_ping_tool(name: str = "world") -> dict[str, str]:
    """Return a tiny payload so the MCP wiring is easy to test."""
    return {"message": f"hello, {name}"}
```

Куда положить:

- `src/scireason/mcp/toolsets/demo.py`
- или новый модуль внутри `src/scireason/mcp/toolsets/`

Что нужно:

- type hints
- docstring
- json-совместимый return


## 4. Ручная регистрация

Если нужен ручной способ, см. `manual.py`.

Там через `mcp.tool(...)` добавлены:

- `doctor_tool`
- `health_tool`
