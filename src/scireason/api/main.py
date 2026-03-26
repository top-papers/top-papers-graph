from __future__ import annotations

import os

try:
    import uvicorn
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "API extra is not installed. Install with: pip install -e '.[api]'"
    ) from e

from .app import create_app


def main() -> None:
    host = os.environ.get("TPG_HOST", "0.0.0.0")
    port = int(os.environ.get("TPG_PORT", "8000"))
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    main()
