from __future__ import annotations

from .runtime import create_mcp_server, run_mcp_server


mcp = create_mcp_server()


def main() -> None:
    run_mcp_server(mcp)


if __name__ == "__main__":
    main()
