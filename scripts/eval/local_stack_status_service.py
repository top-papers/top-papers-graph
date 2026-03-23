#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict

from scireason.config import settings
from scireason.graph.qdrant_store import QdrantStore
from scireason.graph.memgraph_store import MemgraphTemporalStore


def qdrant_report() -> Dict[str, Any]:
    store = QdrantStore(url=settings.qdrant_url)
    client = store._client
    collections = []
    for c in getattr(client.get_collections(), 'collections', []):
        name = c.name
        try:
            info = client.get_collection(name)
            points = int(getattr(info, 'vectors_count', None) or getattr(info, 'points_count', 0) or 0)
        except Exception:
            points = 0
        collections.append({'name': name, 'points': points})
    return {'url': settings.qdrant_url, 'collections': collections}


def memgraph_report() -> Dict[str, Any]:
    mg = MemgraphTemporalStore(uri=settings.memgraph_uri)
    try:
        counts: Dict[str, int] = {}
        with mg._driver.session() as s:
            queries = {
                'papers': 'MATCH (n:Paper) RETURN count(n) AS c',
                'chunks': 'MATCH (n:Chunk) RETURN count(n) AS c',
                'assertions': 'MATCH (n:Assertion) RETURN count(n) AS c',
                'events': 'MATCH (n:Event) RETURN count(n) AS c',
            }
            for key, q in queries.items():
                rec = s.run(q).single()
                counts[key] = int(rec['c']) if rec else 0
        analytics = mg.run_mage_analytics(limit=10)
        return {'uri': settings.memgraph_uri, 'counts': counts, 'analytics_preview': analytics}
    finally:
        mg.close()


def build_report() -> Dict[str, Any]:
    return {
        'status': 'ok',
        'qdrant': qdrant_report(),
        'memgraph': memgraph_report(),
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {'/health', '/'}:
            self._send(200, {'status': 'ok'})
            return
        if self.path == '/report':
            self._send(200, build_report())
            return
        self._send(404, {'status': 'not_found', 'path': self.path})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


def main() -> None:
    host = os.environ.get('LOCAL_STACK_SERVICE_HOST', '127.0.0.1')
    port = int(os.environ.get('LOCAL_STACK_SERVICE_PORT', '8787'))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f'local-stack-status-service listening on http://{host}:{port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
