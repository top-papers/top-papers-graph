# -*- coding: utf-8 -*-
"""
Веб-сервер формы создателя набора Task 3 (A/B, top-papers-graph).

Поднимает GUI-форму task3_ab_creator_offline_form_ru.html на http://localhost:9000
и сохраняет заполненный манифест прямо в репозиторий (без конфигов, только GUI).

Маршруты:
    GET  /              — сама форма (HTML).
    GET  /api/manifest  — текущий task3_ab_case_manifest.filled.json с диска.
    POST /api/manifest  — записать присланный JSON в task3_ab_case_manifest.filled.json.

Зависимости: только стандартная библиотека Python 3.
Запуск:      python3 serve.py [--port 9000]
"""
import os, sys, json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)                                  # корень репозитория
FORM = os.path.join(HERE, 'task3_ab_creator_offline_form_ru.html')
MANIFEST = os.path.join(BASE, 'task3_ab_case_manifest.filled.json')

DEFAULT_PORT = 9000


def pick_port():
    # --port N  или  PORT=N  или  дефолт 9000
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == '--port' and i + 1 < len(argv):
            return int(argv[i + 1])
        if a.startswith('--port='):
            return int(a.split('=', 1)[1])
    return int(os.environ.get('PORT', DEFAULT_PORT))


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype='application/json; charset=utf-8'):
        data = body if isinstance(body, bytes) else body.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(data)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False), 'application/json; charset=utf-8')

    def do_GET(self):
        path = self.path.split('?', 1)[0]
        if path in ('/', '/index.html'):
            if not os.path.exists(FORM):
                return self._send(503,
                    'Форма не найдена. Сначала соберите её: python3 scripts/build.py',
                    'text/plain; charset=utf-8')
            with open(FORM, 'rb') as fh:
                return self._send(200, fh.read(), 'text/html; charset=utf-8')
        if path == '/api/manifest':
            if not os.path.exists(MANIFEST):
                return self._json(200, {})
            try:
                with open(MANIFEST, encoding='utf-8') as fh:
                    return self._json(200, json.load(fh))
            except Exception as e:
                return self._json(500, {'ok': False, 'error': 'битый манифест на диске: %s' % e})
        return self._send(404, 'not found', 'text/plain; charset=utf-8')

    def do_HEAD(self):
        self.do_GET()

    def do_POST(self):
        path = self.path.split('?', 1)[0]
        if path != '/api/manifest':
            return self._send(404, 'not found', 'text/plain; charset=utf-8')
        length = int(self.headers.get('Content-Length', 0) or 0)
        raw = self.rfile.read(length) if length else b''
        try:
            obj = json.loads(raw.decode('utf-8'))
        except Exception as e:
            return self._json(400, {'ok': False, 'error': 'не JSON: %s' % e})
        if not isinstance(obj, dict) or 'cases' not in obj:
            return self._json(400, {'ok': False, 'error': 'ожидался манифест с полем "cases"'})
        try:
            if os.path.exists(MANIFEST):
                with open(MANIFEST, 'rb') as fh:
                    prev = fh.read()
                with open(MANIFEST + '.bak', 'wb') as fh:
                    fh.write(prev)
            with open(MANIFEST, 'w', encoding='utf-8') as fh:
                json.dump(obj, fh, ensure_ascii=False, indent=2)
        except Exception as e:
            return self._json(500, {'ok': False, 'error': 'не удалось записать: %s' % e})
        return self._json(200, {'ok': True, 'path': MANIFEST, 'cases': len(obj.get('cases', []))})

    def log_message(self, fmt, *args):
        sys.stderr.write('%s - %s\n' % (self.address_string(), fmt % args))


def main():
    port = pick_port()
    httpd = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    print('Форма создателя набора Task 3 — http://localhost:%d' % port)
    print('Манифест на диске: %s' % MANIFEST)
    print('Ctrl+C для остановки.')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\nОстановлено.')
        httpd.server_close()


if __name__ == '__main__':
    main()
