from __future__ import annotations

import tempfile
from pathlib import Path

from scireason.mm import vlm


class _DummyWorker:
    def __init__(self, text: str):
        self._scireason_stderr_file = tempfile.TemporaryFile(mode='w+t', encoding='utf-8')
        self._scireason_stderr_file.write(text)
        self._scireason_stderr_file.flush()


def test_tail_worker_stderr_reads_from_file() -> None:
    worker = _DummyWorker('abc\n' * 100)
    tail = vlm._tail_worker_stderr(worker, limit=20)
    assert tail.endswith('abc')
    worker._scireason_stderr_file.close()
