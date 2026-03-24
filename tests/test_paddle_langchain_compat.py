from __future__ import annotations

import sys
from types import ModuleType

from scireason.ingest.paddleocr_pipeline import _install_langchain_docstore_shim


def test_langchain_docstore_shim_exposes_legacy_document_path(monkeypatch) -> None:
    for name in [
        "langchain.docstore.document",
        "langchain.docstore",
    ]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    _install_langchain_docstore_shim()

    document_mod = sys.modules.get("langchain.docstore.document")
    assert isinstance(document_mod, ModuleType)
    assert hasattr(document_mod, "Document")


def test_langchain_text_splitter_shim_exposes_legacy_path(monkeypatch) -> None:
    for name in [
        "langchain.text_splitter",
    ]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    _install_langchain_docstore_shim()

    splitter_mod = sys.modules.get("langchain.text_splitter")
    assert isinstance(splitter_mod, ModuleType)
    assert hasattr(splitter_mod, "RecursiveCharacterTextSplitter")
