from __future__ import annotations

import re
import xml.etree.ElementTree as ET


def tei_to_plaintext(tei_xml: str) -> str:
    """Грубое извлечение текста из TEI XML.
    На практике можно улучшить, но для учебного проекта этого достаточно.
    """
    try:
        root = ET.fromstring(tei_xml)
    except Exception:
        return ""

    # TEI namespaces
    ns = {"tei": "http://www.tei-c.org/ns/1.0"}
    texts = []
    for tag in ["title", "abstract", "p"]:
        for node in root.findall(f".//tei:{tag}", ns):
            if node.text and node.text.strip():
                texts.append(node.text.strip())

    raw = "\n\n".join(texts)
    raw = re.sub(r"\s+", " ", raw)
    return raw.strip()
