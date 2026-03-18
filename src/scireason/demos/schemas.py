from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


DemoTask = Literal["temporal_triplets", "hypothesis_test"]
DemoQuality = Literal["gold", "silver"]


class DemoSource(BaseModel):
    artifact_path: str = Field(description="Path to the originating expert artifact in the repo/datalake.")
    reviewer_id: Optional[str] = None
    timestamp: Optional[str] = None


class DemoExample(BaseModel):
    """A single retrieval-few-shot demonstration example.

    We store demos in Qdrant with vector = embedding(input_text), payload = {input, output, metadata}.
    The agent retrieves top-k demos for the current input and injects them into the prompt as few-shot examples.
    """

    id: str
    task: DemoTask
    domain: str = Field(description="Domain identifier, e.g. science")
    schema_version: str = "1.0"
    quality: DemoQuality = "gold"

    source: Optional[DemoSource] = None
    tags: List[str] = Field(default_factory=list)

    # We keep input/output as JSON-serializable dicts to avoid circular imports across modules.
    input: Dict[str, Any]
    output: Any

    def input_text(self) -> str:
        """Text used for semantic search.

        For different tasks we can combine different fields.
        """
        if self.task == "temporal_triplets":
            txt = str(self.input.get("chunk_text", ""))
            yr = self.input.get("paper_year")
            return f"[paper_year={yr}] {txt}" if yr else txt
        if self.task == "hypothesis_test":
            hyp = str(self.input.get("hypothesis_text", ""))
            # keep short: do not include full ctx to avoid leaking large data into embeddings
            ctx = str(self.input.get("ctx_head", ""))
            return f"{hyp}\n\n{ctx}".strip()
        return str(self.input)
