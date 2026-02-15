from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ===== LLM =====
    # Default: g4f (routes requests to available providers; see g4f docs).
    # You can always switch back to LiteLLM providers via env:
    #   LLM_PROVIDER=ollama  LLM_MODEL=llama3.2
    llm_provider: str = "g4f"  # g4f|ollama|openai|anthropic|...
    llm_model: str = "deepseek-r1"
    ollama_base_url: str = "http://localhost:11434"

    # ===== Embeddings =====
    # Default: hash embeddings (no heavyweight deps, no API keys). If you want higher quality
    # local embeddings: `pip install -e '.[embeddings]'` and set EMBED_PROVIDER=sentence-transformers.
    embed_provider: str = "hash"  # hash|sentence-transformers|openai|ollama|...
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hash_embed_dim: int = 384

    # ===== Infra =====
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "please_change_me"

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None

    grobid_url: str = "http://localhost:8070"

    # ===== HTTP Client (cache + rate limiting) =====
    http_cache_dir: str = ".cache/http"
    http_cache_ttl_seconds: int = 7 * 24 * 3600
    http_timeout_seconds: int = 30


    # ===== External APIs =====
    # A single place to provide your contact email for "polite" API pools (Crossref/OpenAlex/etc.).
    contact_email: str | None = None
    # Optional explicit User-Agent. If not set, we will build one from the project name + contact_email.
    user_agent: str | None = None

    # Semantic Scholar
    s2_api_key: str | None = None

    # NCBI / PubMed (E-utilities). API key is optional but can raise rate limits.
    ncbi_api_key: str | None = None
    ncbi_tool: str = "top-papers-graph"
    ncbi_email: str | None = None

    # Crossref / OpenAlex "polite" pool mailto (optional; falls back to contact_email)
    crossref_mailto: str | None = None
    openalex_mailto: str | None = None
    openalex_api_key: str | None = None

    # ===== Demo store (retrieval few-shot) =====
    demo_enabled: bool = True
    demo_schema_version: str = "1.0"
    demo_quality: str = "gold"  # gold|silver
    demo_top_k_triplets: int = 3
    demo_top_k_hypothesis: int = 2
    demo_max_chars_total: int = 3500
    demo_collection_triplets: str = "demos_temporal_triplets"
    demo_collection_hypothesis: str = "demos_hypothesis_test"

    # ===== Multimodal (VL + MM embeddings) =====
    vlm_backend: str = "none"  # none|qwen2_vl|llava|phi3_vision
    vlm_model_id: str = "Qwen/Qwen2-VL-7B-Instruct"
    mm_embed_backend: str = "none"  # none|open_clip
    open_clip_model: str = "ViT-B-32"
    open_clip_pretrained: str = "laion2b_s34b_b79k"
    pdf_render_dpi: int = 150

    # ===== Temporal GraphRAG =====
    temporal_default_granularity: str = "year"  # year|month|day

    # ===== Domain =====
    domain_id: str = "science"  # e.g. ied_fastcharge
    domain_config_path: str | None = None  # optional explicit path to configs/domains/<id>.yaml


settings = Settings()
