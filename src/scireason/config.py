from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ===== LLM =====
    # Default: `auto` for classroom robustness.
    # auto -> try local Ollama (if reachable) -> g4f (if installed) -> LiteLLM providers -> mock (offline)
    llm_provider: str = "auto"  # auto|mock|ollama|g4f|openai|anthropic|...
    llm_model: str = "auto"
    ollama_base_url: str = "http://localhost:11434"

    # g4f fine-tuning (optional)
    g4f_providers: str | None = None
    g4f_api_key: str | None = None

    # ===== Embeddings =====
    embed_provider: str = "hash"  # hash|sentence-transformers|openai|ollama|...
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hash_embed_dim: int = 384

    # ===== Infra (optional services) =====
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
    contact_email: str | None = None
    user_agent: str | None = None

    # Semantic Scholar
    s2_api_key: str | None = None

    # NCBI / PubMed
    ncbi_api_key: str | None = None
    ncbi_tool: str = "top-papers-graph"
    ncbi_email: str | None = None

    # Crossref / OpenAlex
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
    domain_id: str = "science"
    domain_config_path: str | None = None

    # ===== Agentic hypothesis generation (code-writing agent) =====
    hyp_agent_enabled: bool = True
    # internal|smolagents
    # - internal: lightweight built-in code agent (works fully offline with --llm-provider mock)
    # - smolagents: Hugging Face smolagents CodeAgent (optional dependency)
    hyp_agent_backend: str = "internal"
    hyp_agent_max_steps: int = 4
    hyp_agent_timeout_seconds: int = 20

    # ===== smolagents integration (optional) =====
    # Only used when HYP_AGENT_BACKEND=smolagents.
    #
    # smol_model_backend:
    # - scireason: wrap this project's LLM router (LLM_PROVIDER=auto|g4f|ollama|...) into a smolagents Model
    # - transformers: smolagents.TransformersModel for local HF models (requires smolagents[transformers])
    # - g4f: direct smolagents Model that calls g4f client (requires g4f)
    smol_model_backend: str = "scireason"
    smol_model_id: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    smol_max_new_tokens: int = 768
    smol_device_map: str | None = None
    smol_torch_dtype: str | None = None  # e.g. float16|bfloat16
    smol_g4f_model: str = "auto"
    smol_executor: str = "local"  # local|docker (docker requires Docker)
    smol_print_steps: bool = False

    # ===== Optional GNN (PyTorch Geometric) for hypothesis discovery =====
    # Disabled by default to keep the base installation lightweight.
    # Enable via:
    #   HYP_GNN_ENABLED=1
    hyp_gnn_enabled: bool = False
    hyp_gnn_epochs: int = 80
    hyp_gnn_hidden_dim: int = 64
    hyp_gnn_lr: float = 0.01
    # To keep training fast for classroom-sized runs, we restrict to an induced
    # subgraph of the most connected terms.
    hyp_gnn_node_cap: int = 300
    hyp_gnn_seed: int = 7


settings = Settings()
