from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ===== LLM =====
    llm_provider: str = "auto"  # auto|mock|ollama|g4f|openai|anthropic|...
    llm_model: str = "auto"
    ollama_base_url: str = "http://localhost:11434"
    g4f_providers: str | None = None
    g4f_api_key: str | None = None
    task2_default_g4f_model: str = "r1-1776"
    task2_default_local_vlm_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    llm_request_timeout_seconds: int = 25
    g4f_async_enabled: bool = True
    g4f_async_max_concurrency: int = 3
    g4f_async_retries: int = 3
    g4f_async_retry_backoff_seconds: float = 1.0
    g4f_async_retry_backoff_max_seconds: float = 8.0
    g4f_async_max_models_per_request: int = 3

    # ===== Embeddings =====
    embed_provider: str = "hash"  # hash|sentence-transformers|openai|ollama|...
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hash_embed_dim: int = 384

    # ===== OCR / parsing =====
    ocr_backend: str = "auto"  # auto(default=PaddleOCR->local fallback)|paddleocr|grobid|pymupdf
    paddleocr_lang: str | None = None
    paddleocr_worker_timeout_seconds: int = 90

    # ===== Infra =====
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "please_change_me"

    memgraph_uri: str = "bolt://localhost:7688"
    memgraph_user: str = ""
    memgraph_password: str = ""
    graph_backend: str = "dual"  # dual|neo4j|memgraph|none

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_dense_vector_name: str = "dense"
    qdrant_sparse_vector_name: str = "sparse"
    qdrant_sparse_dim: int = 2048
    qdrant_retrieval_mode: str = "hybrid"  # hybrid|dense
    qdrant_check_compatibility: bool = False

    grobid_url: str = "http://localhost:8070"

    # ===== HTTP Client =====
    http_cache_dir: str = ".cache/http"
    http_cache_ttl_seconds: int = 7 * 24 * 3600
    http_timeout_seconds: int = 30

    # ===== External APIs =====
    contact_email: str | None = None
    user_agent: str | None = None
    s2_api_key: str | None = None
    ncbi_api_key: str | None = None
    ncbi_tool: str = "top-papers-graph"
    ncbi_email: str | None = None
    crossref_mailto: str | None = None
    openalex_mailto: str | None = None
    openalex_api_key: str | None = None

    # ===== Demo store =====
    demo_enabled: bool = True
    demo_schema_version: str = "1.0"
    demo_quality: str = "gold"
    demo_top_k_triplets: int = 3
    demo_top_k_hypothesis: int = 2
    demo_max_chars_total: int = 3500
    demo_collection_triplets: str = "demos_temporal_triplets"
    demo_collection_hypothesis: str = "demos_hypothesis_test"

    # ===== Multimodal =====
    vlm_backend: str = "g4f"  # g4f(default)|none|qwen2_vl|qwen3_vl|llava|phi3_vision
    vlm_model_id: str = "auto"
    vlm_max_new_tokens: int = 512
    vlm_structured_output: bool = True
    mm_embed_backend: str = "none"  # none|open_clip
    open_clip_model: str = "ViT-B-32"
    open_clip_pretrained: str = "laion2b_s34b_b79k"
    pdf_render_dpi: int = 150
    vlm_request_timeout_seconds: int = 45
    vlm_min_pixels: int = 256 * 28 * 28
    vlm_max_pixels: int = 1280 * 28 * 28
    local_vlm_allow_inprocess_fallback: bool = False

    # ===== Temporal GraphRAG =====
    temporal_default_granularity: str = "year"

    # ===== Domain =====
    domain_id: str = "science"
    domain_config_path: str | None = None

    # ===== Agentic hypothesis generation =====
    hyp_agent_enabled: bool = True
    hyp_agent_backend: str = "internal"
    hyp_agent_max_steps: int = 4
    hyp_agent_timeout_seconds: int = 20

    # ===== smolagents =====
    smol_model_backend: str = "scireason"
    smol_model_id: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    smol_max_new_tokens: int = 768
    smol_device_map: str | None = None
    smol_torch_dtype: str | None = None
    smol_g4f_model: str = "auto"
    smol_executor: str = "local"
    smol_print_steps: bool = False

    # ===== Temporal link prediction / TGNN =====
    hyp_tgnn_enabled: bool = True
    hyp_tgnn_backend: str = "auto"  # auto|heuristic|pyg
    hyp_tgnn_recent_window_years: int = 3
    hyp_tgnn_half_life_years: float = 2.0
    hyp_tgnn_min_candidate_score: float = 0.05
    hyp_tgnn_memory_dim: int = 64
    hyp_tgnn_time_dim: int = 16
    hyp_tgnn_epochs: int = 25

    # ===== Static GNN baseline =====
    hyp_gnn_enabled: bool = False
    hyp_gnn_epochs: int = 80
    hyp_gnn_hidden_dim: int = 64
    hyp_gnn_lr: float = 0.01
    hyp_gnn_node_cap: int = 300
    hyp_gnn_seed: int = 7

    # ===== Vector indexing =====
    neo4j_vector_enabled: bool = True
    neo4j_vector_chunk_dimensions: int = 384
    neo4j_vector_assertion_dimensions: int = 384


settings = Settings()
