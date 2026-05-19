from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ===== LLM =====
    llm_provider: str = "auto"  # auto|mock|ollama|g4f|openai|anthropic|...
    llm_model: str = "auto"
    ollama_base_url: str = "http://localhost:11434"
    ollama_runtime_serialized: bool = True
    g4f_providers: str | None = None
    g4f_api_key: str | None = None
    task2_default_g4f_model: str = "r1-1776"
    task2_default_local_vlm_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    task2_default_ollama_vlm_model: str = "qwen2.5vl:3b"
    llm_request_timeout_seconds: int = 180
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
    ocr_vlm_fallback_enabled: bool = False
    ocr_vlm_min_chars_per_page: int = 120
    ocr_vlm_repair_noisy_text_enabled: bool = True
    ocr_vlm_repair_suspicious_chars_per_1k: int = 12

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
    vlm_max_new_tokens: int = 256
    vlm_structured_output: bool = True
    mm_embed_backend: str = "none"  # none|open_clip
    open_clip_model: str = "ViT-B-32"
    open_clip_pretrained: str = "laion2b_s34b_b79k"
    pdf_render_dpi: int = 150
    vlm_request_timeout_seconds: int = 45
    local_vlm_request_timeout_seconds: int = 180
    local_vlm_startup_timeout_seconds: int = 900
    vlm_min_pixels: int = 256 * 28 * 28
    vlm_max_pixels: int = 1280 * 28 * 28
    local_vlm_allow_inprocess_fallback: bool = False

    # ===== Temporal GraphRAG =====
    temporal_default_granularity: str = "year"

    # ===== Качество извлечения триплетов =====
    # Контекст статьи в промпте (title + abstract → каждый чанк)
    triplet_paper_context_enabled: bool = False
    triplet_paper_context_max_chars: int = 500

    # Fuzzy-нормализация сущностей после построения графа
    entity_normalization_enabled: bool = False
    entity_normalization_threshold: float = 0.85

    # Summary-триплеты (один LLM-вызов на статью для ключевых утверждений)
    paper_summary_triplets_enabled: bool = False
    paper_summary_max_input_chars: int = 3000

    # Каноническая лексика для кросс-документных рёбер
    canonical_vocabulary_enabled: bool = False
    canonical_vocabulary_max_concepts: int = 30

    # Верификация рёбер (confidence gate + LLM-проверка)
    triplet_verify_enabled: bool = False
    triplet_verify_confidence_threshold: float = 0.5
    triplet_verify_batch_size: int = 80

    # Оценочная функция качества утверждений (10 признаков → логистическая регрессия)
    assertion_scorer_enabled: bool = True
    assertion_scorer_weights_path: str = "data/derived/assertion_scorer_weights.json"
    assertion_scorer_accept_threshold: float = 0.7
    assertion_scorer_reject_threshold: float = 0.25

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
