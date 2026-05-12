# Updated expert notebooks bundle

This archive includes synchronized expert notebooks:

- `task1_reasoning_trajectories_onine_offline_forms.ipynb`
- `task2_temporal_graph_validation.ipynb`
- `task3_multimodal_temporal_hypothesis_generation.ipynb`

Both copies are aligned with the tested repository notebooks in `notebooks/` and use the current Task 2 pipeline with:

- YAML/JSON exclusion filters for anti-leakage review
- triplet importance scoring and threshold filtering
- graph analytics in visualization pages (communities, cliques, centralities, k-core)
- offline review package generation compatible with downstream VLM fine-tuning datasets


Task 3 notebook adds:

- query / identifier / commands / Task 1 YAML inputs
- local Hugging Face / Transformers VLM routing and g4f routing
- offline A/B expert review HTML generation
- expert artifact ZIP packaging for hypothesis review

## Task 3 HF benchmark uploader

Added `notebooks/top_papers_graph_task3_hf_benchmark_colab.ipynb` for converting Task 3 case manifests from Google Sheets into a VLM generation benchmark dataset and uploading it to `top-papers/top-papers-graph-benchmark`.

See `README_TASK3_HF_BENCHMARK_DATASET_RU.md` for details.
