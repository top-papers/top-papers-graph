# Validation summary (2026-04-10)

- Pytest: **85 passed, 3 skipped, 14 warnings in 28.97s**
- Pytest log: `reports/final_pytest_run_2026_04_10.log`

## Task 3 notebook runs

### task3_single_main

- ok: **True**
- root: `reports/validation_2026_04_10/notebook_runs/manual_single`
- manifest: `reports/validation_2026_04_10/notebook_runs/manual_single/temporal-catalyst-latency-forecasting/task3_manifest.json`
- hypotheses_json: `reports/validation_2026_04_10/notebook_runs/manual_single/temporal-catalyst-latency-forecasting/hypotheses_ranked.json`
- offline_html: `reports/validation_2026_04_10/notebook_runs/manual_single/temporal-catalyst-latency-forecasting/expert_review/offline_review/task3_hypothesis_review_offline_ab.html`
- expert_zip: `reports/validation_2026_04_10/notebook_runs/manual_single/temporal-catalyst-latency-forecasting/expert_review/expert_hypothesis_artifacts_bundle.zip`

### task3_dual_main

- ok: **True**
- root: `reports/validation_2026_04_10/notebook_runs/manual_dual`
- manifest_alpha: `reports/validation_2026_04_10/notebook_runs/manual_dual/variant_alpha/offline-dual-local-model-blind-review/task3_manifest.json`
- manifest_beta: `reports/validation_2026_04_10/notebook_runs/manual_dual/variant_beta/offline-dual-local-model-blind-review/task3_manifest.json`
- offline_html: `reports/validation_2026_04_10/notebook_runs/manual_dual/variant_alpha/dual_local_model_blind_review/expert_review/offline_review/task3_dual_local_model_review_offline_ab.html`
- expert_zip: `reports/validation_2026_04_10/notebook_runs/manual_dual/variant_alpha/dual_local_model_blind_review/expert_review/expert_review/expert_dual_model_blind_review_bundle.zip`
- owner_key: `reports/validation_2026_04_10/notebook_runs/manual_dual/variant_alpha/dual_local_model_blind_review/expert_review/owner_only/task3_dual_local_model_blind_key.json`

### task3_single_colab

- ok: **True**
- root: `reports/validation_2026_04_10/notebook_runs/manual_single_colab`
- manifest: `reports/validation_2026_04_10/notebook_runs/manual_single_colab/temporal-catalyst-latency-forecasting/task3_manifest.json`
- hypotheses_json: `reports/validation_2026_04_10/notebook_runs/manual_single_colab/temporal-catalyst-latency-forecasting/hypotheses_ranked.json`
- offline_html: `reports/validation_2026_04_10/notebook_runs/manual_single_colab/temporal-catalyst-latency-forecasting/expert_review/offline_review/task3_hypothesis_review_offline_ab.html`
- expert_zip: `reports/validation_2026_04_10/notebook_runs/manual_single_colab/temporal-catalyst-latency-forecasting/expert_review/expert_hypothesis_artifacts_bundle.zip`

### task3_dual_colab

- ok: **True**
- root: `reports/validation_2026_04_10/notebook_runs/manual_dual_colab_alone`
- manifest_alpha: `reports/validation_2026_04_10/notebook_runs/manual_dual_colab_alone/variant_alpha/offline-dual-local-model-blind-review/task3_manifest.json`
- manifest_beta: `reports/validation_2026_04_10/notebook_runs/manual_dual_colab_alone/variant_beta/offline-dual-local-model-blind-review/task3_manifest.json`
- offline_html: `reports/validation_2026_04_10/notebook_runs/manual_dual_colab_alone/variant_alpha/dual_local_model_blind_review/expert_review/offline_review/task3_dual_local_model_review_offline_ab.html`
- expert_zip: `reports/validation_2026_04_10/notebook_runs/manual_dual_colab_alone/variant_alpha/dual_local_model_blind_review/expert_review/expert_review/expert_dual_model_blind_review_bundle.zip`
- owner_key: `reports/validation_2026_04_10/notebook_runs/manual_dual_colab_alone/variant_alpha/dual_local_model_blind_review/expert_review/owner_only/task3_dual_local_model_blind_key.json`

## Notes

- Task 3 notebooks were executed one-by-one in isolated nbclient runs.
- Validation used synthetic `processed_papers/` input and offline-safe mock settings.
- This validates notebook control flow, artifact creation, blind A/B packaging and offline review generation, but not the quality of real local VLM weights.
