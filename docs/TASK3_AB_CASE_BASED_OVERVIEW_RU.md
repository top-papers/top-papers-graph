# Task 3 case-based A/B assets

Новые артефакты для переработанного дизайна эксперимента:

- `src/scireason/task3_ab_case_manifest.py` — authoring bundle + case-based blind review builder
- `scripts/task3/run_task3_case_based_blind_ab.py` — основной CLI runner
- `scripts/datasphere/run_task3_case_based_blind_ab_job.py` — entrypoint для DataSphere Jobs
- `configs/datasphere/task3_case_based_blind_ab_job.config.yaml` — шаблон DataSphere job config
- `notebooks/task3_ab_testset_authoring_colab.ipynb` — Colab notebook для эксперта-создателя
- `notebooks/task3_case_based_blind_ab_kaggle_cli_launcher.ipynb` — Kaggle notebook, который просто подготавливает окружение и запускает CLI
