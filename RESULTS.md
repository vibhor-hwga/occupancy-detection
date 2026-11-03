# Experiment log

One row per run you intend to keep. The MLflow run id links to the full
record (config, scaler, model, metrics, plots).

| Date | Config | Model | Source test acc | Target acc | Drop | MLflow run id | Purpose / notes |
|------|--------|-------|-----------------|------------|------|---------------|-----------------|
| 2026-11-04 | baseline | random_forest (100 trees, depth 8) | 0.9467 | 0.3747 | 0.5721 | 165f6a6323ae4db9a33b250bd7f7fcfd | Zero-shot UCI -> HPDmobile (all 6 homes, 965,957 1-min rows). Commit b594deb. Target balanced acc 0.4971 (chance level); F1 0.9256 -> 0.3131. Model predicts mostly unoccupied (target recall 0.20). Raw shift: Temperature +7.5 and Light +7.9 source SDs. |
