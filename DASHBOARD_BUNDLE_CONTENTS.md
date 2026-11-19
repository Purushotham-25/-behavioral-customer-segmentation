# DASHBOARD_BUNDLE_CONTENTS.md

## Archive

- Archive: `C:\Users\PURUSHOTHAM\OneDrive\Desktop\Behavioral-customer-segmentation\dashboard_review_bundle.zip`
- Size: **5.5 MiB (5,764,682 bytes)**
- 512 MiB limit: **PASS**

## Files Included

| Relative path | Purpose |
|---|---|
| `dashboard/app.py` | Main saved Phase 2 Streamlit segmentation dashboard. |
| `dashboard/live_marketing.py` | Streamlit live transaction entry and recommendation interface. |
| `dashboard/app_backup.py` | Backup copy of the saved Phase 2 dashboard. |
| `customer_segmentation_dashboard_starter/dashboard/app.py` | Standalone starter copy of the saved Phase 2 dashboard. |
| `src/__init__.py` | Marks the source package. |
| `src/segmentation_pipeline.py` | Authoritative Phase 1 QLDE + K-Means research implementation. |
| `src/live_marketing/__init__.py` | Marks the live-marketing package. |
| `src/live_marketing/config.py` | Project, dataset, model, and live-storage paths plus limits. |
| `src/live_marketing/feature_engine.py` | Historical/live behavioral feature generation. |
| `src/live_marketing/customer_segmenter.py` | Saved assignment reuse and reconstructed Phase 2 model prediction. |
| `src/live_marketing/marketing_engine.py` | Personalized marketing recommendation rules and output generation. |
| `src/live_marketing/service.py` | Live transaction orchestration: save, feature, segment, recommend. |
| `src/live_marketing/transaction_manager.py` | Transaction validation, duplicate detection, append, and loading. |
| `tests/test_live_marketing.py` | Existing marketing-engine unit tests. |
| `scripts/validate_phase2_reconstruction.py` | Phase 2 reconstruction and artifact-validation script. |
| `requirements.txt` | Core project Python dependencies. |
| `dashboard/requirements.txt` | Streamlit dashboard dependencies. |
| `customer_segmentation_dashboard_starter/dashboard/requirements.txt` | Starter dashboard dependencies. |
| `README.md` | Primary project documentation. |
| `dashboard/README.md` | Streamlit dashboard run and usage documentation. |
| `customer_segmentation_dashboard_starter/dashboard/README.md` | Starter dashboard documentation. |
| `data/processed/clean_transactions.csv` | Original processed transaction data required by live feature generation. |
| `data/live/transactions.csv` | Existing separately stored live transaction records. |
| `data/processed/customer_features.csv` | Phase 1 customer feature reference used by reconstruction validation. |
| `results/phase2/phase2_customer_segments.csv` | Saved Phase 2 customer assignments consumed by the dashboard. |
| `results/phase2/tables/phase2_model_comparison.csv` | Phase 2 model-comparison output. |
| `results/phase2/tables/phase2_qlde_kmeans_results.csv` | Phase 2 QLDE + K-Means results. |
| `results/phase2/tables/phase2_qlde_results.csv` | Phase 2 QLDE results. |
| `results/phase2/tables/phase2_segment_definitions.csv` | Phase 2 segment definitions. |
| `results/phase2/tables/phase2_marketing_strategy.csv` | Phase 2 marketing strategy definitions. |
| `models/live_marketing/phase2_scaler.joblib` | Phase 2 reconstructed scaler used for prediction. |
| `models/live_marketing/phase2_pca.joblib` | Phase 2 reconstructed PCA used for prediction. |
| `models/live_marketing/phase2_kmeans.joblib` | Phase 2 reconstructed K-Means used for prediction. |
| `models/live_marketing/phase2_feature_columns.joblib` | Phase 2 model input feature-column list. |

## Dependencies Between Dashboards and Modules

### `dashboard/app.py`

- Internal imports: none detected.
- External imports: `numpy`, `pandas`, `pathlib`, `plotly`, `streamlit`

### `dashboard/live_marketing.py`

- Internal imports: `src.live_marketing.config`, `src.live_marketing.service`
- External imports: `pandas`, `pathlib`, `streamlit`, `sys`

### `dashboard/app_backup.py`

- Internal imports: none detected.
- External imports: `numpy`, `pandas`, `pathlib`, `plotly`, `streamlit`

### `customer_segmentation_dashboard_starter/dashboard/app.py`

- Internal imports: none detected.
- External imports: `numpy`, `pandas`, `pathlib`, `plotly`, `streamlit`

- `dashboard/live_marketing.py` imports `src.live_marketing.config` and `src.live_marketing.service`.
- `service.py` imports `transaction_manager.py`, `feature_engine.py`, `customer_segmenter.py`, and `marketing_engine.py`.
- `feature_engine.py` imports `config.py` and `transaction_manager.py`; it reads the cleaned transaction CSV and live CSV.
- `customer_segmenter.py` imports `config.py`, loads existing Phase 2 assignments, and loads the four Joblib artifacts.
- `marketing_engine.py` operates on segmented customer data and returns recommendation fields.
- `transaction_manager.py` imports `config.py`, validates the transaction schema, and writes `data/live/transactions.csv`.
- `dashboard/app.py`, `dashboard/app_backup.py`, and the starter dashboard do not import project Python modules; they read saved CSV outputs and optional uploads.

## Files Excluded

- `.venv/`, `.venv-1/`, `.venv-project/`: virtual-environment and third-party package files.
- `.git/`: repository metadata.
- `__pycache__/`, `.pytest_cache/`, `.ipynb_checkpoints/`, `.pyc`, `.pyo`: caches and temporary compiled files.
- `data/raw/data.csv`: raw source dataset; not imported directly by the selected dashboards.
- Large notebook files and unrelated research notebook outputs: not direct dashboard dependencies.
- Phase 1 full experiment tables, figures, and convergence/history outputs: not direct dashboard dependencies; the selected dashboards use a smaller Phase 2/dashboard-specific set.
- Other virtual-environment copies and generated test/cache artifacts.

## Missing or Runtime-Uncertain Dependencies

- No selected file was missing at packaging time.
- The live dashboard requires `data/processed/clean_transactions.csv`, `data/live/transactions.csv`, the existing segments CSV, and the four Phase 2 Joblib artifacts; these are included.
- The saved dashboard accepts CSV uploads, but at runtime it also expects `results/phase2/phase2_customer_segments.csv` and its optional model/profile CSVs; these are included where present.

## Packaging Verification

- Existing project source files were not modified or executed.
- The bundle and report were created only as review artifacts.
- Relative project paths are preserved in the ZIP.
