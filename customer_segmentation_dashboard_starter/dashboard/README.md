# Customer Segmentation Dashboard

Streamlit dashboard for the existing Phase 2 customer segmentation CSV outputs.

## Expected files
- `results/phase2/phase2_customer_segments.csv`
- `results/phase2/tables/phase2_model_comparison.csv`
- Optional: `results/phase2/tables/phase2_segment_definitions.csv`
- Optional: `results/phase2/tables/phase2_marketing_strategy.csv`

The app searches for the expected paths. If your outputs are elsewhere, upload the CSV files using the sidebar.

## Run in VS Code on Windows
From the root of `Behavioral-customer-segmentation`:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r .\dashboard\requirements.txt
streamlit run .\dashboard\app.py
```

If PowerShell blocks activation:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\dashboard\requirements.txt
.\.venv\Scripts\python.exe -m streamlit run .\dashboard\app.py
```

Open the local URL Streamlit prints, usually `http://localhost:8501`.

## Dashboard tabs
Overview, Segment profiles, Marketing actions, Model evaluation, and Customer explorer with filtered CSV export.

This dashboard reads saved outputs; it does not rerun or retrain QLDE/K-Means.
