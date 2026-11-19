# Unified Dashboard Guide

## Launch

From the project root, activate your project virtual environment and run:

```powershell
python -m pip install -r requirements.txt
streamlit run dashboard/app.py
```

## Navigation

The sidebar contains eight workspaces:

1. **Overview** — customer and segment KPIs, segment mix and value distribution.
2. **Customer Segmentation** — segment population and behavioural feature comparisons.
3. **Segment Profiles** — segment-level averages, median feature charts and definitions.
4. **Customer Explorer** — customer ID search, campaign filtering, customer detail and CSV export.
5. **Marketing Intelligence** — campaign/priority summaries by segment.
6. **Live Transactions** — existing transaction form, live service and recommendation table.
7. **Model Evaluation** — saved clustering metrics and model comparison.
8. **Reports & Downloads** — filtered customer data, evaluation tables, strategy reference and live transaction exports.

## Data behaviour

- The dashboard reads the saved Phase 2 customer segmentation output under `results/phase2/`.
- Optional model and strategy tables are loaded from `results/phase2/tables/`.
- New transactions continue to be stored separately under `data/live/`.
- The original historical transaction dataset remains read-only.
- The dashboard visualizes saved research outputs; it does not run model training.

## Interactivity

Plotly charts include hover tooltips, zoom, pan, selection and image export controls. Tables use Streamlit's interactive dataframe display. Sidebar segment and priority filters apply to the analytical sections.
