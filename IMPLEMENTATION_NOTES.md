# Implementation Notes

## Completed in this working copy

- Combined the saved segmentation dashboard and live-marketing interface behind one sidebar navigation.
- Added eight approved dashboard workspaces.
- Added consistent page styling, KPI cards, Plotly interaction controls, segment/priority filters, customer search, campaign filtering, customer detail view, and CSV exports.
- Reused the existing live-marketing renderer and service rather than duplicating transaction/recommendation logic.
- Changed the live dashboard refresh button to trigger a Streamlit rerun so the customer table is recalculated on refresh.
- Added project structure, dashboard usage, and testing documentation.
- Updated root requirements to include Streamlit and Plotly.

## Preserved

- `src/segmentation_pipeline.py` and the QLDE + K-Means algorithm.
- Existing Phase 2 model artifacts and saved research results.
- Original historical transaction data.
- Separate live transaction storage.
- Original dashboard backup and starter copy.

## Validation performed here

- Python compile check: passed.
- Existing pytest suite: 3 passed.
- Full interactive Streamlit runtime test: not performed in this environment because Streamlit is not installed here. Install dependencies and perform the manual smoke test described in `TESTING.md` on the target machine.

## Important live-data note

The live transaction workflow writes to `data/live/transactions.csv`. Test with an explicitly identified test invoice/customer. This package does not execute or insert any transactions automatically.
