# Testing

Run from the project root:

```powershell
python -m compileall dashboard src tests scripts
python -m pytest -q
streamlit run dashboard/app.py
```

## Current automated checks

The supplied test suite contains three marketing-engine unit tests:
- VIP/high-value recommendation.
- At-risk customer recommendation.
- Default new-customer recommendation.

## Manual dashboard smoke test

1. Confirm the Overview page loads and KPI counts are populated.
2. Change segment and priority filters; confirm charts and tables update.
3. Open each of the eight sidebar sections.
4. Search for a known CustomerID in Customer Explorer.
5. Download the filtered customer CSV.
6. Check Model Evaluation and Reports & Downloads.
7. In Live Transactions, submit a clearly identified test transaction only if you intend to write to `data/live/transactions.csv`.
8. Confirm the transaction appears in the live table and that a recommendation is displayed.

The live transaction check writes to the project's live CSV. Use a test invoice/customer and remove test data only after reviewing it; do not use historical source data for testing.
