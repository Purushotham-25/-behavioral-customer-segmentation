# Project Structure

```text
Behavioral-customer-segmentation/
├── dashboard/
│   ├── __init__.py
│   ├── app.py                 # Unified eight-section Streamlit entry point
│   ├── live_marketing.py      # Existing live transaction and recommendation UI
│   └── app_backup.py          # Original backup; retained unchanged
├── src/
│   ├── segmentation_pipeline.py  # Existing QLDE + K-Means research pipeline
│   └── live_marketing/
│       ├── config.py
│       ├── feature_engine.py
│       ├── customer_segmenter.py
│       ├── marketing_engine.py
│       ├── service.py
│       └── transaction_manager.py
├── data/
│   ├── processed/             # Historical, read-only inputs and customer features
│   └── live/                  # Separate live transaction storage
├── models/live_marketing/     # Existing Phase 2 model artifacts
├── results/phase2/            # Saved segmentation and evaluation outputs
├── tests/                     # Automated tests
├── scripts/                   # Validation utilities
├── requirements.txt
├── README.md
├── DASHBOARD_GUIDE.md
├── TESTING.md
└── PROJECT_STRUCTURE.md
```

The dashboard is a presentation layer. It reads saved Phase 2 outputs and calls the existing live-marketing service. It does not retrain or alter the QLDE + K-Means research algorithm.
