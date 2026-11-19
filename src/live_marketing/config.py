from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
LIVE_DATA_DIR = DATA_DIR / "live"

# Original data — read-only for the live module
ORIGINAL_TRANSACTIONS_PATH = (
    DATA_DIR / "processed" / "clean_transactions.csv"
)

# Separate storage for newly entered transactions
LIVE_TRANSACTIONS_PATH = LIVE_DATA_DIR / "transactions.csv"

# Existing Phase 2 segmentation output
EXISTING_SEGMENTS_PATH = (
    PROJECT_ROOT / "results" / "phase2" / "phase2_customer_segments.csv"
)

# Customer feature definitions
CUSTOMER_ID_COLUMN = "CustomerID"

FEATURE_COLUMNS = [
    "Recency",
    "Frequency",
    "Monetary",
    "CustomerLifespanDays",
    "AveragePurchaseIntervalDays",
    "TotalQuantity",
    "ProductDiversity",
    "AverageBasketQuantity",
    "AverageTransactionValue",
    "SpendingTrend",
    "FrequencyTrend",
    "QuantityTrend",
    "NewCustomer",
]

# Live module settings
MAX_TRANSACTION_QUANTITY = 100000
MAX_UNIT_PRICE = 1000000