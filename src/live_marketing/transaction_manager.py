import pandas as pd

from .config import (
    LIVE_DATA_DIR,
    LIVE_TRANSACTIONS_PATH,
    MAX_TRANSACTION_QUANTITY,
    MAX_UNIT_PRICE,
)

TRANSACTION_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]


def validate_transaction(transaction: dict) -> dict:
    """Validate a single transaction before saving it."""

    missing = [
        column
        for column in TRANSACTION_COLUMNS
        if column not in transaction
    ]
    if missing:
        raise ValueError(f"Missing fields: {', '.join(missing)}")

    row = {column: transaction[column] for column in TRANSACTION_COLUMNS}

    if pd.isna(row["InvoiceNo"]) or not str(row["InvoiceNo"]).strip():
        raise ValueError("InvoiceNo is required.")

    if pd.isna(row["StockCode"]) or not str(row["StockCode"]).strip():
        raise ValueError("StockCode is required.")

    if pd.isna(row["CustomerID"]):
        raise ValueError("CustomerID is required.")

    row["Quantity"] = int(row["Quantity"])
    row["UnitPrice"] = float(row["UnitPrice"])
    row["CustomerID"] = float(row["CustomerID"])

    if row["Quantity"] <= 0:
        raise ValueError("Quantity must be greater than zero.")

    if row["Quantity"] > MAX_TRANSACTION_QUANTITY:
        raise ValueError("Quantity exceeds the configured maximum.")

    if not 0 < row["UnitPrice"] <= MAX_UNIT_PRICE:
        raise ValueError("UnitPrice must be positive and within the configured maximum.")

    row["InvoiceDate"] = pd.to_datetime(
        row["InvoiceDate"], errors="raise"
    )

    for field in ["InvoiceNo", "StockCode", "Description", "Country"]:
        row[field] = str(row[field]).strip()

    return row


def add_transaction(transaction: dict) -> dict:
    """Validate and append a transaction to separate live storage."""

    row = validate_transaction(transaction)

    LIVE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if LIVE_TRANSACTIONS_PATH.exists():
        existing = pd.read_csv(LIVE_TRANSACTIONS_PATH)

        # Prevent accidental duplicate submissions.
        same_transaction = (
            existing["InvoiceNo"].astype(str).eq(str(row["InvoiceNo"]))
            & existing["StockCode"].astype(str).eq(str(row["StockCode"]))
            & pd.to_numeric(existing["CustomerID"], errors="coerce").eq(
                row["CustomerID"]
            )
            & pd.to_datetime(existing["InvoiceDate"], errors="coerce").eq(
                row["InvoiceDate"]
            )
        )

        if same_transaction.any():
            raise ValueError(
                "This transaction appears to have already been submitted."
            )

        updated = pd.concat(
            [existing, pd.DataFrame([row])],
            ignore_index=True,
        )
    else:
        updated = pd.DataFrame([row], columns=TRANSACTION_COLUMNS)

    updated.to_csv(LIVE_TRANSACTIONS_PATH, index=False)

    return row


def load_live_transactions() -> pd.DataFrame:
    """Load separately stored live transactions."""

    if not LIVE_TRANSACTIONS_PATH.exists():
        return pd.DataFrame(columns=TRANSACTION_COLUMNS)

    return pd.read_csv(
        LIVE_TRANSACTIONS_PATH,
        parse_dates=["InvoiceDate"],
    )