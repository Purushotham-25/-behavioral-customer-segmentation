import numpy as np
import pandas as pd

from .config import ORIGINAL_TRANSACTIONS_PATH, FEATURE_COLUMNS
from .transaction_manager import load_live_transactions


PHASE2_SPLIT_DATE = pd.Timestamp("2011-07-01")
HISTORICAL_REFERENCE_DATE = pd.Timestamp("2011-12-10 12:50:00")
LIVE_DATE_THRESHOLD = pd.Timestamp("2026-01-01")
HISTORICAL_MONTHS = 7
RECENT_MONTHS = 6

PHASE2_FEATURE_COLUMNS = [
    "RecentSpendingShare",
    "ActivityConsistency",
]


def load_all_transactions() -> pd.DataFrame:
    """Combine historical and live transactions without modifying either file."""

    if not ORIGINAL_TRANSACTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Original transaction file not found: {ORIGINAL_TRANSACTIONS_PATH}"
        )

    original = pd.read_csv(
        ORIGINAL_TRANSACTIONS_PATH,
        encoding="ISO-8859-1",
        parse_dates=["InvoiceDate"],
    )
    live = load_live_transactions()

    transactions = pd.concat([original, live], ignore_index=True)

    for column in ["Quantity", "UnitPrice", "CustomerID"]:
        transactions[column] = pd.to_numeric(
            transactions[column], errors="coerce"
        )

    transactions["InvoiceDate"] = pd.to_datetime(
        transactions["InvoiceDate"], errors="coerce"
    )

    transactions = transactions.dropna(
        subset=[
            "InvoiceNo",
            "InvoiceDate",
            "CustomerID",
            "Quantity",
            "UnitPrice",
        ]
    )

    # Match Phase 2: exclude cancelled invoices.
    transactions = transactions[
        ~transactions["InvoiceNo"].astype(str).str.startswith("C")
    ].copy()

    return transactions


def build_phase2_behavior_features(
    transactions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Reproduce Phase 2 historical/recent behavioral features."""

    columns = ["CustomerID", *PHASE2_FEATURE_COLUMNS]

    if transactions is None:
        transactions = load_all_transactions()

    if transactions.empty:
        return pd.DataFrame(columns=columns)

    df = transactions.copy()

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], errors="coerce"
    )
    for column in ["Quantity", "UnitPrice", "CustomerID"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(
        subset=[
            "CustomerID",
            "InvoiceNo",
            "InvoiceDate",
            "Quantity",
            "UnitPrice",
        ]
    )

    df = df[
        (df["Quantity"] > 0)
        & (df["UnitPrice"] > 0)
        & (~df["InvoiceNo"].astype(str).str.startswith("C"))
    ].copy()

    if df.empty:
        return pd.DataFrame(columns=columns)

    df["TransactionValue"] = df["Quantity"] * df["UnitPrice"]

    historical = df[df["InvoiceDate"] < PHASE2_SPLIT_DATE]
    recent = df[df["InvoiceDate"] >= PHASE2_SPLIT_DATE]

    def period_features(
        period: pd.DataFrame,
        period_name: str,
        months: int,
    ) -> pd.DataFrame:
        result = (
            period.groupby("CustomerID")
            .agg(
                Spending=("TransactionValue", "sum"),
                Frequency=("InvoiceNo", "nunique"),
                Quantity=("Quantity", "sum"),
            )
            .reset_index()
        )

        result[f"{period_name}MonthlySpending"] = result["Spending"] / months
        result[f"{period_name}MonthlyFrequency"] = result["Frequency"] / months
        result[f"{period_name}MonthlyQuantity"] = result["Quantity"] / months

        return result[
            [
                "CustomerID",
                f"{period_name}MonthlySpending",
                f"{period_name}MonthlyFrequency",
                f"{period_name}MonthlyQuantity",
            ]
        ]

    historical_features = period_features(
        historical, "Historical", HISTORICAL_MONTHS
    )
    recent_features = period_features(
        recent, "Recent", RECENT_MONTHS
    )

    result = historical_features.merge(
        recent_features,
        on="CustomerID",
        how="outer",
    ).fillna(0)

    historical_spending = result["HistoricalMonthlySpending"]
    recent_spending = result["RecentMonthlySpending"]
    spending_total = historical_spending + recent_spending

    result["RecentSpendingShare"] = np.divide(
        recent_spending,
        spending_total,
        out=np.zeros(len(result), dtype=float),
        where=spending_total.to_numpy() > 0,
    )

    historical_frequency = result["HistoricalMonthlyFrequency"]
    recent_frequency = result["RecentMonthlyFrequency"]

    frequency_max = pd.concat(
        [historical_frequency, recent_frequency], axis=1
    ).max(axis=1)
    frequency_min = pd.concat(
        [historical_frequency, recent_frequency], axis=1
    ).min(axis=1)

    result["ActivityConsistency"] = np.divide(
        frequency_min,
        frequency_max,
        out=np.zeros(len(result), dtype=float),
        where=frequency_max.to_numpy() > 0,
    )

    return result[columns]


def build_customer_features(
    transactions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Calculate customer features and Phase 2 behavioral features."""

    if transactions is None:
        transactions = load_all_transactions()

    output_columns = [
        "CustomerID",
        *FEATURE_COLUMNS,
        *PHASE2_FEATURE_COLUMNS,
    ]

    if transactions.empty:
        return pd.DataFrame(columns=output_columns)

    df = transactions.copy()

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], errors="coerce"
    )
    for column in ["Quantity", "UnitPrice", "CustomerID"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(
        subset=[
            "CustomerID",
            "InvoiceDate",
            "Quantity",
            "UnitPrice",
            "InvoiceNo",
        ]
    )

    df = df[
        (df["Quantity"] > 0)
        & (df["UnitPrice"] > 0)
        & (~df["InvoiceNo"].astype(str).str.startswith("C"))
    ].copy()

    if df.empty:
        return pd.DataFrame(columns=output_columns)

    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    live_reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    records = []

    for customer_id, group in df.groupby("CustomerID"):
        latest_date = group["InvoiceDate"].max()

        # Historical-only customers use the original dataset's reference date.
        # Customers with recent live-era activity use the latest combined date.
        reference_date = (
            live_reference_date
            if latest_date >= LIVE_DATE_THRESHOLD
            else HISTORICAL_REFERENCE_DATE
        )

        dates = (
            group.groupby("InvoiceNo")["InvoiceDate"]
            .min()
            .sort_values()
        )

        monetary = group["Revenue"].sum()
        frequency = group["InvoiceNo"].nunique()

        lifespan = max(0, (latest_date - group["InvoiceDate"].min()).days)

        intervals = dates.diff().dt.total_seconds().dropna() / 86400
        average_interval = (
            float(intervals.mean()) if not intervals.empty else 0.0
        )

        ordered = group.sort_values("InvoiceDate")
        midpoint = len(ordered) // 2
        first_half = ordered.iloc[:midpoint]
        second_half = ordered.iloc[midpoint:]

        first_spend = (
            first_half["Quantity"] * first_half["UnitPrice"]
        ).sum()
        second_spend = (
            second_half["Quantity"] * second_half["UnitPrice"]
        ).sum()

        first_frequency = first_half["InvoiceNo"].nunique()
        second_frequency = second_half["InvoiceNo"].nunique()
        first_quantity = first_half["Quantity"].sum()
        second_quantity = second_half["Quantity"].sum()

        def trend(first, second):
            denominator = abs(first)
            if denominator == 0:
                return 0.0 if second == 0 else 1.0
            return float(np.clip((second - first) / denominator, -1, 1))

        records.append(
            {
                # Required for merging Phase 2 behavior features.
                "CustomerID": customer_id,
                "Recency": max(0, (reference_date - latest_date).days),
                "Frequency": int(frequency),
                "Monetary": float(monetary),
                "CustomerLifespanDays": int(lifespan),
                "AveragePurchaseIntervalDays": average_interval,
                "TotalQuantity": float(group["Quantity"].sum()),
                "ProductDiversity": int(group["StockCode"].nunique()),
                "AverageBasketQuantity": (
                    float(group["Quantity"].sum() / frequency)
                    if frequency else 0.0
                ),
                "AverageTransactionValue": (
                    float(monetary / frequency) if frequency else 0.0
                ),
                "SpendingTrend": trend(first_spend, second_spend),
                "FrequencyTrend": trend(first_frequency, second_frequency),
                "QuantityTrend": trend(first_quantity, second_quantity),
                "NewCustomer": int(frequency == 1),
            }
        )

    features = pd.DataFrame(records)

    phase2_features = build_phase2_behavior_features(df)

    features = features.merge(
        phase2_features,
        on="CustomerID",
        how="left",
        validate="one_to_one",
    )

    return features[output_columns]