import pandas as pd

from .transaction_manager import add_transaction
from .feature_engine import build_customer_features
from .customer_segmenter import assign_customer_segments
from .marketing_engine import generate_marketing_recommendations


def process_transaction(transaction: dict) -> dict:
    """
    Save one transaction and regenerate customer features,
    segment assignments, and personalized marketing recommendations.
    """

    # 1. Validate and save the transaction separately.
    saved_transaction = add_transaction(transaction)

    # 2. Recalculate customer features from original + live data.
    features = build_customer_features()

    # 3. Attach existing or provisional segment assignments.
    segmented = assign_customer_segments(features)

    # 4. Generate personalized marketing recommendations.
    recommendations = generate_marketing_recommendations(segmented)

    # 5. Return the updated record for the affected customer.
    customer_id = float(saved_transaction["CustomerID"])
    customer_result = recommendations[
        pd.to_numeric(recommendations["CustomerID"], errors="coerce")
        == customer_id
    ]

    if customer_result.empty:
        raise RuntimeError(
            "The transaction was saved, but no updated customer result was produced."
        )

    return {
        "transaction": saved_transaction,
        "customer": customer_result.iloc[-1].to_dict(),
        "customers": recommendations,
    }


def get_current_customer_data() -> pd.DataFrame:
    """Rebuild and return the current customer and marketing data."""

    features = build_customer_features()
    segmented = assign_customer_segments(features)
    return generate_marketing_recommendations(segmented)