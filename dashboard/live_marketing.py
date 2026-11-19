import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make the project root importable when Streamlit runs this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.live_marketing.config import LIVE_TRANSACTIONS_PATH
from src.live_marketing.service import (
    get_current_customer_data,
    process_transaction,
)


def render_live_marketing():
    """Render the independent live transaction and marketing interface."""

    st.title("Live Transactions & Personalized Marketing")
    st.caption(
        "Independent module: new transactions are stored separately "
        "from the original dataset."
    )

    with st.form("live_transaction_form", clear_on_submit=False):
        st.subheader("Enter a transaction")

        col1, col2 = st.columns(2)

        with col1:
            invoice_no = st.text_input("Invoice number", value="LIVE-1001")
            stock_code = st.text_input("Product code", value="PROD-001")
            description = st.text_input("Product description")
            quantity = st.number_input(
                "Quantity", min_value=1, value=1, step=1
            )

        with col2:
            invoice_date = st.date_input("Transaction date")
            unit_price = st.number_input(
                "Unit price", min_value=0.01, value=100.0, step=1.0
            )
            customer_id = st.number_input(
                "Customer ID", min_value=1, value=900001, step=1
            )
            country = st.text_input("Country", value="India")

        submitted = st.form_submit_button("Save & update recommendations")

    if submitted:
        transaction = {
            "InvoiceNo": invoice_no,
            "StockCode": stock_code,
            "Description": description or stock_code,
            "Quantity": quantity,
            "InvoiceDate": pd.Timestamp(invoice_date),
            "UnitPrice": unit_price,
            "CustomerID": customer_id,
            "Country": country,
        }

        try:
            with st.spinner(
                "Saving transaction and recalculating customer results..."
            ):
                result = process_transaction(transaction)

            st.session_state["live_last_customer"] = result["customer"]
            st.session_state["live_refresh"] = (
                st.session_state.get("live_refresh", 0) + 1
            )

            st.success("Transaction saved and customer results updated.")

        except Exception as exc:
            st.error(f"Could not process transaction: {exc}")

    st.divider()
    st.subheader("Latest customer recommendation")

    last_customer = st.session_state.get("live_last_customer")

    if last_customer:
        st.write(f"**Customer ID:** {last_customer.get('CustomerID')}")
        st.write(f"**Segment:** {last_customer.get('Cluster')}")
        st.write(f"**Campaign:** {last_customer.get('CampaignType')}")
        st.write(
            f"**Priority:** {last_customer.get('MarketingPriority')}"
        )
        st.write(
            f"**Recommended action:** "
            f"{last_customer.get('RecommendedAction')}"
        )
        st.info(last_customer.get("MarketingMessage", ""))

    st.divider()
    st.subheader("Current customer marketing table")

    if st.button("Refresh customer results"):
        st.rerun()

    try:
        # Recompute only when this component is rendered/refreshed.
        customers = get_current_customer_data()

        display_columns = [
            column
            for column in [
                "CustomerID",
                "Cluster",
                "Recency",
                "Frequency",
                "Monetary",
                "CampaignType",
                "MarketingPriority",
                "RecommendedAction",
                "MarketingMessage",
                "AssignmentMethod",
            ]
            if column in customers.columns
        ]

        st.dataframe(
            customers[display_columns],
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download current marketing recommendations",
            customers[display_columns].to_csv(index=False).encode("utf-8"),
            file_name="live_marketing_recommendations.csv",
            mime="text/csv",
        )

    except Exception as exc:
        st.error(f"Could not load customer results: {exc}")

    if LIVE_TRANSACTIONS_PATH.exists():
        st.caption(
            f"Live transaction storage: {LIVE_TRANSACTIONS_PATH}"
        )


if __name__ == "__main__":
    render_live_marketing()