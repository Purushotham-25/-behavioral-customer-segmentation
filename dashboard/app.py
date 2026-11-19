"""Unified Streamlit application for behavioral customer segmentation and live marketing.

The dashboard is a presentation layer over the existing Phase 2 outputs and live
marketing services. It deliberately does not retrain or modify the QLDE + K-Means
research implementation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Resolve the project root so this app works when launched from any directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.live_marketing.config import LIVE_TRANSACTIONS_PATH
from dashboard.live_marketing import render_live_marketing

st.set_page_config(
    page_title="Behavioral Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Shared visual system keeps all dashboard sections consistent.
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.5rem; padding-bottom: 2.5rem;}
      [data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.18);}
      .hero {padding: 1.15rem 1.35rem; border-radius: 14px;
             background: linear-gradient(115deg, rgba(35,80,135,.16), rgba(35,160,145,.10));
             border: 1px solid rgba(128,128,128,.18); margin-bottom: 1rem;}
      .hero h1 {margin: 0 0 .25rem 0; font-size: 1.8rem;}
      .hero p {margin: 0; opacity: .78;}
      div[data-testid="stMetric"] {background: rgba(128,128,128,.07);
        border: 1px solid rgba(128,128,128,.15); padding: .8rem 1rem; border-radius: 12px;}
      div[data-testid="stDataFrame"] {border-radius: 10px; overflow: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Search known locations relative to the project rather than relying on cwd.
SEGMENT_PATHS = [
    PROJECT_ROOT / "results/phase2/phase2_customer_segments.csv",
    PROJECT_ROOT / "results/phase2_customer_segments.csv",
]
MODEL_PATHS = [
    PROJECT_ROOT / "results/phase2/tables/phase2_model_comparison.csv",
    PROJECT_ROOT / "results/phase2/tables/phase2_qlde_kmeans_results.csv",
    PROJECT_ROOT / "results/phase2/tables/phase2_qlde_results.csv",
]
PROFILE_PATHS = [
    PROJECT_ROOT / "results/phase2/tables/phase2_segment_definitions.csv",
    PROJECT_ROOT / "results/phase2/tables/phase2_marketing_strategy.csv",
]


def first_existing(paths: list[Path]) -> Path | None:
    """Return the first existing data file from an ordered list of candidates."""
    return next((path for path in paths if path.is_file()), None)


@st.cache_data(show_spinner=False)
def read_csv(path_string: str) -> pd.DataFrame:
    """Cache CSV reads; Streamlit invalidates the cache when the file changes."""
    return pd.read_csv(path_string)


def load_optional_csv(paths: list[Path]) -> tuple[pd.DataFrame | None, str]:
    """Load the first available optional CSV and return its source description."""
    path = first_existing(paths)
    if path is None:
        return None, "Not found"
    try:
        return read_csv(str(path)), str(path.relative_to(PROJECT_ROOT))
    except Exception as exc:  # Surface the failure in the relevant section.
        return None, f"Could not read {path.name}: {exc}"


def find_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find the first matching column name from known schema variations."""
    return next((name for name in candidates if name in frame.columns), None)


def chart_config() -> dict:
    """Use consistent interactive Plotly controls across all charts."""
    return {"displayModeBar": True, "displaylogo": False, "scrollZoom": True}


segments, segment_source = load_optional_csv(SEGMENT_PATHS)
models, model_source = load_optional_csv(MODEL_PATHS)
definitions, definitions_source = load_optional_csv(PROFILE_PATHS)

with st.sidebar:
    st.markdown("## 📊 Customer Intelligence")
    st.caption("Behavioral segmentation · Personalized marketing")
    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "Customer Segmentation",
            "Segment Profiles",
            "Customer Explorer",
            "Marketing Intelligence",
            "Live Transactions",
            "Model Evaluation",
            "Reports & Downloads",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Data and model controls")
    if st.button("↻ Refresh dashboard data", use_container_width=True):
        read_csv.clear()
        st.rerun()
    st.caption("Research model: existing QLDE + K-Means outputs")
    st.caption("Historical transaction data is treated as read-only.")

st.markdown(
    '<div class="hero"><h1>Behavioral Customer Intelligence</h1>'
    '<p>Understand customer behaviour, explore segments and activate personalized marketing.</p></div>',
    unsafe_allow_html=True,
)

if page == "Live Transactions":
    # Reuse the existing live service and transaction form without duplicating logic.
    render_live_marketing()

elif segments is None or segments.empty:
    st.error("Customer segment data is unavailable.")
    st.info("Expected: results/phase2/phase2_customer_segments.csv")
    st.caption(f"Lookup result: {segment_source}")
    st.stop()

else:
    cluster_col = find_column(segments, ["Cluster", "cluster", "Segment", "segment"])
    customer_col = find_column(segments, ["CustomerID", "customer_id", "Customer Id"])
    action_col = find_column(segments, ["Recommendation", "MarketingObjective", "CampaignType", "recommendation"])
    campaign_col = find_column(segments, ["CampaignType", "MarketingObjective"])
    priority_col = find_column(segments, ["Priority", "priority"])
    if cluster_col is None:
        st.error(f"No cluster column found. Available columns: {', '.join(segments.columns)}")
        st.stop()

    data_all = segments.copy()
    data_all[cluster_col] = data_all[cluster_col].astype(str)
    with st.sidebar:
        st.subheader("Global filters")
        clusters = sorted(data_all[cluster_col].dropna().unique().tolist())
        selected_clusters = st.multiselect("Customer segments", clusters, default=clusters)
        filtered = data_all[data_all[cluster_col].isin(selected_clusters)].copy()
        if priority_col:
            priorities = sorted(filtered[priority_col].dropna().astype(str).unique())
            selected_priorities = st.multiselect("Marketing priority", priorities, default=priorities)
            filtered = filtered[filtered[priority_col].astype(str).isin(selected_priorities)]

    numeric_cols = [
        c for c in filtered.select_dtypes(include=np.number).columns
        if c.lower() not in {"customerid", "customer_id"}
    ]
    customer_count = len(filtered)
    segment_count = filtered[cluster_col].nunique()
    revenue_col = find_column(filtered, ["Monetary", "Revenue", "TotalSpend"])
    frequency_col = find_column(filtered, ["Frequency", "Orders"])
    high_priority_count = int((filtered[priority_col].astype(str).str.lower() == "high").sum()) if priority_col else 0

    if page == "Overview":
        st.subheader("Business overview")
        kpis = st.columns(4)
        kpis[0].metric("Customers in view", f"{customer_count:,}")
        kpis[1].metric("Active segments", f"{segment_count:,}")
        kpis[2].metric("High-priority customers", f"{high_priority_count:,}")
        if revenue_col:
            kpis[3].metric("Average customer value", f"{filtered[revenue_col].mean():,.2f}")
        else:
            kpis[3].metric("Available features", f"{len(numeric_cols):,}")
        left, right = st.columns(2)
        distribution = filtered[cluster_col].value_counts().rename_axis("Segment").reset_index(name="Customers")
        distribution["Share (%)"] = distribution["Customers"] / max(customer_count, 1) * 100
        with left:
            st.plotly_chart(px.bar(distribution, x="Segment", y="Customers", text="Customers",
                                   title="Customer count by segment"), use_container_width=True,
                             config=chart_config())
        with right:
            st.plotly_chart(px.pie(distribution, names="Segment", values="Customers", hole=.55,
                                   title="Customer segment mix"), use_container_width=True,
                             config=chart_config())
        if revenue_col:
            st.plotly_chart(px.box(filtered, x=cluster_col, y=revenue_col, color=cluster_col,
                                   points="outliers", title="Customer value distribution by segment"),
                            use_container_width=True, config=chart_config())
        st.caption(f"Source: {segment_source}")

    elif page == "Customer Segmentation":
        st.subheader("Customer segmentation")
        st.write("Explore segment size and compare the behavioural variables used in the customer profiles.")
        distribution = filtered[cluster_col].value_counts().rename_axis("Segment").reset_index(name="Customers")
        st.plotly_chart(px.bar(distribution, x="Segment", y="Customers", color="Segment",
                               title="Segment population"), use_container_width=True,
                         config=chart_config())
        if numeric_cols:
            defaults = [c for c in numeric_cols if any(k in c.lower() for k in
                        ["recency", "frequency", "monetary", "spend", "quantity"])]
            chosen_features = st.multiselect("Compare behavioural features", numeric_cols,
                                             default=(defaults[:4] or numeric_cols[:min(4, len(numeric_cols))]))
            if chosen_features:
                summary = filtered.groupby(cluster_col)[chosen_features].mean().reset_index()
                long = summary.melt(id_vars=cluster_col, var_name="Feature", value_name="Mean value")
                st.plotly_chart(px.bar(long, x="Feature", y="Mean value", color=cluster_col,
                                       barmode="group", title="Average behavioural features by segment"),
                                use_container_width=True, config=chart_config())
        st.dataframe(distribution, use_container_width=True, hide_index=True)

    elif page == "Segment Profiles":
        st.subheader("Segment profiles")
        rows = []
        for label, group in filtered.groupby(cluster_col):
            row = {"Segment": label, "Customers": len(group),
                   "Share (%)": round(len(group) / max(len(filtered), 1) * 100, 2)}
            for feature in numeric_cols:
                row[f"Average {feature}"] = group[feature].mean()
            if campaign_col and group[campaign_col].notna().any():
                row["Campaign"] = ", ".join(group[campaign_col].dropna().astype(str).unique()[:3])
            rows.append(row)
        profile_table = pd.DataFrame(rows)
        st.dataframe(profile_table, use_container_width=True, hide_index=True)
        if numeric_cols:
            selected_feature = st.selectbox("Profile metric", numeric_cols)
            medians = filtered.groupby(cluster_col)[selected_feature].median().reset_index()
            st.plotly_chart(px.bar(medians, x=cluster_col, y=selected_feature, color=cluster_col,
                                   title=f"Median {selected_feature} by segment"),
                            use_container_width=True, config=chart_config())
        if definitions is not None:
            st.subheader("Existing segment definitions and marketing strategy")
            st.dataframe(definitions, use_container_width=True, hide_index=True)
            st.caption(f"Reference: {definitions_source}")

    elif page == "Customer Explorer":
        st.subheader("Customer explorer")
        view = filtered.copy()
        search = st.text_input("Search by customer ID", placeholder="Enter all or part of a customer ID")
        if customer_col and search.strip():
            view = view[view[customer_col].astype(str).str.contains(search.strip(), case=False, na=False)]
        if campaign_col:
            campaign_values = sorted(view[campaign_col].dropna().astype(str).unique())
            chosen_campaigns = st.multiselect("Campaign", campaign_values, default=campaign_values)
            view = view[view[campaign_col].astype(str).isin(chosen_campaigns)]
        st.caption(f"{len(view):,} customers match the current filters.")
        st.dataframe(view, use_container_width=True, hide_index=True)
        if customer_col and not view.empty:
            chosen_id = st.selectbox("Open customer detail", view[customer_col].astype(str).unique())
            record = view[view[customer_col].astype(str) == chosen_id].iloc[0]
            st.json(record.where(pd.notna(record), None).to_dict())
        st.download_button("Download filtered customers", view.to_csv(index=False).encode("utf-8"),
                           "filtered_customer_segments.csv", "text/csv")

    elif page == "Marketing Intelligence":
        st.subheader("Marketing intelligence")
        if action_col:
            group_cols = [c for c in [cluster_col, campaign_col or action_col, priority_col] if c]
            summary = filtered.groupby(group_cols, dropna=False).size().reset_index(name="Customers")
            left, right = st.columns([1.2, 1])
            with left:
                st.dataframe(summary.sort_values("Customers", ascending=False),
                             use_container_width=True, hide_index=True)
            with right:
                st.plotly_chart(px.bar(summary, x=cluster_col, y="Customers",
                                       color=campaign_col or action_col, barmode="stack",
                                       title="Marketing actions by segment"),
                                use_container_width=True, config=chart_config())
        else:
            st.info("No marketing action field is available in the segment data.")
        if definitions is not None:
            st.subheader("Marketing strategy reference")
            st.dataframe(definitions, use_container_width=True, hide_index=True)

    elif page == "Model Evaluation":
        st.subheader("Model evaluation")
        if models is None or models.empty:
            st.info("Model evaluation CSV was not found in the expected results folder.")
        else:
            st.caption(f"Source: {model_source}")
            st.dataframe(models, use_container_width=True, hide_index=True)
            metric_candidates = {
                "Silhouette (higher is better)": ["Silhouette_mean", "Silhouette", "silhouette_score"],
                "Davies–Bouldin (lower is better)": ["DB_mean", "DaviesBouldin", "Davies_Bouldin"],
                "Calinski–Harabasz (higher is better)": ["CH_mean", "CalinskiHarabasz", "Calinski_Harabasz"],
                "SSE (lower is better)": ["SSE_mean", "SSE", "Inertia"],
            }
            available = {label: find_column(models, candidates) for label, candidates in metric_candidates.items()}
            available = {label: column for label, column in available.items() if column}
            k_col = find_column(models, ["K", "k", "n_clusters"])
            method_col = find_column(models, ["Method", "method", "Algorithm"])
            if available and k_col:
                metric = st.selectbox("Evaluation metric", list(available))
                st.plotly_chart(px.line(models, x=k_col, y=available[metric],
                                        color=method_col if method_col else None, markers=True,
                                        title=metric), use_container_width=True, config=chart_config())
            st.caption("Interpret clustering metrics alongside cluster size, stability and business usefulness.")

    elif page == "Reports & Downloads":
        st.subheader("Reports and downloads")
        st.write("Download the current filtered customer results and the available research reference tables.")
        st.download_button("Download customer segments CSV", filtered.to_csv(index=False).encode("utf-8"),
                           "customer_segments.csv", "text/csv")
        if models is not None:
            st.download_button("Download model evaluation CSV", models.to_csv(index=False).encode("utf-8"),
                               "model_evaluation.csv", "text/csv")
        if definitions is not None:
            st.download_button("Download segment / marketing reference CSV",
                               definitions.to_csv(index=False).encode("utf-8"),
                               "segment_marketing_reference.csv", "text/csv")
        if LIVE_TRANSACTIONS_PATH.exists():
            live = pd.read_csv(LIVE_TRANSACTIONS_PATH)
            st.metric("Stored live transactions", f"{len(live):,}")
            st.download_button("Download live transactions CSV", live.to_csv(index=False).encode("utf-8"),
                               "live_transactions.csv", "text/csv")
        with st.expander("Data quality and provenance"):
            st.write({"rows": len(data_all), "columns": len(data_all.columns),
                      "duplicate_customer_ids": int(data_all[customer_col].duplicated().sum()) if customer_col else None,
                      "missing_segment_labels": int(data_all[cluster_col].isna().sum()),
                      "segment_source": segment_source, "model_source": model_source})
            st.caption("This dashboard visualizes saved Phase 2 outputs. It does not retrain QLDE + K-Means.")
