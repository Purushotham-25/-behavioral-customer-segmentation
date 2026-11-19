from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Customer Segmentation Dashboard", page_icon="📊", layout="wide")
ROOTS = [Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent, Path.cwd()]
SEGMENT_PATHS = ["results/phase2/phase2_customer_segments.csv", "results/phase2_customer_segments.csv"]
MODEL_PATHS = ["results/phase2/tables/phase2_model_comparison.csv",
               "results/phase2/tables/phase2_qlde_kmeans_results.csv",
               "results/phase2/tables/phase2_qlde_results.csv",
               "data/processed/qlde_k_screening_results.csv"]
PROFILE_PATHS = ["results/phase2/tables/phase2_segment_definitions.csv",
                 "results/phase2/tables/phase2_marketing_strategy.csv"]

def locate(paths):
    for root in ROOTS:
        for rel in paths:
            p = root / rel
            if p.is_file():
                return p
    return None

def load(upload, path):
    if upload is not None:
        return pd.read_csv(upload), f"uploaded: {upload.name}"
    if path:
        try:
            return pd.read_csv(path), str(path)
        except Exception as e:
            st.warning(f"Could not read {path}: {e}")
    return None, "not found"

def col(df, candidates):
    return next((c for c in candidates if df is not None and c in df.columns), None)

st.title("📊 Customer Segmentation Dashboard")
st.caption("Behavioral Customer Segmentation · Phase 2")
st.write("Explore customer groups, compare clustering metrics, and review marketing recommendations.")

with st.sidebar:
    st.header("Data sources")
    seg_upload = st.file_uploader("Customer segments CSV", type="csv")
    model_upload = st.file_uploader("Model comparison CSV (optional)", type="csv")
    profile_upload = st.file_uploader("Segment definitions / marketing CSV (optional)", type="csv")

segments, seg_source = load(seg_upload, locate(SEGMENT_PATHS))
models, model_source = load(model_upload, locate(MODEL_PATHS))
profiles, profile_source = load(profile_upload, locate(PROFILE_PATHS))

if segments is None or segments.empty:
    st.warning("Customer segment data was not found. Place `phase2_customer_segments.csv` in `results/phase2/`, or upload it in the sidebar.")
    st.code("Behavioral-customer-segmentation/\n  dashboard/app.py\n  results/phase2/phase2_customer_segments.csv\n  results/phase2/tables/phase2_model_comparison.csv")
    st.stop()

cluster = col(segments, ["Cluster", "cluster", "Segment", "segment"])
customer = col(segments, ["CustomerID", "customer_id", "Customer Id"])
segment_name = col(segments, ["SegmentName", "Segment_Name", "SegmentLabel", "Segment_Label"])
action = col(segments, ["Recommendation", "MarketingObjective", "CampaignType", "recommendation"])
priority = col(segments, ["Priority", "priority"])
if cluster is None:
    st.error(f"No cluster column detected. Available columns: {list(segments.columns)}")
    st.stop()

segments = segments.copy()
segments[cluster] = segments[cluster].astype(str)
with st.sidebar:
    st.header("Filters")
    all_clusters = sorted(segments[cluster].dropna().unique())
    chosen_clusters = st.multiselect("Segment / cluster", all_clusters, default=all_clusters)
    data = segments[segments[cluster].isin(chosen_clusters)].copy()
    if action:
        acts = sorted(data[action].dropna().astype(str).unique())
        chosen_actions = st.multiselect("Marketing action", acts, default=acts)
        data = data[data[action].astype(str).isin(chosen_actions)]

numeric = data.select_dtypes(include=np.number).columns.tolist()
numeric = [c for c in numeric if c.lower() not in {"customerid", "customer_id"}]
k1, k2, k3, k4 = st.columns(4)
k1.metric("Customers in view", f"{len(data):,}")
k2.metric("Segments in view", f"{data[cluster].nunique():,}")
k3.metric("Total customers", f"{len(segments):,}")
k4.metric("Numeric features", f"{len(numeric):,}")

overview, profiles_tab, marketing_tab, models_tab, explorer_tab = st.tabs(
    ["Overview", "Segment profiles", "Marketing actions", "Model evaluation", "Customer explorer"])

with overview:
    a, b = st.columns(2)
    counts = data[cluster].value_counts().rename_axis("Cluster").reset_index(name="Customers")
    counts["Share (%)"] = counts["Customers"] / max(len(data), 1) * 100
    with a:
        st.subheader("Customers by segment")
        st.plotly_chart(px.bar(counts, x="Cluster", y="Customers", text="Customers"), use_container_width=True)
    with b:
        st.subheader("Segment share")
        st.plotly_chart(px.pie(counts, names="Cluster", values="Customers", hole=.5), use_container_width=True)
    if numeric:
        default = [c for c in numeric if any(t in c.lower() for t in ["frequency", "monetary", "recency", "spend", "quantity", "diversity"])]
        chosen = st.multiselect("Behavioral features to compare", numeric, default=(default[:4] or numeric[:min(4, len(numeric))]))
        if chosen:
            long = data.groupby(cluster)[chosen].mean().reset_index().melt(id_vars=cluster, var_name="Feature", value_name="Mean")
            st.plotly_chart(px.bar(long, x="Feature", y="Mean", color=cluster, barmode="group"), use_container_width=True)
    st.caption(f"Segment file: {seg_source}")

with profiles_tab:
    st.subheader("Segment summary")
    rows = []
    for label, grp in data.groupby(cluster):
        row = {"Cluster": label, "Customers": len(grp), "Share (%)": len(grp) / max(len(data), 1) * 100}
        if segment_name:
            names = grp[segment_name].dropna().astype(str).unique()
            if len(names): row["Segment name"] = names[0]
        for feature in numeric:
            row[f"Mean {feature}"] = grp[feature].mean()
        if action:
            vals = grp[action].dropna().astype(str).unique()
            if len(vals): row["Marketing action"] = ", ".join(vals[:3])
        if priority:
            vals = grp[priority].dropna().astype(str).unique()
            if len(vals): row["Priority"] = ", ".join(vals[:3])
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if numeric:
        feature = st.selectbox("Profile feature", numeric)
        medians = data.groupby(cluster)[feature].median().reset_index()
        st.plotly_chart(px.bar(medians, x=cluster, y=feature, color=cluster), use_container_width=True)
    if profiles is not None:
        st.subheader("Existing definitions / strategy")
        st.dataframe(profiles, use_container_width=True, hide_index=True)
        st.caption(f"Reference file: {profile_source}")

with marketing_tab:
    st.subheader("Marketing actions by segment")
    if action:
        group_cols = [cluster, action] + ([priority] if priority else [])
        summary = data.groupby(group_cols, dropna=False).size().reset_index(name="Customers")
        summary["Share (%)"] = summary["Customers"] / max(len(data), 1) * 100
        st.dataframe(summary.sort_values("Customers", ascending=False), use_container_width=True, hide_index=True)
        st.plotly_chart(px.bar(summary, x=cluster, y="Customers", color=action, barmode="stack"), use_container_width=True)
    else:
        st.info("No marketing-action column detected in the segment file. Upload the Phase 2 marketing strategy CSV in the sidebar.")
    if profiles is not None:
        st.dataframe(profiles, use_container_width=True, hide_index=True)

with models_tab:
    st.subheader("Model comparison")
    if models is None or models.empty:
        st.info("Model comparison output not found. Upload `phase2_model_comparison.csv` or a QLDE results CSV in the sidebar.")
    else:
        st.caption(f"Model file: {model_source}")
        st.dataframe(models, use_container_width=True, hide_index=True)
        metric_candidates = {
            "Silhouette (higher is better)": ["Silhouette_mean", "Silhouette", "silhouette_score"],
            "Davies–Bouldin (lower is better)": ["DB_mean", "DaviesBouldin", "Davies_Bouldin"],
            "Calinski–Harabasz (higher is better)": ["CH_mean", "CalinskiHarabasz", "Calinski_Harabasz"],
            "SSE (lower is better)": ["SSE_mean", "SSE", "Inertia"],
        }
        available = {label: next((c for c in cs if c in models.columns), None) for label, cs in metric_candidates.items()}
        available = {k:v for k,v in available.items() if v}
        kcol = col(models, ["K", "k", "n_clusters"])
        if available and kcol:
            label = st.selectbox("Metric", list(available))
            method = col(models, ["Method", "method", "Algorithm"])
            st.plotly_chart(px.line(models, x=kcol, y=available[label], color=method if method else None, markers=True), use_container_width=True)
        st.caption("Interpret clustering metrics together with cluster size, stability, and business usefulness.")

with explorer_tab:
    st.subheader("Customer explorer")
    view = data.copy()
    if customer:
        search = st.text_input("Search customer ID")
        if search.strip():
            view = view[view[customer].astype(str).str.contains(search.strip(), na=False)]
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button("Download filtered customers CSV", view.to_csv(index=False).encode("utf-8"),
                       "filtered_customer_segments.csv", "text/csv")

with st.expander("Data quality notes"):
    st.write(f"Loaded rows: {len(segments):,}; columns: {len(segments.columns)}")
    st.write("Columns:", ", ".join(map(str, segments.columns)))
    if customer and segments[customer].duplicated().any():
        st.warning("Duplicate customer IDs detected; confirm whether the output should have one row per customer.")
    if segments[cluster].isna().any():
        st.warning("Some rows have missing cluster labels.")
    st.caption("This app visualizes existing Phase 2 CSVs; it does not retrain the segmentation model.")
