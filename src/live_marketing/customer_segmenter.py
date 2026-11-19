import joblib
import numpy as np
import pandas as pd

from .config import EXISTING_SEGMENTS_PATH, PROJECT_ROOT


MODEL_DIR = PROJECT_ROOT / "models" / "live_marketing"

SCALER_PATH = MODEL_DIR / "phase2_scaler.joblib"
PCA_PATH = MODEL_DIR / "phase2_pca.joblib"
KMEANS_PATH = MODEL_DIR / "phase2_kmeans.joblib"
FEATURE_COLUMNS_PATH = MODEL_DIR / "phase2_feature_columns.joblib"

SEGMENT_DETAILS = {
    0: {
        "Status": "Inactive/Declining",
        "MarketingObjective": "Reactivation",
        "Recommendation": (
            "Win-back campaign with personalized offers based on previous purchases."
        ),
        "CampaignType": "Reactivation",
        "Priority": "High",
    },
    1: {
        "Status": "Established Active",
        "MarketingObjective": "Retention/Cross-sell",
        "Recommendation": (
            "Cross-sell complementary products and provide loyalty-oriented offers."
        ),
        "CampaignType": "Retention / Cross-sell",
        "Priority": "Medium",
    },
    2: {
        "Status": "New/Recently Acquired",
        "MarketingObjective": "Onboarding",
        "Recommendation": (
            "Welcome journey with product discovery and a second-purchase incentive."
        ),
        "CampaignType": "New Customer Onboarding",
        "Priority": "High",
    },
    3: {
        "Status": "Very High-Value Active",
        "MarketingObjective": "High-value retention",
        "Recommendation": (
            "Premium personalized recommendations, loyalty benefits, and early access."
        ),
        "CampaignType": "High-Value Retention",
        "Priority": "High",
    },
}


def load_existing_segments() -> pd.DataFrame:
    """Load saved Phase 2 assignments without modifying the source file."""

    if not EXISTING_SEGMENTS_PATH.exists():
        return pd.DataFrame(columns=["CustomerID", "Cluster"])

    segments = pd.read_csv(EXISTING_SEGMENTS_PATH)

    if "CustomerID" not in segments.columns or "Cluster" not in segments.columns:
        raise ValueError(
            "The Phase 2 segments file must contain CustomerID and Cluster."
        )

    segments["CustomerID"] = pd.to_numeric(
        segments["CustomerID"], errors="coerce"
    )

    return segments.dropna(subset=["CustomerID"]).drop_duplicates(
        subset=["CustomerID"], keep="last"
    )


def predict_new_customer_clusters(features: pd.DataFrame) -> pd.Series:
    """Predict clusters using the separately saved reconstructed Phase 2 model."""

    required_columns = [
        "Recency",
        "Frequency",
        "Monetary",
        "RecentSpendingShare",
        "ActivityConsistency",
        "NewCustomer",
    ]

    missing = [column for column in required_columns if column not in features.columns]

    if missing:
        raise ValueError(
            "Cannot predict new customer clusters. Missing Phase 2 features: "
            + ", ".join(missing)
        )

    for path in [
        SCALER_PATH,
        PCA_PATH,
        KMEANS_PATH,
        FEATURE_COLUMNS_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required Phase 2 model artifact was not found: {path}"
            )

    scaler = joblib.load(SCALER_PATH)
    pca = joblib.load(PCA_PATH)
    model = joblib.load(KMEANS_PATH)
    model_columns = joblib.load(FEATURE_COLUMNS_PATH)

    model_input = features.loc[:, model_columns].copy()

    for column in ["Recency", "Frequency", "Monetary"]:
        model_input[column] = np.log1p(
            model_input[column].clip(lower=0)
        )

    scaled_features = scaler.transform(model_input)
    reduced_features = pca.transform(scaled_features)

    return pd.Series(
        model.predict(reduced_features),
        index=features.index,
        name="Cluster",
    )


def assign_customer_segments(features: pd.DataFrame) -> pd.DataFrame:
    """
    Preserve saved assignments for known customers and predict clusters
    for new customers using the reconstructed Phase 2 model.
    """

    if features.empty:
        return features.copy()

    if "CustomerID" not in features.columns:
        raise ValueError("Customer features must contain CustomerID.")

    result = features.copy()

    result["CustomerID"] = pd.to_numeric(
        result["CustomerID"], errors="coerce"
    )

    existing = load_existing_segments()

    # Merge only saved assignment columns to avoid duplicate feature columns.
    saved_columns = [
        column
        for column in [
            "CustomerID",
            "Cluster",
            "Recommendation",
            "Priority",
            "Status",
            "MarketingObjective",
            "CampaignType",
        ]
        if column in existing.columns
    ]

    # Remove old assignment fields from the input before merging.
    assignment_columns = [
        column
        for column in saved_columns
        if column != "CustomerID" and column in result.columns
    ]
    result = result.drop(columns=assignment_columns)

    if not existing.empty:
        result = result.merge(
            existing[saved_columns],
            on="CustomerID",
            how="left",
            validate="many_to_one",
        )
    else:
        result["Cluster"] = pd.NA

    if "Cluster" not in result.columns:
        result["Cluster"] = pd.NA

    result["AssignmentMethod"] = pd.NA

    known_mask = result["Cluster"].notna()
    result.loc[known_mask, "AssignmentMethod"] = (
        "Existing Phase 2 assignment"
    )

    new_mask = ~known_mask

    if new_mask.any():
        predicted_clusters = predict_new_customer_clusters(
            result.loc[new_mask]
        )

        result.loc[new_mask, "Cluster"] = predicted_clusters.astype(int)
        result.loc[new_mask, "AssignmentMethod"] = (
            "Reconstructed Phase 2 model"
        )

    # Fill marketing details only when the saved assignment lacks them.
    result["Cluster"] = pd.to_numeric(result["Cluster"], errors="raise").astype(int)

    for column in [
        "Status",
        "MarketingObjective",
        "Recommendation",
        "CampaignType",
        "Priority",
    ]:
        if column not in result.columns:
            result[column] = pd.NA

    for cluster_id, details in SEGMENT_DETAILS.items():
        cluster_mask = result["Cluster"] == cluster_id

        for column, value in details.items():
            missing_value_mask = result[column].isna() | (
                result[column].astype(str).str.strip() == ""
            )
            result.loc[cluster_mask & missing_value_mask, column] = value

    return result

