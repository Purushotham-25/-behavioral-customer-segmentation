from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]

TRANSACTIONS_PATH = ROOT / "data/processed/clean_transactions.csv"
CUSTOMER_FEATURES_PATH = ROOT / "data/processed/customer_features.csv"
SAVED_SEGMENTS_PATH = ROOT / "results/phase2/phase2_customer_segments.csv"

SPLIT_DATE = pd.Timestamp("2011-07-01")
HISTORICAL_MONTHS = 7
RECENT_MONTHS = 6

FEATURES = [
    "Recency",
    "Frequency",
    "Monetary",
    "RecentSpendingShare",
    "ActivityConsistency",
    "NewCustomer",
]

LOG_FEATURES = {"Recency", "Frequency", "Monetary"}


def period_features(transactions, months, prefix):
    data = transactions.copy()
    data["TransactionValue"] = data["Quantity"] * data["UnitPrice"]

    grouped = data.groupby("CustomerID").agg(
        Spending=("TransactionValue", "sum"),
        Frequency=( "InvoiceNo", "nunique"),
        Quantity=("Quantity", "sum"),
    )

    grouped[f"{prefix}MonthlySpending"] = grouped["Spending"] / months
    grouped[f"{prefix}MonthlyFrequency"] = grouped["Frequency"] / months
    grouped[f"{prefix}MonthlyQuantity"] = grouped["Quantity"] / months

    return grouped[
        [
            f"{prefix}MonthlySpending",
            f"{prefix}MonthlyFrequency",
            f"{prefix}MonthlyQuantity",
        ]
    ]


def main():
    print("Loading historical data...")

    transactions = pd.read_csv(
        TRANSACTIONS_PATH,
        usecols=[
            "InvoiceNo",
            "Quantity",
            "InvoiceDate",
            "UnitPrice",
            "CustomerID",
        ],
        parse_dates=["InvoiceDate"],
    )

    transactions["CustomerID"] = pd.to_numeric(
        transactions["CustomerID"], errors="coerce"
    )
    transactions["Quantity"] = pd.to_numeric(
        transactions["Quantity"], errors="coerce"
    )
    transactions["UnitPrice"] = pd.to_numeric(
        transactions["UnitPrice"], errors="coerce"
    )

    transactions = transactions.dropna(
        subset=["CustomerID", "Quantity", "UnitPrice", "InvoiceDate"]
    )

    transactions = transactions[
        ~transactions["InvoiceNo"].astype(str).str.startswith("C")
    ]

    historical = transactions[transactions["InvoiceDate"] < SPLIT_DATE]
    recent = transactions[transactions["InvoiceDate"] >= SPLIT_DATE]

    historical_features = period_features(
        historical, HISTORICAL_MONTHS, "Historical"
    )
    recent_features = period_features(recent, RECENT_MONTHS, "Recent")

    phase2 = historical_features.join(recent_features, how="outer").fillna(0)

    historical_frequency = phase2["HistoricalMonthlyFrequency"]
    recent_frequency = phase2["RecentMonthlyFrequency"]

    total_spending = (
        phase2["HistoricalMonthlySpending"]
        + phase2["RecentMonthlySpending"]
    )

    phase2["RecentSpendingShare"] = np.where(
        total_spending > 0,
        phase2["RecentMonthlySpending"] / total_spending,
        0,
    )

    max_frequency = np.maximum(historical_frequency, recent_frequency)

    phase2["ActivityConsistency"] = np.divide(
        np.minimum(historical_frequency, recent_frequency),
        max_frequency,
        out=np.zeros(len(phase2), dtype=float),
        where=max_frequency.to_numpy() > 0,
    )

    phase2["NewCustomer"] = (
        (historical_frequency == 0) & (recent_frequency > 0)
    ).astype(int)

    customers = pd.read_csv(CUSTOMER_FEATURES_PATH)
    saved = pd.read_csv(SAVED_SEGMENTS_PATH)

    customers["CustomerID"] = pd.to_numeric(
        customers["CustomerID"], errors="coerce"
    )
    saved["CustomerID"] = pd.to_numeric(
        saved["CustomerID"], errors="coerce"
    )

    # Preserve notebook row-order assumptions and stop if they do not match.
    if not customers["CustomerID"].reset_index(drop=True).equals(
        phase2.index.to_series().reset_index(drop=True)
    ):
        raise ValueError(
            "Customer order differs from the notebook's assumed row order. "
            "Reconstruction stopped to avoid misleading cluster comparisons."
        )

    master = customers.copy()
    master["RecentSpendingShare"] = phase2["RecentSpendingShare"].to_numpy()
    master["ActivityConsistency"] = phase2["ActivityConsistency"].to_numpy()

    transformed = master[FEATURES].copy()

    for feature in LOG_FEATURES:
        transformed[feature] = np.log1p(transformed[feature])

    scaler = StandardScaler()
    scaled = scaler.fit_transform(transformed)

    pca = PCA(n_components=0.90, svd_solver="full")
    reduced = pca.fit_transform(scaled)

    model = KMeans(
        n_clusters=4,
        random_state=82,
        n_init=20,
        max_iter=300,
    )
    predicted = model.fit_predict(reduced)

    comparison = saved[["CustomerID", "Cluster"]].copy()
    comparison = comparison.merge(
        customers[["CustomerID"]].assign(ReconstructedCluster=predicted),
        on="CustomerID",
        how="inner",
        validate="one_to_one",
    )

    print("\n--- Reconstruction results ---")
    print("Customers:", len(comparison))
    print("PCA components:", pca.n_components_)
    print("Explained variance:", round(pca.explained_variance_ratio_.sum(), 4))
    print("Adjusted Rand Index:", round(
        adjusted_rand_score(
            comparison["Cluster"],
            comparison["ReconstructedCluster"],
        ),
        4,
    ))
    print("\nSaved cluster counts:")
    print(comparison["Cluster"].value_counts().sort_index())
    print("\nReconstructed cluster counts:")
    print(comparison["ReconstructedCluster"].value_counts().sort_index())
    print(
        "\nExact label matches:",
        int((comparison["Cluster"] == comparison["ReconstructedCluster"]).sum()),
        "of",
        len(comparison),
    )

    print(
        "\nNote: this is a reconstruction, not the original fitted model. "
        "Do not use it for live assignments until its results are reviewed."
    )
        # Save reconstructed model artifacts separately.
    model_dir = ROOT / "models" / "live_marketing"
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, model_dir / "phase2_scaler.joblib")
    joblib.dump(pca, model_dir / "phase2_pca.joblib")
    joblib.dump(model, model_dir / "phase2_kmeans.joblib")
    joblib.dump(FEATURES, model_dir / "phase2_feature_columns.joblib")

    print("\nModel artifacts saved to:", model_dir)

if __name__ == "__main__":
    main()