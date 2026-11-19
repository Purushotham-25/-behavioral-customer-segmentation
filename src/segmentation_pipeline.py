"""Reproducible QLDE + K-Means customer-segmentation experiment."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


FEATURE_COLUMNS = [
    "Recency", "Frequency", "Monetary", "CustomerLifespanDays",
    "AveragePurchaseIntervalDays", "TotalQuantity", "ProductDiversity",
    "AverageBasketQuantity", "AverageTransactionValue", "SpendingTrend",
    "FrequencyTrend", "QuantityTrend", "NewCustomer",
]


@dataclass(frozen=True)
class ExperimentConfig:
    pca_variance_target: float = 0.90
    candidate_pcs: tuple = (4, 5, 7)
    k_values: tuple = tuple(range(2, 11))
    seeds: tuple = (42, 52, 62, 72, 82)
    population_cap: int = 80
    iterations: int = 100
    f_initial: float = 0.55
    f_min: float = 0.10
    f_max: float = 0.90
    crossover_rate: float = 0.90
    q_alpha: float = 0.15
    q_gamma: float = 0.90
    q_epsilon: float = 0.20
    q_actions: tuple = (-0.05, -0.02, 0.0, 0.02, 0.05)
    stagnation_limit: int = 15
    reinitialization_fraction: float = 0.10
    kmeans_max_iter: int = 300
    kmeans_n_init: int = 20


def _squared_distances(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return np.maximum(
        (x * x).sum(axis=1)[:, None]
        + (centroids * centroids).sum(axis=1)[None, :]
        - 2.0 * x @ centroids.T,
        0.0,
    )


def assign_and_valid_sse(x: np.ndarray, centroids: np.ndarray):
    """Nearest-centroid labels and SSE; empty clusters receive infinite fitness."""
    squared = _squared_distances(x, centroids)
    labels = squared.argmin(axis=1)
    counts = np.bincount(labels, minlength=len(centroids))
    if np.any(counts == 0):
        return np.inf, labels, counts
    return float(squared[np.arange(len(x)), labels].sum()), labels, counts


def chromosome_to_centroids(chromosome: np.ndarray, k: int, d: int) -> np.ndarray:
    return chromosome.reshape(k, d)


def centroids_to_chromosome(centroids: np.ndarray) -> np.ndarray:
    return centroids.ravel()


def initialize_population(x: np.ndarray, k: int, size: int, rng: np.random.Generator):
    """Diverse valid population: one k-means++ seed and sampled/jittered centroids."""
    n, d = x.shape
    lower, upper = x.min(axis=0), x.max(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-8)
    population = []
    for member in range(size):
        # A k-means++ draw provides non-duplicate, data-supported starting centres.
        _, indices = kmeans_plusplus(x, n_clusters=k, random_state=int(rng.integers(2**31 - 1)))
        centroids = x[indices].copy()
        if member:
            centroids += rng.normal(0.0, 0.05, size=(k, d)) * scale
            centroids = np.clip(centroids, lower, upper)
        sse, _, _ = assign_and_valid_sse(x, centroids)
        if np.isfinite(sse):
            population.append(centroids_to_chromosome(centroids))
    # K-means++ normally makes all starts valid; retain a safe deterministic fallback.
    while len(population) < size:
        indices = rng.choice(n, size=k, replace=False)
        population.append(centroids_to_chromosome(x[indices]))
    return np.asarray(population), lower, upper


def evaluate_centroid_model(x: np.ndarray, centroids: np.ndarray) -> dict:
    sse, labels, counts = assign_and_valid_sse(x, centroids)
    base = {
        "SSE": sse,
        "EmptyClusters": int((counts == 0).sum()),
        "MinClusterSize": int(counts.min()),
        "MaxClusterSize": int(counts.max()),
        "labels": labels,
        "counts": counts,
    }
    if not np.isfinite(sse) or len(np.unique(labels)) < 2:
        return base | {"Silhouette": np.nan, "DaviesBouldin": np.nan, "CalinskiHarabasz": np.nan}
    return base | {
        "Silhouette": float(silhouette_score(x, labels)),
        "DaviesBouldin": float(davies_bouldin_score(x, labels)),
        "CalinskiHarabasz": float(calinski_harabasz_score(x, labels)),
    }


def _state(reward: float, diversity: float) -> int:
    # 0 = material improvement, 1 = small/no improvement, 2 = low diversity.
    if diversity < 0.03:
        return 2
    return 0 if reward > 0.001 else 1


def run_qlde(x: np.ndarray, k: int, seed: int, config: ExperimentConfig) -> dict:
    """QLDE with valid fitness, individual Q tables, adaptive F, and elitist diversity resets."""
    rng = np.random.default_rng(seed)
    _, d = x.shape
    dimension = k * d
    pop_size = min(config.population_cap, max(30, 5 * dimension))
    population, lower, upper = initialize_population(x, k, pop_size, rng)
    fitness = np.array([assign_and_valid_sse(x, chromosome_to_centroids(c, k, d))[0] for c in population])
    actions = np.asarray(config.q_actions)
    # Individual policies avoid order-dependent sharing of unrelated trajectories.
    q_tables = np.zeros((pop_size, 3, len(actions)))
    states = np.ones(pop_size, dtype=int)
    f_values = np.full(pop_size, config.f_initial)
    stagnation = np.zeros(pop_size, dtype=int)
    best_idx = int(fitness.argmin())
    best = population[best_idx].copy()
    best_sse = float(fitness[best_idx])
    history = [{"Iteration": 0, "BestSSE": best_sse, "MeanSSE": float(fitness.mean()), "Diversity": float(np.std(population, axis=0).mean())}]

    for iteration in range(1, config.iterations + 1):
        diversity = float(np.std(population, axis=0).mean())
        epsilon = config.q_epsilon * (1.0 - (iteration - 1) / config.iterations)
        chosen = np.empty(pop_size, dtype=int)
        for i in range(pop_size):
            if rng.random() < epsilon:
                chosen[i] = int(rng.integers(len(actions)))
            else:
                chosen[i] = int(np.flatnonzero(q_tables[i, states[i]] == q_tables[i, states[i]].max())[0])
        f_values = np.clip(f_values + actions[chosen], config.f_min, config.f_max)
        order = np.argsort(fitness)
        best_chromosome = population[order[0]]
        trials = population.copy()
        for i in range(pop_size):
            candidates = np.array([j for j in range(pop_size) if j != i])
            r1, r2 = rng.choice(candidates, size=2, replace=False)
            # current-to-best/1: target, r1, and r2 are distinct.
            mutant = population[i] + f_values[i] * (best_chromosome - population[i]) + f_values[i] * (population[r1] - population[r2])
            mutant = np.clip(mutant.reshape(k, d), lower, upper).ravel()
            mask = rng.random(dimension) < config.crossover_rate
            mask[int(rng.integers(dimension))] = True
            trials[i, mask] = mutant[mask]
        trial_fitness = np.array([assign_and_valid_sse(x, chromosome_to_centroids(c, k, d))[0] for c in trials])
        for i in range(pop_size):
            improvement = max(0.0, (fitness[i] - trial_fitness[i]) / max(abs(fitness[i]), 1e-12))
            reward = improvement if np.isfinite(trial_fitness[i]) else -0.01
            next_state = _state(reward, diversity)
            old = q_tables[i, states[i], chosen[i]]
            q_tables[i, states[i], chosen[i]] = old + config.q_alpha * (reward + config.q_gamma * q_tables[i, next_state].max() - old)
            if trial_fitness[i] < fitness[i]:
                population[i], fitness[i], stagnation[i] = trials[i], trial_fitness[i], 0
            else:
                stagnation[i] += 1
            states[i] = next_state
        # Preserve the best individual; reset only stale non-elite members from real observations.
        stale = np.where((stagnation >= config.stagnation_limit) & (np.arange(pop_size) != int(fitness.argmin())))[0]
        reset_count = min(len(stale), max(1, int(np.ceil(pop_size * config.reinitialization_fraction))))
        if reset_count:
            for i in stale[:reset_count]:
                _, idx = kmeans_plusplus(x, n_clusters=k, random_state=int(rng.integers(2**31 - 1)))
                candidate = x[idx] + rng.normal(0, 0.03, size=(k, d)) * np.maximum(x.std(axis=0), 1e-8)
                candidate = np.clip(candidate, lower, upper)
                candidate_sse, _, _ = assign_and_valid_sse(x, candidate)
                if np.isfinite(candidate_sse):
                    population[i], fitness[i], stagnation[i] = candidate.ravel(), candidate_sse, 0
        idx = int(fitness.argmin())
        if fitness[idx] < best_sse:
            best, best_sse = population[idx].copy(), float(fitness[idx])
        history.append({"Iteration": iteration, "BestSSE": best_sse, "MeanSSE": float(fitness.mean()), "Diversity": float(np.std(population, axis=0).mean())})
    return {"centroids": chromosome_to_centroids(best, k, d), "history": pd.DataFrame(history), "population_size": pop_size}


def _record(model: str, k: int, seed: int, metrics: dict) -> dict:
    return {"Model": model, "K": k, "Seed": seed} | {key: metrics[key] for key in ["SSE", "Silhouette", "DaviesBouldin", "CalinskiHarabasz", "EmptyClusters", "MinClusterSize", "MaxClusterSize"]}


def _choose_k(summary: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    selection = summary.copy()
    valid = (selection.EmptyClustersMean == 0) & (selection.MinClusterSizeMean >= 0.01 * 4335)
    selection["Eligible"] = valid
    eligible = selection[valid].copy()
    # Equal-weight rank aggregation: quality measures first; SSE is intentionally excluded.
    eligible["DecisionRank"] = (
        eligible.SilhouetteMean.rank(ascending=False, method="min")
        + eligible.DaviesBouldinMean.rank(ascending=True, method="min")
        + eligible.CalinskiHarabaszMean.rank(ascending=False, method="min")
    )
    chosen = eligible.sort_values(["DecisionRank", "SilhouetteMean", "DaviesBouldinMean"], ascending=[True, False, True]).iloc[0]
    selection = selection.merge(eligible[["K", "DecisionRank"]], on="K", how="left")
    return int(chosen.K), selection


def run_experiment(project_root: str | Path) -> dict:
    root = Path(project_root)
    data_path = root / "data" / "processed" / "customer_features_scaled.csv"
    output = root / "results" / "tables"
    output.mkdir(parents=True, exist_ok=True)
    config = ExperimentConfig()
    data = pd.read_csv(data_path)
    assert len(data) == 4335 and set(FEATURE_COLUMNS).issubset(data.columns)
    x = data[FEATURE_COLUMNS].to_numpy(float)
    assert np.isfinite(x).all() and np.allclose(x.mean(0), 0, atol=1e-8) and np.allclose(x.std(0), 1, atol=1e-8)
    pca_full = PCA().fit(x)
    cumulative = pca_full.explained_variance_ratio_.cumsum()
    pca_dimension = 5 if cumulative[4] >= config.pca_variance_target else int(np.argmax(cumulative >= config.pca_variance_target) + 1)
    pca = PCA(n_components=pca_dimension, random_state=42)
    x_model = pca.fit_transform(x)
    pca_variance = pd.DataFrame({"Component": np.arange(1, len(cumulative) + 1), "ExplainedVariance": pca_full.explained_variance_ratio_, "CumulativeVariance": cumulative})
    loadings = pd.DataFrame(pca_full.components_.T, index=FEATURE_COLUMNS, columns=[f"PC{i}" for i in range(1, len(FEATURE_COLUMNS) + 1)])
    candidate_variance = pd.DataFrame({"Components": config.candidate_pcs, "CumulativeVariance": [cumulative[n - 1] for n in config.candidate_pcs]})

    baseline_rows = []
    for k in config.k_values:
        for seed in config.seeds:
            km = KMeans(n_clusters=k, init="k-means++", n_init=config.kmeans_n_init, max_iter=config.kmeans_max_iter, random_state=seed).fit(x_model)
            baseline_rows.append(_record("Standard K-Means", k, seed, evaluate_centroid_model(x_model, km.cluster_centers_)))
    baseline = pd.DataFrame(baseline_rows)
    baseline_summary = baseline.groupby("K", as_index=False).agg(**{f"{col}{stat}": (col, stat.lower()) for col in ["SSE", "Silhouette", "DaviesBouldin", "CalinskiHarabasz", "EmptyClusters", "MinClusterSize", "MaxClusterSize"] for stat in ["Mean", "Std"]})
    selected_k, k_selection = _choose_k(baseline_summary)

    rows, histories, final_runs = [], [], []
    # QLDE is repeated across every candidate K/seed. The refinement uses its actual centroids.
    for k in config.k_values:
        for seed in config.seeds:
            qlde = run_qlde(x_model, k, seed, config)
            qlde_metrics = evaluate_centroid_model(x_model, qlde["centroids"])
            rows.append(_record("QLDE", k, seed, qlde_metrics))
            refined = KMeans(n_clusters=k, init=qlde["centroids"], n_init=1, max_iter=config.kmeans_max_iter, random_state=seed).fit(x_model)
            refined_metrics = evaluate_centroid_model(x_model, refined.cluster_centers_)
            rows.append(_record("QLDE + K-Means", k, seed, refined_metrics))
            history = qlde["history"].assign(K=k, Seed=seed)
            histories.append(history)
            if k == selected_k:
                final_runs.append((seed, qlde, refined, refined_metrics))
    proposed = pd.DataFrame(rows)
    all_runs = pd.concat([baseline, proposed], ignore_index=True)
    summary = all_runs.groupby(["Model", "K"], as_index=False).agg(**{f"{col}{stat}": (col, stat.lower()) for col in ["SSE", "Silhouette", "DaviesBouldin", "CalinskiHarabasz", "EmptyClusters", "MinClusterSize", "MaxClusterSize"] for stat in ["Mean", "Std"]})
    # Select proposed final model by lowest refined SSE at the selected, predeclared K.
    final_seed, qlde_final, refined_final, final_metrics = min(final_runs, key=lambda v: v[3]["SSE"])
    labels = refined_final.labels_
    assignments = pd.DataFrame({"CustomerID": data.CustomerID, "Cluster": labels})
    profile_columns = ["Recency", "Frequency", "Monetary", "TotalQuantity", "ProductDiversity", "AverageBasketQuantity", "AverageTransactionValue", "SpendingTrend", "FrequencyTrend", "QuantityTrend"]
    original = pd.read_csv(root / "data" / "processed" / "customer_features.csv")
    profiles = original[["CustomerID"] + profile_columns].merge(assignments, on="CustomerID", validate="one_to_one").groupby("Cluster")[profile_columns].mean().reset_index()
    distribution = assignments.Cluster.value_counts().sort_index().rename_axis("Cluster").reset_index(name="CustomerCount")
    distribution["Percentage"] = 100 * distribution.CustomerCount / len(assignments)
    histories_df = pd.concat(histories, ignore_index=True)
    # Save explicit, reproducible artifacts.
    pca_variance.to_csv(output / "pca_variance.csv", index=False)
    loadings.to_csv(output / "pca_loadings.csv")
    candidate_variance.to_csv(output / "pca_candidate_variance.csv", index=False)
    k_selection.to_csv(output / "k_selection.csv", index=False)
    all_runs.to_csv(output / "all_model_runs.csv", index=False)
    summary.to_csv(output / "final_model_comparison.csv", index=False)
    assignments.to_csv(output / "final_cluster_assignments.csv", index=False)
    profiles.to_csv(output / "final_cluster_profiles.csv", index=False)
    distribution.to_csv(output / "final_cluster_distribution.csv", index=False)
    histories_df.to_csv(output / "qlde_convergence_history.csv", index=False)
    np.save(output / "final_qlde_kmeans_centroids.npy", refined_final.cluster_centers_)
    dump(pca, output / "pca_model.joblib")
    with open(output / "final_configuration.json", "w", encoding="utf-8") as handle:
        json.dump(asdict(config) | {"selected_pca_components": pca_dimension, "selected_k": selected_k, "final_seed": final_seed, "k_selection_rule": "Equal-weight rank of mean silhouette (high), Davies-Bouldin (low), and Calinski-Harabasz (high); eligible only when no empty clusters and mean minimum size >=1% of customers. SSE excluded from K decision."}, handle, indent=2)
    return {"config": config, "pca_dimension": pca_dimension, "selected_k": selected_k, "final_seed": final_seed, "baseline": baseline, "proposed": proposed, "summary": summary, "k_selection": k_selection, "distribution": distribution, "profiles": profiles, "history": histories_df, "final_metrics": final_metrics}

def main() -> None:
    """Run the complete Phase 01 segmentation experiment."""
    project_root = Path(__file__).resolve().parents[1]

    result = run_experiment(project_root)

    print("\n" + "=" * 70)
    print("PHASE 01 SEGMENTATION EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Customers:          {result['distribution']['CustomerCount'].sum()}")
    print(f"Features:           {len(FEATURE_COLUMNS)}")
    print(f"PCA components:     {result['pca_dimension']}")
    print(f"Selected K:         {result['selected_k']}")
    print(f"Final seed:         {result['final_seed']}")
    print("\nFinal cluster distribution:")
    print(result["distribution"].to_string(index=False))

    print("\nFinal evaluation:")
    for metric in [
        "SSE",
        "Silhouette",
        "DaviesBouldin",
        "CalinskiHarabasz",
    ]:
        print(f"{metric:20s}: {result['final_metrics'][metric]:.6f}")

    print("=" * 70)


if __name__ == "__main__":
    main()
