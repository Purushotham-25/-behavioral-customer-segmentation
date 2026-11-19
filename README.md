# Behavioral Customer Segmentation for Personalized Marketing

## Project Overview

This project develops a behavioral customer segmentation pipeline for personalized marketing using transactional retail data.

The project combines customer-level behavioral feature engineering, feature transformation, standardization, Principal Component Analysis (PCA), K-Means clustering, and a Q-Learning based Differential Evolution (QLDE) clustering approach with K-Means refinement.

## Phase 01

Phase 01 establishes a reproducible customer segmentation and model-evaluation pipeline.

### Pipeline

Raw Transaction Data
? Data Preprocessing
? Customer Feature Engineering
? Feature Transformation
? Feature Scaling
? PCA
? K Selection
? Standard K-Means
? QLDE
? QLDE + K-Means
? Model Evaluation
? Customer Cluster Profiles

## Customer Features

The customer-level representation contains 13 features:

1. Recency
2. Frequency
3. Monetary
4. CustomerLifespanDays
5. AveragePurchaseIntervalDays
6. TotalQuantity
7. ProductDiversity
8. AverageBasketQuantity
9. AverageTransactionValue
10. SpendingTrend
11. FrequencyTrend
12. QuantityTrend
13. NewCustomer

`CustomerID` is retained for identification and excluded from model training.

## Dimensionality Reduction

PCA is applied after feature transformation and standardization.

- PCA variance target: 90%
- Selected components: 5
- Explained variance: 90.9369%

The PCA representation is used consistently for the clustering experiments.

## Clustering Models

### Standard K-Means

Standard K-Means is used as the baseline model.

Configuration includes:

- k-means++ initialization
- Multiple initializations
- Maximum iteration limit
- Repeated evaluation across controlled seeds

### QLDE

The proposed clustering implementation uses a Q-Learning based Differential Evolution framework.

The implementation includes:

- Centroid-based chromosome representation
- Valid population initialization
- Differential Evolution mutation
- Binomial crossover
- Selection based on clustering fitness
- Individual Q-tables
- Adaptive differential-evolution scaling factor
- Q-learning states and rewards
- Elitism
- Stagnation handling
- Empty-cluster validation

### QLDE + K-Means

QLDE-generated centroids are subsequently refined using K-Means.

This hybrid model is evaluated under the same PCA representation, candidate K values, and random seeds as the baseline.

## K Selection

Candidate values from K = 2 through K = 10 are evaluated.

K selection uses a documented multi-metric ranking based on:

- Silhouette Score � higher is better
- Davies-Bouldin Index � lower is better
- Calinski-Harabasz Index � higher is better

A candidate is eligible only when:

- No empty clusters are produced
- Mean minimum cluster size is at least 1% of the customer population

SSE is not used to select K.

### Selected K

**K = 2**

The selected solution contains:

| Cluster | Customers | Percentage |
|---|---:|---:|
| Cluster 0 | 1,850 | 42.68% |
| Cluster 1 | 2,485 | 57.32% |
| **Total** | **4,335** | **100%** |

## Model Evaluation

The models are evaluated using:

- SSE
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Empty-cluster validation
- Cluster-size statistics
- Repeated-seed stability

Five controlled seeds are used:

`42, 52, 62, 72, 82`

### K = 2 Results

| Model | SSE | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---:|---:|---:|---:|
| Standard K-Means | 34196.544 | 0.312118 | 1.306380 | 2160.501 |
| QLDE | 34197.249 | 0.312150 | 1.306271 | 2160.488 |
| QLDE + K-Means | 34196.550 | 0.312153 | 1.306310 | 2160.501 |

## Phase 01 Result

The Phase 01 experiments demonstrate that the corrected QLDE implementation is valid, reproducible, stable, and produces non-empty customer clusters.

At the selected K = 2, QLDE + K-Means produces results extremely close to Standard K-Means.

Therefore, Phase 01 does not claim an overall performance improvement over the baseline. The experimental baseline and validated segmentation pipeline established in Phase 01 will be used as the foundation for the proposed improvements in Phase 02.

## Repository Structure

```text
Behavioral-customer-segmentation/
�
+-- data/
�   +-- raw/
�   +-- processed/
�
+-- notebooks/
�   +-- 01_dataset_understanding.ipynb
�   +-- 02_data_preprocessing.ipynb
�   +-- 03_customer_feature_engineering.ipynb
�   +-- 04_customer_segmentation_modeling.ipynb
�
+-- src/
�   +-- __init__.py
�   +-- segmentation_pipeline.py
�
+-- results/
�   +-- archive/
�   +-- figures/
�   +-- metrics/
�   +-- tables/
�
+-- .gitignore
+-- README.md
+-- requirements.txt

## Unified Streamlit Dashboard

The primary dashboard entry point is now `dashboard/app.py`. Launch it from the project root:

```powershell
python -m pip install -r requirements.txt
streamlit run dashboard/app.py
```

The unified sidebar provides Overview, Customer Segmentation, Segment Profiles, Customer Explorer, Marketing Intelligence, Live Transactions, Model Evaluation, and Reports & Downloads. See `DASHBOARD_GUIDE.md` for usage and `PROJECT_STRUCTURE.md` for the folder map.

The dashboard reads saved Phase 2 outputs and reuses the existing live-marketing service. It does not retrain or modify the QLDE + K-Means research implementation. Historical transaction data remains read-only and new transactions remain separate under `data/live/`.
