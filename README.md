# Behavioral Customer Segmentation for Personalized Marketing

An academic machine-learning project focused on behavioral customer segmentation using **Q-Learning based Differential Evolution (QLDE), K-Means clustering, and an interactive Streamlit dashboard**.

The project combines customer behavioral analysis, dimensionality reduction, clustering experiments, model evaluation, customer profiling, and personalized marketing intelligence.

---

## Project Overview

This project develops a behavioral customer segmentation pipeline using transactional retail data.

The research combines:

- Customer-level behavioral feature engineering
- Feature transformation and standardization
- Principal Component Analysis (PCA)
- Standard K-Means clustering
- Q-Learning based Differential Evolution (QLDE)
- QLDE + K-Means hybrid refinement
- Multi-metric model evaluation
- Customer segment profiling
- Personalized marketing intelligence

The repository also includes a unified Streamlit dashboard for exploring customer segments, evaluating models, and interacting with the live-marketing functionality.

## Key Highlights

| Component | Description |
|---|---|
| Customer features | 13 behavioral features |
| Dimensionality reduction | PCA |
| PCA components | 5 |
| Explained variance | 90.9369% |
| Candidate clusters | K = 2 to K = 10 |
| Selected clusters | K = 2 |
| Customer population | 4,335 |
| Clustering approaches | K-Means, QLDE, QLDE + K-Means |
| Evaluation seeds | 42, 52, 62, 72, 82 |
| Dashboard | Streamlit |

---

## Research Methodology

### Phase 01: Customer Segmentation Pipeline

The first phase establishes a reproducible customer segmentation and model-evaluation pipeline.

### Pipeline Architecture

```text
Raw Transaction Data
        |
        v
Data Preprocessing
        |
        v
Customer Feature Engineering
        |
        v
Feature Transformation
        |
        v
Feature Scaling
        |
        v
Principal Component Analysis
        |
        v
K Selection
        |
        +----------------------+
        |                      |
        v                      v
   Standard K-Means           QLDE
        |                      |
        |                      v
        |                K-Means Refinement
        |                      |
        +----------+-----------+
                   |
                   v
             Model Evaluation
                   |
                   v
          Customer Cluster Profiles
```

### Customer Behavioral Features

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

### Dimensionality Reduction

PCA is applied after feature transformation and standardization.

| Parameter | Value |
|---|---:|
| PCA variance target | 90% |
| Selected components | 5 |
| Explained variance | 90.9369% |

The PCA representation is used consistently for the clustering experiments.

---

## Clustering Algorithms

### 1. Standard K-Means

Standard K-Means serves as the baseline clustering model.

Configuration includes:

- K-Means++ initialization
- Multiple initializations
- Maximum iteration limit
- Repeated evaluation across controlled random seeds

### 2. QLDE

The proposed clustering implementation uses a Q-Learning based Differential Evolution framework.

Its implementation includes:

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

### 3. QLDE + K-Means

QLDE-generated centroids are subsequently refined using K-Means.

The hybrid model is evaluated under the same PCA representation, candidate K values, and random seeds as the baseline.

---

## Cluster Selection

Candidate values from K = 2 through K = 10 are evaluated.

K selection uses a documented multi-metric ranking based on:

| Metric | Preferred Direction |
|---|---|
| Silhouette Score | Higher |
| Davies-Bouldin Index | Lower |
| Calinski-Harabasz Index | Higher |

A candidate is eligible only when:

- No empty clusters are produced.
- Mean minimum cluster size is at least 1% of the customer population.

SSE is not used to select K.

### Selected Configuration

**Selected K = 2**

The selected solution contains 4,335 customers.

| Cluster | Customers | Percentage |
|---|---:|---:|
| Cluster 0 | 1,850 | 42.68% |
| Cluster 1 | 2,485 | 57.32% |
| **Total** | **4,335** | **100%** |

---

## Model Evaluation

The models are evaluated using:

- Sum of Squared Errors (SSE)
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Empty-cluster validation
- Cluster-size statistics
- Repeated-seed stability

Five controlled seeds are used:

`42, 52, 62, 72, 82`

### Experimental Results at K = 2

| Model | SSE | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---:|---:|---:|---:|
| Standard K-Means | 34196.544 | 0.312118 | 1.306380 | 2160.501 |
| QLDE | 34197.249 | 0.312150 | 1.306271 | 2160.488 |
| QLDE + K-Means | 34196.550 | 0.312153 | 1.306310 | 2160.501 |

### Phase 01 Findings

The Phase 01 experiments demonstrate that the corrected QLDE implementation is valid, reproducible, stable, and produces non-empty customer clusters.

At the selected K = 2, QLDE + K-Means produces results extremely close to Standard K-Means.

Therefore, Phase 01 does not claim an overall performance improvement over the baseline.

The experimental baseline and validated segmentation pipeline established in Phase 01 provide the foundation for the proposed improvements in Phase 02.

---

## Interactive Streamlit Dashboard

The project includes a unified Streamlit dashboard for exploring the segmentation outputs and live-marketing functionality.

### Dashboard Modules

| Module | Functionality |
|---|---|
| Overview | Project and segmentation overview |
| Customer Segmentation | Explore customer clustering results |
| Segment Profiles | Examine customer cluster characteristics |
| Customer Explorer | Explore customer-level information |
| Marketing Intelligence | Access marketing insights |
| Live Transactions | Work with the live transaction functionality |
| Model Evaluation | Explore model evaluation results |
| Reports & Downloads | Access available reports and downloadable outputs |

The dashboard reads saved Phase 2 outputs and reuses the existing live-marketing service.

It does not retrain or modify the QLDE + K-Means research implementation.

Historical transaction data remains read-only, and new transactions remain separate under `data/live/`.

---

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Plotly
- Jupyter Notebook
- Git and GitHub

---

## Repository Structure

```text
behavioral_customer_segmentation_unified/
│
├── dashboard/
│   ├── app.py
│   ├── live_marketing.py
│   └── README.md
│
├── data/
│   ├── live/
│   ├── processed/
│   └── raw/
│
├── models/
│   └── live_marketing/
│
├── notebooks/
│   ├── 01_dataset_understanding.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_customer_feature_engineering.ipynb
│   ├── 04_customer_segmentation_modeling.ipynb
│   └── 05_phase2_feature_engineering.ipynb
│
├── results/
│   ├── archive/
│   ├── figures/
│   ├── phase2/
│   └── tables/
│
├── scripts/
│   └── validate_phase2_reconstruction.py
│
├── src/
│   ├── live_marketing/
│   └── segmentation_pipeline.py
│
├── tests/
│   └── test_live_marketing.py
│
├── DASHBOARD_GUIDE.md
├── IMPLEMENTATION_NOTES.md
├── PROJECT_STRUCTURE.md
├── TESTING.md
├── requirements.txt
└── README.md
```

---

## Installation and Setup

### Prerequisites

- Python 3.12 recommended
- Git
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/Purushotham-25/-behavioral-customer-segmentation.git
```

### 2. Navigate to the Project

```bash
cd -behavioral-customer-segmentation
```

### 3. Create a Virtual Environment

**Windows:**

```powershell
py -m venv .venv
```

### 4. Activate the Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

---

## Running the Dashboard

From the project root, execute:

```bash
python -m streamlit run dashboard/app.py
```

Streamlit will start the application and provide a local browser address.

For additional dashboard instructions, see:

- `DASHBOARD_GUIDE.md`
- `PROJECT_STRUCTURE.md`

---

## Running Tests

Run the project tests from the repository root:

```bash
python -m pytest tests -v
```

The live-marketing test suite includes checks for:

- At-risk customer identification
- High-value customer identification
- Default classification for new customers

---

## Project Documentation

Additional project documentation is available in the repository:

| File | Purpose |
|---|---|
| `DASHBOARD_GUIDE.md` | Dashboard usage instructions |
| `IMPLEMENTATION_NOTES.md` | Implementation information |
| `PROJECT_STRUCTURE.md` | Project folder organization |
| `TESTING.md` | Testing instructions |
| `DASHBOARD_BUNDLE_CONTENTS.md` | Dashboard bundle contents |

---

## Future Development

Potential future work includes:

- Enhanced customer behavioral analysis
- More advanced interactive visualizations
- Improved customer segment interpretation
- Expanded personalized marketing strategies
- Further dashboard usability improvements

---

## Author

**Purushotham-25**

GitHub: [Purushotham-25](https://github.com/Purushotham-25)

---

*Behavioral Customer Segmentation for Personalized Marketing — combining machine learning research with interactive marketing intelligence.*
