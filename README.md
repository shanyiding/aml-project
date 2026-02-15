# AML Risk Scoring & Behavioral Anomaly Detection (Work in Progress)

This project implements an end-to-end Anti-Money Laundering (AML) risk scoring pipeline using transaction behavior, KYC data, and unsupervised learning.

The current focus is on building strong behavioral features and learning customer representations using unsupervised methods, following approaches used by top competition solutions and industry AML systems.

This project is currently in active development and serves as a foundation for future explainable AML risk scoring models.

---

# Current Capabilities

The pipeline currently supports:

- Master dataset construction from raw transaction and KYC data  
- Train / validation / test splits at the customer level  
- Behavioral feature engineering from transaction history  
- Unsupervised customer embedding using:
  - PCA embeddings
  - K-Means clustering
  - Isolation Forest anomaly detection  
- Baseline supervised model training using engineered + unsupervised features  

These components form the foundation for AML anomaly detection and risk scoring.

---

# Project Structure

aml-risk-scoring/

data/
raw/ # Raw AML transaction and KYC data (not committed)
processed/ # Master datasets and train/test splits
interim/ # Engineered features and unsupervised embeddings

src/
data/
build_master.py
split_customers_master.py
build_customer_features.py
build_interim_features.py
build_unsupervised_features.py

features/           # Feature engineering modules (in progress)
models/             # Model training code (in progress)
pipeline/           # Pipeline orchestration (in progress)
notebooks/ # Jupyter notebooks for exploration and prototyping

knowledge_base/ # AML mappings, codes, and domain knowledge

docs/ # Documentation and methodology notes

README.md


---

# Pipeline Overview

The AML pipeline currently consists of the following stages:

---

## Stage 1 — Build Master Datasets

Combines raw transaction data and KYC data into unified master tables.

Outputs:

data/processed/customers_master.csv
data/processed/transactions_master.csv


These datasets contain:

- customer profile information
- transaction history
- fraud labels (when available)

---

## Stage 2 — Train / Validation / Test Split

Customers are split into:

customers_train.csv
customers_val.csv
customers_test.csv

transactions_train.csv
transactions_val.csv
transactions_test.csv


Splitting is done at the customer level to prevent data leakage.

---

## Stage 3 — Behavioral Feature Engineering

Customer-level behavioral features are created from transaction history, including:

- transaction frequency
- transaction amount statistics
- channel usage patterns
- international transaction behavior
- temporal activity patterns
- behavioral burstiness and velocity

Outputs stored in:

data/interim/


These features capture behavioral characteristics commonly used in AML detection systems.

---

## Stage 4 — Unsupervised Learning (Customer Embeddings)

Customer behavior representations are learned using unsupervised methods.

Models used:

- PCA (dimensionality reduction / embedding)
- K-Means clustering (customer segmentation)
- Isolation Forest anomaly detection

Outputs:

data/interim/unsup_features_train.csv
data/interim/unsup_features_val.csv
data/interim/unsup_features_test.csv
data/interim/unsup_features_unlabeled.csv


Each file contains:

customer_id
pca_1 … pca_10
cluster_id
cluster_dist
anomaly_score


These features capture behavioral structure without using labels and serve as learned representations of customer behavior.

---

## Stage 5 — Baseline Supervised Model

The supervised model uses:

- engineered behavioral features
- unsupervised embeddings
- customer profile features

to predict fraud risk.

This stage is currently experimental and being improved.

---

# Environment Setup

Activate virtual environment:

```bash
.venv/Scripts/activate
Install Jupyter support (if needed):

python -m pip install -U ipykernel jupyter
Pipeline Execution (Current Workflow)
Run these scripts in order from project root:

python src/data/build_customer_features.py
python src/data/build_master.py
python src/data/split_customers_master.py
python src/data/build_interim_features.py
python src/data/build_unsupervised_features.py
These scripts generate all processed datasets, behavioral features, and unsupervised embeddings required for modeling.

Generated Data Summary
After running the pipeline, the following datasets will exist:

data/processed/

customers_master.csv
transactions_master.csv

customers_train.csv
customers_val.csv
customers_test.csv

transactions_train.csv
transactions_val.csv
transactions_test.csv


data/interim/

unsup_features_train.csv
unsup_features_val.csv
unsup_features_test.csv
unsup_features_unlabeled.csv
```
These datasets form the current modeling inputs.

Modeling Approach
The system currently combines three modeling paradigms:

Behavioral Feature Engineering
Extracts interpretable behavioral signals from transaction data such as:

transaction velocity

activity burstiness

international transaction behavior

channel diversity and entropy

These features capture behavioral risk signals.

Unsupervised Learning
Learns behavioral representations without labels using:

PCA embeddings

clustering

anomaly detection

This allows detection of abnormal behavioral patterns even when labels are incomplete.

These embeddings serve as learned customer representations.

Supervised Learning (Baseline)
Uses:

behavioral features

unsupervised embeddings

customer profile features

to train fraud detection models.

This stage is currently being optimized.

Current Project Status
Completed:

Master dataset construction

Train / validation / test splitting

Behavioral feature engineering

Unsupervised embedding generation

Baseline supervised modeling pipeline

In Progress:

Feature engineering improvements

Unsupervised representation refinement

Model performance optimization

Evaluation and validation improvements

Planned:

Advanced behavioral feature engineering

Improved anomaly detection

Explainability integration

Production-level risk scoring system

Research Context
This project follows modern AML modeling strategies used in:

fraud detection systems

AML risk scoring systems

competition-winning AML pipelines

Including hybrid approaches combining:

behavioral feature engineering

unsupervised embedding learning

supervised risk scoring

This mirrors approaches used in industry and advanced AML research.

Next Steps
Immediate next goals:

improve behavioral feature engineering

integrate anomaly scores into supervised models

improve model performance metrics

analyze anomaly clusters and behavioral segments

Long-term goals:

production-quality AML risk scoring pipeline

interpretable fraud detection system

explainable customer risk scoring

