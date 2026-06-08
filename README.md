
# Hybrid Centrality-Guided Graph Neural Networks for Influential Node Ranking

## Overview

This project proposes a **Hybrid Centrality-Guided Graph Neural Network (HGNN)** framework for identifying influential nodes in large-scale social networks. The model combines:

* **8 classical network centrality measures**
* **Graph Neural Network embeddings (GraphSAGE + GAT)**
* **XGBoost regression**
* **Epidemic spreading simulations (SIR, SIS, SEIR)**

to accurately rank nodes according to their influence in information diffusion and epidemic propagation.

The framework is evaluated on real-world benchmark networks from SNAP and demonstrates strong performance, outperforming traditional centrality-based approaches and standalone GNN models.


## Key Contributions

✅ Hybrid integration of handcrafted graph features and deep graph embeddings

✅ Multi-model epidemic simulation labels using:

* SIR
* SIS
* SEIR

✅ GraphSAGE-GAT architecture with residual connections

✅ XGBoost-based ranking refinement

✅ Cross-dataset zero-shot generalization

✅ Statistical significance testing with Bonferroni correction

✅ Extensive ablation and sensitivity studies

---

## Architecture

```text
Graph Network
      │
      ▼
8 Centrality Features
(Degree, Betweenness,
 Closeness, Katz,
 K-Shell, Clustering,
 VoteRank, Harmonic)
      │
      ▼
Hybrid GNN
(GraphSAGE → GAT → GraphSAGE)
      │
      ▼
Node Embeddings
      │
      ▼
Concatenate
(Centrality + Embeddings)
      │
      ▼
XGBoost Regressor
      │
      ▼
Influence Score Ranking
```

---

## Datasets

| Dataset  | Nodes |   Edges |
| -------- | ----: | ------: |
| Facebook | 4,039 |  88,234 |
| Email-EU |   986 |  16,064 |
| WikiVote | 7,066 | 100,736 |

Source: SNAP (Stanford Network Analysis Project)

---

## Centrality Features

The model uses the following 8 graph-theoretic features:

1. Degree Centrality
2. Betweenness Centrality
3. Closeness Centrality
4. Katz Centrality
5. K-Shell (Core Number)
6. Clustering Coefficient
7. VoteRank Score
8. Harmonic Centrality

---

## Epidemic Influence Labels

Ground-truth influence labels are generated using Monte-Carlo simulations of:

### SIR

Susceptible → Infected → Recovered

### SIS

Susceptible → Infected → Susceptible

### SEIR

Susceptible → Exposed → Infected → Recovered

Multiple transmission rates are used:

```python
β = [0.1, 0.2, 0.3]
γ = 0.10
σ = 0.15
```

Final influence scores are obtained by averaging results across all epidemic models.

---

## Hybrid GNN Model

### Graph Encoder

```text
Input Features (8)
        │
        ▼
GraphSAGE Layer
        │
        ▼
GAT Layer
        │
        ▼
GraphSAGE Layer
        │
        ▼
Residual Projection
        │
        ▼
32-D Embeddings
```

### Hyperparameters

| Parameter        | Value |
| ---------------- | ----- |
| Hidden Dimension | 64    |
| Output Dimension | 32    |
| Attention Heads  | 4     |
| Dropout          | 0.3   |
| Learning Rate    | 0.005 |
| Epochs           | 300   |
| Early Stopping   | 30    |

---

## Performance on Facebook Dataset

### Test Set Results

| Metric               |      Score |
| -------------------- | ---------: |
| Spearman Correlation | **0.9169** |
| Kendall Tau          | **0.7583** |
| R² Score             | **0.9228** |
| RMSE                 | **0.0466** |
| MAP                  | **0.9995** |
| NDCG@100             | **0.9744** |
| Jaccard@100          | **0.5038** |

---

## 5-Fold Cross Validation

| Metric   | Mean ± Std      |
| -------- | --------------- |
| Spearman | 0.9317 ± 0.0056 |
| Kendall  | 0.7819 ± 0.0096 |
| R²       | 0.9348 ± 0.0048 |
| MSE      | 0.0018 ± 0.0001 |
| NDCG@50  | 0.9642 ± 0.0042 |

---

## Baseline Comparison

| Method            |   Spearman |    Kendall |         R² |
| ----------------- | ---------: | ---------: | ---------: |
| Degree            |     0.7252 |     0.5309 |   -16.8383 |
| Betweenness       |     0.5194 |     0.3823 |   -17.4436 |
| Closeness         |     0.7441 |     0.5675 |    -6.1429 |
| Katz              |     0.7757 |     0.5774 |   -10.8496 |
| Harmonic          |     0.8012 |     0.6236 |    -0.3904 |
| GNN Only          |    -0.0266 |    -0.0168 |    -0.8593 |
| Centrality Only   |    -0.0287 |    -0.0180 |    -0.9831 |
| **Proposed HGNN** | **0.9169** | **0.7583** | **0.9228** |

---

## Cross-Dataset Generalization

### Zero-Shot Evaluation

| Dataset  | Spearman | Kendall |     R² |
| -------- | -------: | ------: | -----: |
| Facebook |   0.9169 |  0.7583 | 0.9228 |
| Email-EU |   0.7795 |  0.6277 | 0.8288 |
| WikiVote |   0.9288 |  0.7654 | 0.8984 |

The model demonstrates strong transferability across unseen graph structures.

---

## Statistical Significance Testing

Wilcoxon Signed-Rank Tests with Bonferroni correction were performed against all baselines.

Result:

```text
All comparisons significant
(p < 0.00625)
```

indicating the proposed framework significantly outperforms existing methods.

---

## Generated Outputs

The pipeline automatically generates:

### Figures

* Training Curve
* Predicted vs Actual Scatter Plot
* Label Distribution
* Feature Importance Analysis
* Method Comparison
* Correlation Heatmap
* Top-30 Centrality Heatmap
* Influential Node Subgraph
* Ranking Comparison
* Top-20 Influential Nodes
* Cross-Validation Results

### Files

```text
dataset_summary.csv
ranked_nodes.csv
main_test_metrics.csv
baseline_comparison.csv
cv_results.csv
ablation_study.csv
significance_tests.csv
runtime_analysis.csv
multi_dataset_results.csv
hybrid_gnn_weights.pt
xgboost_model.pkl
```

---

## Installation

```bash
pip install torch
pip install torch_geometric
pip install torch_scatter
pip install torch_sparse
pip install torch_cluster
pip install torch_spline_conv
pip install xgboost
pip install networkx
pip install pandas
pip install numpy
pip install scipy
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install joblib
```

---

## Usage

Run the notebook:

```bash
jupyter notebook Hybrid_GNN_Influential_Node_Ranking.ipynb
```

or

```bash
python main.py
```

The pipeline will:

1. Download datasets
2. Compute centrality features
3. Generate epidemic labels
4. Train HGNN
5. Train XGBoost
6. Evaluate performance
7. Produce visualizations
8. Save trained models

---

## Results Summary

The proposed Hybrid Centrality-Guided Graph Neural Network achieves:

* **Spearman Correlation:** 0.9169
* **Kendall Tau:** 0.7583
* **R² Score:** 0.9228
* **MAP:** 0.9995

while maintaining strong zero-shot generalization across multiple real-world network datasets.

---

## Citation

```bibtex
@article{kennedy2026hgnn,
  title={Hybrid Centrality-Guided Graph Neural Networks for Influential Node Ranking},
  author={Kennedy Sandhiya},
  year={2026}
}
```

---

## Author

**Sandhiya Kennedy**
B.Tech Artificial Intelligence & Data Science
Amrita Vishwa Vidyapeetham, India

Email: [sandhiyakennedy3008@gmail.com](mailto:sandhiyakennedy3008@gmail.com)

⭐ If you find this project useful, consider starring the repository.
