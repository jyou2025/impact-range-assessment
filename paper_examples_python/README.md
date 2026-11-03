# IRA Paper Examples

This directory contains the Python notebooks used to reproduce the analyses reported in the revised manuscript:

> **Impact Range Assessment (IRA): An Interpretable Sensitivity Measure for Machine Learning Regression Modelling**

Preprint: [arXiv:2602.05239](https://arxiv.org/abs/2602.05239)

## Notebooks

| Notebook | Analysis reproduced |
|---|---|
| `synthetic_linear_example.ipynb` | Synthetic linear validation: Table 1 and Table S1 |
| `synthetic_nonlinear_example.ipynb` | Synthetic nonlinear validation: Tables 2–3 and Table S2 |
| `Benchmark_California_Housing_example.ipynb` | California Housing benchmark: Table 4, Table S3, and Figures S1 and S3 |
| `Benchmark_Concrete_Compressive_Strength_example.ipynb` | Concrete Compressive Strength benchmark: Table 5, Table S4, and Figures S2 and S4 |
| `Runtime_comparison.ipynb` | Single-execution IRA runtime benchmark: Table S6 |

The confidential feed-mill dataset used in the Pellet Durability Index case study cannot be publicly distributed and is therefore not included in this repository.  
Runtime measurements can vary across systems and between executions because of differences in hardware, software environments, and system load. The values saved in `Runtime_comparison.ipynb` correspond to those reported in Table S6 of the revised manuscript.  

## Software environment

The analyses reported in the revised manuscript were conducted using:  

- Python 3.12
- [`irapy` 0.1.2](https://pypi.org/project/irapy/)
- NumPy 2.2.5
- pandas 2.3.3
- scikit-learn 1.8.0
- Matplotlib 3.10.9
- SHAP 0.51.0

## Data requirements

### Synthetic datasets

The synthetic linear and nonlinear datasets are generated directly within their respective notebooks using scikit-learn. No external data files are required.

### California Housing dataset

The California Housing dataset is obtained using `sklearn.datasets.fetch_california_housing`.   

### Concrete Compressive Strength dataset

The Concrete Compressive Strength dataset is not included in this repository. It can be manually downloaded from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength).  
