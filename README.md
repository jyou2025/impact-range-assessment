[![PyPI](https://img.shields.io/pypi/v/irapy?label=PyPI&color=brightgreen)](https://pypi.org/project/irapy/)
[![arXiv](https://img.shields.io/badge/arXiv-2602.05239-b31b1b.svg)](https://arxiv.org/abs/2602.05239)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18675680.svg)](https://doi.org/10.5281/zenodo.18675680)


# Impact Range Assessment (IRA)

## Introduction

**Impact Range Assessment (IRA)** is a model-agnostic sensitivity measure for fitted regression models with numeric predictors. IRA quantifies how much model predictions vary when each predictor is varied across its observed range while the remaining predictors are held at sampled background observations. IRA can be performed as either a single-execution analysis for a rapid estimate of model sensitivity or a repeated analysis that characterizes variability associated with background-observation sampling. The resulting IRA values are expressed in the same units as the model output, providing an intuitive basis for comparing and ranking the sensitivity of model predictions to individual predictors. IRA evaluates predictors using a one-at-a-time perturbation strategy and does not explicitly isolate interaction effects. This repository contains the Python implementation of IRA, available as [`irapy`](https://pypi.org/project/irapy/), and the notebooks used to reproduce the synthetic validation, benchmark analyses, interpretability comparisons, and runtime results reported in the revised manuscript. A revised version of the manuscript has been resubmitted to *MethodsX*. The latest preprint is available at [arXiv:2602.05239](https://arxiv.org/abs/2602.05239).


## Scope

1. IRA is primarily designed for regression models with continuous numeric predictors and discrete numeric predictors for which interpolation is mathematically meaningful. Categorical predictors are not directly supported by the current implementation.
2. IRA is suitable for regression models with outputs that have direct practical interpretations. Applying IRA to bounded outputs such as probabilities may result in values that are less straightforward to interpret in practical terms.
3. IRA values depend on the trained model and input data and are not intended for extrapolation beyond the predictor ranges observed in the input data.
4. IRA values depend on the predictor ranges and distributions present in the dataset and should not be compared across datasets with different predictor distributions or ranges, even when the same model is used.
5. IRA values should be interpreted with caution when predictors contain extreme values or when models are particularly sensitive near the edges of the observed ranges. In such cases, restricting the analysis to a specific and practically relevant region of the range (e.g., the 25th–75th percentile range) may provide more interpretable results.
6. IRA uses a one-at-a-time perturbation strategy and does not explicitly isolate multivariable interaction effects. It may also evaluate uncommon or unrealistic predictor combinations when predictors are strongly correlated.
7. IRA settings, including the number of interpolation points, the number of background observations, and the number of repeats, may require adjustment for different datasets and regression models.

## Links
#### **IRA functions in Python (irapy):** ➡️ [python](./python)
#### **IRA paper examples in Python:** ➡️ [paper_examples_python](./paper_examples_python)

## Project Structure
```
impact-range-assessment/
├── python/
│   ├── irapy/
│   │   ├── __init__.py
│   │   └── irapy.py
│   ├── pyproject.toml
│   ├── LICENSE
│   └── README.md
│
├── paper_examples_python/
│   ├── synthetic_linear_example.ipynb
│   ├── synthetic_nonlinear_example.ipynb
│   ├── Benchmark_California_Housing_example.ipynb
│   ├── Benchmark_Concrete_Compressive_Strength_example.ipynb
│   ├── Runtime_comparison.ipynb
│   └── README.md
│
├── .gitignore
├── README.md
└── LICENSE             
```

## License
This project is licensed under the [MIT License](LICENSE).
