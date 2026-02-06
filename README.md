# Impact Range Assessment (IRA)

## Introduction
**Impact Range Assessment (IRA)** is designed for regression models with continuous numeric predictors and quantifies their potential impact across the predictors’ observed data ranges in data-driven models. It is applicable to both linear and nonlinear models, while partially capturing interaction effects. The method can be implemented either as a single-execution procedure, offering a fast estimation of predictor impact, or as a repeated procedure, which assesses the stability of the estimates. Overall, IRA provides a simple and intuitive way to interpret and rank the impact of predictors on the response variable. 
This repository includes an implementation of IRA in **Python**, as well as examples using synthetic datasets presented in the paper.
The paper describing this method has been submitted to *MethodsX*, and a preprint version is available on *arXiv*. Please see the preprint here: [arXiv:2602.05239](https://arxiv.org/abs/2602.05239)

## Scope
1. IRA works with regression models using continuous numeric predictors. For categorical or discrete numeric predictors, IRA may be applied to the observed values rather than interpolated ranges.
2. IRA is suitable for regression models with outputs that have direct practical interpretations. Applying IRA to regressions like logistic regression (probabilistic outputs) may not have a clear practical interpretation.
3. IRA values depend on the trained model and training data and are not intended for extrapolation beyond the predictor ranges in the training data.
4. IRA values depend on the predictor ranges present in the dataset and should not be compared across datasets with different predictor distributions or ranges, even when the same model is used.
5. IRA values should be interpreted with caution when predictors contain extreme values or when models are particularly sensitive to extreme values. In such cases, restricting the analysis to a specific region of the range (e.g., the 25th–75th percentile) may provide more interpretable results.
6. IRA settings, including the number of interpolation points and the number of background observations, may require tuning.

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
|
├── paper_examples_python/            # validation of IRA using two synthetic datasets reported in the paper   
│   ├── linear_example/               # IRA example on linear model
|   |   ├──linear_data.csv
|   |   ├──linear_data_summary.csv
|   |   └──linear_ira.py
|   |
│   ├── nonlinear_example/            # IRA example on nonlinear model
|   |   ├──nonlinear_data.csv
|   |   ├──nonlinear_data_summary.csv
|   |   └──nonlinear_ira.py            
│   └── README.md
│
|── .gitignore
├── README.md               
└── LICENSE                  
```

## Citation
```bibtex
@article{you2026ira,
  title={Impact Range Assessment (IRA): An interpretable sensitivity measure for regression modeling},
  author={You, Jihao and Tulpan, Dan and Diao, Jiaojiao and Ellis, Jennifer L.},
  journal={arXiv preprint arXiv:2602.05239},
  year={2026}
}
```

## License
This project is licensed under the [MIT License](LICENSE).
