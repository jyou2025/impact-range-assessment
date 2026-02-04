# irapy

**irapy** is a Python library for implementing single-execution Impact Range Assessment (IRA) and repeated IRA analyses.

## Installation
```bash
pip install irapy
```

## Requirements
### irapy Requirements
irapy depends on the following Python libraries:
- NumPy
- pandas
- joblib

These dependencies are installed automatically when using pip install irapy.

**irapy was developed using Python 3.8, NumPy 1.24.3, and pandas 1.5.3, and has been validated on modern Python environments including Python 3.9 (NumPy 1.26, pandas 2.0) and Python 3.12 (NumPy 2.2, pandas 2.3). Compatibility issues may occur if NumPy and pandas are installed in binary-incompatible combinations.**

## Usage
### Run irapy
You can apply either a single-execution or a repeated IRA to a trained regression model using the corresponding dataset.
```Python
from irapy import single_ira, repeated_ira

# single-execution IRA
single_ira_result = single_ira(input_data=X, model=trained_model)
print(single_ira_result)

# repeated IRA
repeated_ira_result = repeated_ira(input_data=X, model=trained_model, n_repeats=50)
print(repeated_ira_result)
```
### Arguments
- `input_data`: predictor pandas DataFrame (i.e., the training predictor matrix, often denoted as X_train; **use the original, unscaled predictors when a scaler is applied**); all predictors must be continuous numeric variables.
- `model`: trained model (object with '.predict()', tested with 'scikit-learn' models) or a callable function
- `scaler`: fitted scaler (**optional**, e.g., 'StandardScaler' / 'MinMaxScaler' in Python)  
- `num_interpol`: number of interpolated points (default: 100)
- `num_background_samples`: number of background observations (default: 200)
- `random_state`: seed for reproducibility  (default: 42)
- `sorted_output`: whether to sort results by IRA value   (default: False)
- `n_repeats` (repeated IRA): number of repeated times  (default: 50)
- `n_jobs` (repeated IRA): number of parallel jobs to run   (default: 1)

## Project Homepage
https://github.com/jyou2025/impact-range-assessment
