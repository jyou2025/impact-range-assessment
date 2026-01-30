# Impact Range Assessment (IRA)

**Impact Range Assessment (IRA)** is an interpretable sensitivity measure for regression modeling.

## Installation
```bash
pip install irapy
```
## Requirements
### IRA Requirements
IRA depends on the following Python libraries:
- Numpy
- pandas
- joblib

These dependencies are installed automatically when using pip install irapy.

**Tested Environment:**
Python 3.8.17 - Numpy 1.24.3, pandas 1.5.3, joblib 1.2.0  

## Usage
### Run IRA
You can apply either a single-execution or a repeated IRA to a trained regression model using the corresponding dataset.
```Python
from ira import single_ira, repeated_ira

# single-execution IRA
result = single_ira(input_data=X, model=trained_model)
print(result)

# repeated IRA
result_repeated = repeated_ira(input_data=X, model=trained_model, n_repeats=50)
print(result_repeated)
```
Arguments
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
