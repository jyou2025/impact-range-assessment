## Run Examples
The two examples use synthetic datasets (linear and nonlinear) and corresponding models to validate IRA, as reported in the paper.  

### Example Requirements
To run the example scripts, you need additional libraries: scikit-learn and matplotlib. 
```bash
pip install scikit-learn matplotlib
```

**The examples were developed using Python 3.8 (NumPy 1.24, pandas 1.5, scikit-learn 1.3, Matplotlib 3.7) and have been tested on Python environments including Python 3.9 (NumPy 1.26, pandas 2.0, scikit-learn 1.5, Matplotlib 3.9) and Python 3.12 (NumPy 2.2, pandas 2.3, scikit-learn 1.8, Matplotlib 3.10). Compatibility issues may occur when NumPy and pandas are installed in binary-incompatible combinations.**

### Linear IRA Example
Based on a linear regression model using the linear dataset, perform a single-execution IRA with different parameter settings   
**The commands below assume the working directory is "linear_example".**
```bash
python linear_ira.py
```
### Nonlinear IRA Example
Based on a Random Forest regression model using the nonlinear dataset, perform a single-execution IRA with different parameter settings, and a repeated IRA with different numbers of repeats  
**The commands below assume the working directory is "nonlinear_example".**
```bash
python nonlinear_ira.py
```
