## Run Examples
The two examples use synthetic datasets (linear and nonlinear) and models to validate IRA, as reported in the paper.  

### Example Requirements
To run the example scripts, you need additional libraries: scikit-learn, matplotlib 
```bash
pip install scikit-learn matplotlib
```
**Tested Environment:**
Python 3.8.17 - scikit-learn 1.3.0, Matplotlib 3.7.2

### Linear IRA Example
Based on a linear regression model using the linear data, perform a single-execution IRA with different parameter settings   
**The commands below assume the working directory is "linear_example".**
```bash
python linear_ira.py
```
### Nonlinear IRA Example
Based on a Random Forest regression model using the nonlinear data, perform a single-run IRA with different parameter settings, and a repeated IRA with different numbers of repeats  
**The commands below assume the working directory is "nonlinear_example".**
```bash
python nonlinear_ira.py
```
