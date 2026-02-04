from irapy import single_ira, repeated_ira
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestRegressor

# -------------------------------
# Step 1: Setup and Load Resources
# -------------------------------
print("\n" + "="*70)
print("STEP 1: SETUP AND LOAD DATA")
print("="*70)

script_dir = os.path.dirname(os.path.abspath(__file__))

np.set_printoptions(suppress=True)

# Load dataset
data = pd.read_csv(os.path.join(script_dir, "nonlinear_data.csv"))
print(f"Data loaded from 'nonlinear_data.csv' — shape: {data.shape}")

# -------------------------------
# Step 2: Train and Save Model
# -------------------------------
print("\n" + "="*60)
print("STEP 2: TRAIN RANDOM FOREST MODEL")
print("="*60)

X, y = data.iloc[:, :-1], data.iloc[:, -1]
rf = RandomForestRegressor(random_state=0).fit(X, y)
r2 = rf.score(X, y)
print(f"Model trained — R² score on training data: {r2:.4f}")

model_path = os.path.join(script_dir, "nonlinear_rf_model.pkl")
with open(model_path, "wb") as file:
    pickle.dump(rf, file)
print(f"Model saved to '{model_path}'")

# ------------------------------------------
# Step 3: Run Single-execution IRA (No CI)
# ------------------------------------------
print("\n" + "="*70)
print("STEP 3: SINGLE-EXECUTION IRA")
print("="*70)

def ira_cal():
    column_name = ['N. background samp', 'N. interpl'] + list(data.columns[:-1])
    result = []
    step_counter = 1
    for n in [50, 100, 200, 500]:
        for i in [50, 100, 200, 500]:
            print(f"[Single IRA] Step {step_counter}: n_background={n}, n_interpol={i}")
            ira_result = single_ira(data.iloc[:, :-1], rf, num_interpol=i, num_background_samples=n, random_state=42)
            result.append([n, i] + ira_result.iloc[:, 1].tolist())
            step_counter += 1
    result_df = pd.DataFrame(result, columns=column_name)
    return result_df

ira_cal_result = ira_cal()
print("\n--- Final IRA Result Table (Single execution) ---")
print(ira_cal_result)
ira_cal_result.to_csv('nonlinear_ira_single.csv', index=False)
print("Single IRA results saved to 'nonlinear_ira_single.csv'")

# ------------------------------------------
# Step 4: Run Repeated IRA (with CI)
# ------------------------------------------
print("\n" + "="*70)
print("STEP 4: REPEATED IRA (with Confidence Intervals)")
print("="*70)

def run_repeated():
    delta_result = pd.DataFrame()
    delta_result['predictor'] = data.iloc[:, :-1].columns
    df_list = [pd.Series(list(data.columns[:-1]), name='predictor')]

    for i in [10, 30, 50, 70, 90]:
        print(f"[Repeated IRA] Running {i} repeats...")
        repeated_result = repeated_ira(
            input_data=data.iloc[:, :-1],
            model=rf,
            n_repeats=i,
            random_state=42,
            n_jobs=15
        )
        delta_result[f'{i}'] = repeated_result['ci_upper'] - repeated_result['ci_lower']
        repeated_result = repeated_result.iloc[:, 1:].copy()
        repeated_result = repeated_result.rename(columns={
            "mean": f"mean_{i}",
            "ci_lower": f"ci_lower_{i}",
            "ci_upper": f"ci_upper_{i}"
        })
        print(repeated_result)
        df_list.append(repeated_result)

    df_result = pd.concat(df_list, axis=1)
    return df_result, delta_result

repeated_ira_result, ci_result = run_repeated()
print("\n--- Final Repeated IRA Result Table ---")
print(repeated_ira_result)
repeated_ira_result.to_csv('nonlinear_ira_repeated.csv', index=False)
print("Repeated IRA results saved to 'nonlinear_ira_repeated.csv'")

# ------------------------------------------
# Step 5: Plot Average CI Width vs. Repeats
# ------------------------------------------
print("\n" + "="*70)
print("STEP 5: PLOT CI WIDTH VS. NUMBER OF REPEATS")
print("="*70)

print("\n--- CI Result Table ---")
print(ci_result)

def plot_run():
    x = ci_result.columns[1:]
    y = ci_result.iloc[:, 1:].mean(axis=0)
    
    plt.figure()
    plt.scatter(x, y, color='green')
    plt.plot(x, y, color='green')
    plt.xlabel('Number of Repeats', fontsize=14)
    plt.ylabel('Average CI Width', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('nonlinear_repeats.jpg', dpi=300)
    print("Plot saved as 'nonlinear_repeats.jpg'")

plot_run()
