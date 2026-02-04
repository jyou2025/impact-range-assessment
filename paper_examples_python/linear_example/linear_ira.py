from irapy import single_ira
import pickle
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression

# -------------------------------
# Step 1: Setup and Load Data
# -------------------------------
print("\n" + "="*60)
print("STEP 1: SETUP AND LOAD DATA")
print("="*60)

script_dir = os.path.dirname(os.path.abspath(__file__))

np.set_printoptions(suppress=True)

# Load dataset
data_path = os.path.join(script_dir, "linear_data.csv")
data = pd.read_csv(data_path)
print(f"Loaded dataset from '{data_path}' — shape: {data.shape}")


# -------------------------------
# Step 2: Train and Save Model
# -------------------------------
print("\n" + "="*60)
print("STEP 2: TRAIN LINEAR REGRESSION MODEL")
print("="*60)

X_train, y_train = data.iloc[:, :-1], data.iloc[:, -1]
model = LinearRegression().fit(X_train, y_train)

model_path = os.path.join(script_dir,"linear_model.pkl")
with open(model_path, "wb") as file:
    pickle.dump(model, file)

print(f"Model saved to '{model_path}'")


# -------------------------------
# Step 3: Run IRA Calculation
# -------------------------------
print("\n" + "="*60)
print("STEP 3: SINGLE-EXECUTION IRA")
print("="*60)

def ira_cal():
    column_name = ['N. background samp', 'N. interpl'] + list(data.columns[:-1])
    result = []
    step = 1

    for n in [50, 100, 200, 500]:
        for i in [50, 100, 200, 500]:
            print(f"→ Step {step}: background_samples={n}, interpolation_points={i}")
            ira_result = single_ira(
                data.iloc[:, :-1],
                model,
                num_interpol=i,
                num_background_samples=n,
                random_state=42
            )
            result.append([n, i] + ira_result.iloc[:, 1].tolist())
            step += 1

    result_df = pd.DataFrame(result, columns=column_name)
    return result_df

ira_cal_result = ira_cal()

# -------------------------------
# Step 4: Output and Save Results
# -------------------------------
print("\n" + "="*60)
print("STEP 4: SAVE IRA RESULTS")
print("="*60)

print("\n--- Final Single IRA Results ---")
print(ira_cal_result)

ira_result_path = os.path.join(script_dir, "linear_ira_result.csv")
ira_cal_result.to_csv(ira_result_path, index=False)
print(f"Single IRA results saved to '{ira_result_path}'")
