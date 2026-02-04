import pandas as pd
import numpy as np
from joblib import Parallel, delayed


def single_ira(input_data, model, scaler=None, num_interpol=100, num_background_samples=200, random_state=42, sorted_output=False):
    """
    Single-execution IRA

    Parameters:
    - input_data: predictor pandas DataFrame (i.e., the training predictor matrix, often denoted as X_train). When using a scaler, this must be the original predictors used to fit the scaler. All predictors must be continuous numeric.
    - model: A model object with a .predict() method, trained on data with the same structure as input_data.
             Alternatively, a callable function can be used, such as:
                 def func(a, b):
                     y = 3 * a + 1 * b
                     return y
    - scaler: Optional fitted scaler (e.g., StandardScaler or MinMaxScaler). Must be trained on a DataFrame with the
              same structure as input_data. If None, no scaling is applied.
    - num_interpol: Number of interpolation points between min and max of the focus predictor (default: 100).
    - num_background_samples: Number of background observations randomly drawn from the dataset. (default: 200).
    - random_state: Seed for reproducibility when method is 'random'. Integer or None (default: 42).
    - sorted_output: Whether to sort the result by IRA values in descending order (default: False).

    Returns:
    - A DataFrame with predictor names and their corresponding Impact Range Assessment (IRA).
    """

    if not isinstance(num_interpol, int) or num_interpol < 2:
        raise ValueError("num_interpol must be an integer ≥ 2.")

    if not isinstance(num_background_samples, int) or num_background_samples <= 0:
        raise ValueError("num_background_samples must be a positive integer.")

    if random_state is not None and not isinstance(random_state, int):
        raise ValueError("random_state must be an integer or None.")

    # Calculate descriptive statistics for training input data
    predictor_summary = input_data.describe()

    # Create an empty list to store results
    output_range_list = []

    # Iterate all predictors
    for focus_predictor in predictor_summary.columns:

        if predictor_summary.loc['min', focus_predictor] == predictor_summary.loc['max', focus_predictor]:
            output_range_list.append([focus_predictor, 0.0])
            continue

        else:

            #  Create a number of values for the focus predictor between its minimum and maximum
            created_predictor_values = np.linspace(
                predictor_summary.loc['min', focus_predictor],
                predictor_summary.loc['max', focus_predictor],
                num_interpol)

            # random sampling with seed or without seed
            if isinstance(random_state, int):
                background_samples = input_data.sample(n=num_background_samples, replace=True, random_state=random_state)
            elif random_state is None:
                background_samples = input_data.sample(n=num_background_samples, replace=True)
            else:
                raise ValueError("random_state must be an integer or None.")

            # Reset index for the sampled data
            background_samples = background_samples.reset_index(drop=True)

            # Add one index for grouping after making prediction
            background_samples['Index'] = np.arange(num_background_samples)

            # Create replicates for each observation in the sampled data
            # Number of replicates is equal to the number of interpolating points for focus predictors
            background_samples_replicates = background_samples.loc[
                background_samples.index.repeat(num_interpol)].reset_index(drop=True)

            # Create dataset including non-focus predictors and the focus predictor
            observations = background_samples_replicates.assign(
                **{focus_predictor: np.tile(created_predictor_values, num_background_samples)})

            # Excluding the added "index" column for prediction in the next step
            x_input = observations.drop(columns=['Index'])

            # Use the scaler and model to predict the output, which is an array (num_interpol,)
            if hasattr(model, 'predict'):
                if scaler is not None:
                    observations['Prediction'] = model.predict(pd.DataFrame(scaler.transform(x_input),
                                                                            columns=x_input.columns))
                else:
                    observations['Prediction'] = model.predict(x_input)
            elif callable(model):
                observations['Prediction'] = [model(*row) for row in x_input.itertuples(index=False, name=None)]
            else:
                raise TypeError("Unsupported model type.")

            # Find the maximum and minimum for each sampled observation
            grouped = observations.loc[:, ['Index', 'Prediction']].copy().groupby('Index')['Prediction']
            max_per_group = grouped.max()
            min_per_group = grouped.min()

            # Calculate the IRA values and then average them
            ira_values = (max_per_group - min_per_group).mean()

            # Save the result for the predictor
            output_range_list.append([focus_predictor, ira_values])

    result = pd.DataFrame(output_range_list, columns=['predictor', 'IRA value'])

    if sorted_output:
        # sort the result in ascending order based on IRA values.
        result = result.sort_values(by='IRA value', ascending=False).reset_index(drop=True)

    # Return to the final result dataframe
    return result


def repeated_ira(input_data, model, scaler=None, num_interpol=100, num_background_samples=200, random_state=42,
                 n_repeats=50, n_jobs=1, sorted_output=False):
    """
    Repeated IRA with confidence intervals

    Parameters:
    - input_data: predictor pandas DataFrame (i.e., the training predictor matrix, often denoted as X_train). When using a scaler, this must be the original predictors used to fit the scaler.
    - model: A model object with a .predict() method, trained on data with the same structure as input_data.
             Alternatively, a callable function can be used, such as:
                 def func(a, b):
                     y = 3 * a + 1 * b
                     return y
    - scaler: Optional fitted scaler (e.g., StandardScaler or MinMaxScaler). Must be trained on a DataFrame with the
              same structure as input_data. If None, no scaling is applied.
    - num_interpol: Number of interpolation points between min and max of the focus predictor (default: 100).
    - num_background_samples: Number of background observations randomly drawn from the dataset (default: 200).
    - random_state: Seed for reproducibility (default: 42).
    - n_repeats: Number of repeated times (default: 50).
    - n_jobs: Number of parallel jobs to run (default: 1).
    - sorted_output: If True, sorts the output by mean IRA values in descending order (default: False).

    Returns:
    - DataFrame with predictor names, mean IRA, and 95% CI (lower, upper).
    """

    if not isinstance(n_repeats, int) or n_repeats <= 0:
        raise ValueError("n_repeats must be a positive integer.")

    if not isinstance(n_jobs, int) or n_jobs == 0:
        raise ValueError("n_jobs must be a non-zero integer.")
                     
    # Create a list of seeds for reproducible random sampling per iteration
    seeds = [int(s) for s in np.random.RandomState(random_state).randint(0, 1_000_000, size=n_repeats)]

    # Run IRA in parallel using joblib
    results = Parallel(n_jobs=n_jobs)(
        delayed(single_ira)(
            input_data=input_data,
            model=model,
            scaler=scaler,
            num_interpol=num_interpol,
            num_background_samples=num_background_samples,
            random_state=seed,
            sorted_output=False
        ) for seed in seeds
    )

    # Extract IRA values and calculate CI
    ira_array = np.array([r['IRA value'].values for r in results])
    mean = ira_array.mean(axis=0)
    ci_lower = np.percentile(ira_array, 2.5, axis=0)
    ci_upper = np.percentile(ira_array, 97.5, axis=0)

    result = pd.DataFrame({
        'predictor': input_data.columns,
        'mean': mean,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    })

    if sorted_output:
        result = result.sort_values(by='mean', ascending=False).reset_index(drop=True)

    return result
