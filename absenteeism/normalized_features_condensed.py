from hamilton.function_modifiers import parametrized_input
import pandas as pd

NUMERIC_FEATURES = ['age', 'height', 'weight', 'body_mass_index', 'transportation_expense',
                    'distance_from_residence_to_work', 'service_time', 'work_load_average_per_day',
                    'hit_target']
FEATURE_MEANS = {k: (f'{k}_mean', f'Average of {k}') for k in NUMERIC_FEATURES}
FEATURE_STD_DEV = {k: (f'{k}_std_dev', f'Standard deviation of {k}') for k in NUMERIC_FEATURES}


@parametrized_input(parameter='feature', variable_inputs=FEATURE_MEANS)
def mean_computer(feature: pd.Series) -> pd.Series:
    return feature.mean()


@parametrized_input(parameter='feature', variable_inputs=FEATURE_STD_DEV)
def std_dev_computer(feature: pd.Series) -> pd.Series:
    return feature.std()


# TODO: enable hamilton to do this multiple column parameterization?
FEATURE_ZERO_MEAN = {(k, f'{k}_mean'): (f'{k}_zero_mean', f'Zero mean of {k}') for k in NUMERIC_FEATURES}
FEATURE_ZERO_MEAN_UNIT_VAR = {
    (f'{k}_zero_mean', f'{k}_std_dev'): (f'{k}_zero_mean_unit_var', f'Zero mean unit variance value of {k}')
    for k in NUMERIC_FEATURES
}


@parametrized_input('feature', 'feature_mean', variable_inputs=FEATURE_ZERO_MEAN)
def zero_mean_computer(feature: pd.Series, feature_mean: pd.Series) -> pd.Series:
    return feature - feature_mean


@parametrized_input('feature_mean', 'feature_std_dev', variable_inputs=FEATURE_ZERO_MEAN_UNIT_VAR)
def zero_mean_unit_var_computer(feature_mean: pd.Series, feature_std_dev: pd.Series) -> pd.Series:
    return feature_mean / feature_std_dev
