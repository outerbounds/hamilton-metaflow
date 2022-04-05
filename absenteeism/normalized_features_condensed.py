from hamilton.function_modifiers import parametrized_input
import pandas as pd

NUMERIC_FEATURES = ['age', 'height', 'weight', 'body_mass_index']
FEATURE_MEANS = {k: (f'{k}_mean', f'Average of {k}') for k in NUMERIC_FEATURES}
FEATURE_STD_DEV = {k: (f'{k}_std_dev', f'Standard deviation of {k}') for k in NUMERIC_FEATURES}


@parametrized_input(parameter='feature', variable_inputs=FEATURE_MEANS)
def mean_computer(feature: pd.Series) -> pd.Series:
    return feature.mean()


@parametrized_input(parameter='feature', variable_inputs=FEATURE_STD_DEV)
def std_dev_computer(feature: pd.Series) -> pd.Series:
    return feature.std()


def age_zero_mean(age: pd.Series, age_mean: float) -> pd.Series:
    """Zero mean of age"""
    return age - age_mean


def height_zero_mean(height: pd.Series, height_mean: float) -> pd.Series:
    """Zero mean of height"""
    return height - height_mean


def weight_zero_mean(weight: pd.Series, weight_mean: float) -> pd.Series:
    """Zero mean of weight"""
    return weight - weight_mean


def body_mass_index_zero_mean(body_mass_index: pd.Series, body_mass_index_mean: float) -> pd.Series:
    """Zero mean of body_mass_index"""
    return body_mass_index - body_mass_index_mean


def age_zero_mean_unit_variance(age_zero_mean: pd.Series, age_std_dev: pd.Series) -> pd.Series:
    """Zero mean unit variance value of age"""
    return age_zero_mean / age_std_dev


def height_zero_mean_unit_variance(height_zero_mean: pd.Series, height_std_dev: pd.Series) -> pd.Series:
    """Zero mean unit variance value of height"""
    return height_zero_mean / height_std_dev


def weight_zero_mean_unit_variance(weight_zero_mean: pd.Series, weight_std_dev: pd.Series) -> pd.Series:
    """Zero mean unit variance value of weight"""
    return weight_zero_mean / weight_std_dev


def body_mass_index_zero_mean_unit_variance(body_mass_index_zero_mean: pd.Series,
                                            body_mass_index_std_dev: pd.Series) -> pd.Series:
    """Zero mean unit variance value of body_mass_index"""
    return body_mass_index_zero_mean / body_mass_index_std_dev


# TODO: enable hamilton to do this multiple column parameterization?
# FEATURE_ZERO_MEAN = {(k, f'{k}_mean'): (f'{k}_zero_mean', f'Zero mean of {k}') for k in NUMERIC_FEATURES}
# FEATURE_ZERO_MEAN_UNIT_VAR = {
#     (f'{k}_zero_mean', f'{k}_std_dev'): (f'{k}_zero_mean_unit_var', f'Zero mean unit variance value of {k}')
#     for k in NUMERIC_FEATURES
# }
# 
# 
# @parametrized_input('feature', 'feature_mean', variable_inputs=FEATURE_ZERO_MEAN)
# def zero_mean_computer(feature: pd.Series, feature_mean: pd.Series) -> pd.Series:
#     return feature - feature_mean
# 
# 
# @parametrized_input('feature_mean', 'feature_std_dev', variable_inputs=FEATURE_ZERO_MEAN_UNIT_VAR)
# def zero_mean_unit_var_computer(feature_mean: pd.Series, feature_std_dev: pd.Series) -> pd.Series:
#     return feature_mean / feature_std_dev
