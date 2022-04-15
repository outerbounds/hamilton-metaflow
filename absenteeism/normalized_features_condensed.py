from hamilton.function_modifiers import parameterized_inputs
import pandas as pd

NUMERIC_FEATURES = ['age', 'height', 'weight', 'body_mass_index', 'transportation_expense',
                    'distance_from_residence_to_work', 'service_time', 'work_load_average_per_day',
                    'hit_target']


@parameterized_inputs(**{f'{k}_mean': dict(feature=k) for k in NUMERIC_FEATURES})
def mean_computer(feature: pd.Series) -> pd.Series:
    """Average of {feature}"""
    return feature.mean()


@parameterized_inputs(**{f'{k}_std_dev': dict(feature=k) for k in NUMERIC_FEATURES})
def std_dev_computer(feature: pd.Series) -> pd.Series:
    """Standard deviation of {feature}"""
    return feature.std()


@parameterized_inputs(**{f'{k}_zero_mean': dict(feature=k, feature_mean=f'{k}_mean') for k in NUMERIC_FEATURES})
def zero_mean_computer(feature: pd.Series, feature_mean: pd.Series) -> pd.Series:
    """Creates zero mean value of {feature} by subtracting {feature_mean}"""
    return feature - feature_mean


@parameterized_inputs(**{f'{k}_zero_mean_unit_var': dict(feature_mean=f'{k}_mean', feature_std_dev=f'{k}_std_dev')
                        for k in NUMERIC_FEATURES})
def zero_mean_unit_var_computer(feature_mean: pd.Series, feature_std_dev: pd.Series) -> pd.Series:
    """Computes zero mean unit variance of {feature_mean} with {feature_std_dev} to create {output_name}"""
    return feature_mean / feature_std_dev
