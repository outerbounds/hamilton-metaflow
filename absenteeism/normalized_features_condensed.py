"""
This modules shows a "super user" way to write the same features found in normalized_features.py.
It utilizes some Hamilton function decorators to write less code that is more dense.

Why might you use this approach?

- you and the people going to be reading your code are quite familiar with python.
- e.g. you understand python comprehensions & what ** does.
- you want less code to manage and update when thigns change

What is the cost?

- it's less obvious to a new user to the code base, how a particular feature was produced.
- it's a little harder to map & debug outputs to where the code lives to create that output.

So you should determine whether writing feature like this is right for you, depending on 
your environment (who is going to own & modify the code after you write it!). 
"""

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
