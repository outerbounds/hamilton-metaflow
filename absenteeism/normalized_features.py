import pandas as pd


def height_mean(height: pd.Series) -> float:
    """Average of height"""
    return height.mean()


def height_zero_mean(height: pd.Series, height_mean: float) -> pd.Series:
    """Zero mean of height"""
    return height - height_mean


def height_std_dev(height: pd.Series) -> pd.Series:
    """Standard deviation of height."""
    return height.std()


def height_zero_mean_unit_variance(height_zero_mean: pd.Series, height_std_dev: pd.Series) -> pd.Series:
    """Zero mean unit variance value of height"""
    return height_zero_mean / height_std_dev


def weight_mean(weight: pd.Series) -> float:
    """Average weight"""
    return weight.mean()


def weight_zero_mean(weight: pd.Series, weight_mean: float) -> pd.Series:
    """Zero mean of weight"""
    return weight - weight_mean


def weight_std_dev(weight: pd.Series) -> pd.Series:
    """Standard deviation of weight."""
    return weight.std()


def weight_zero_mean_unit_variance(weight_zero_mean: pd.Series, weight_std_dev: pd.Series) -> pd.Series:
    """Zero mean unit variance value of weight"""
    return weight_zero_mean / weight_std_dev


def body_mass_index_mean(body_mass_index: pd.Series) -> float:
    """Average body_mass_index"""
    return body_mass_index.mean()


def body_mass_index_zero_mean(body_mass_index: pd.Series, body_mass_index_mean: float) -> pd.Series:
    """Zero mean of body_mass_index"""
    return body_mass_index - body_mass_index_mean


def body_mass_index_std_dev(body_mass_index: pd.Series) -> pd.Series:
    """Standard deviation of body_mass_index."""
    return body_mass_index.std()


def body_mass_index_zero_mean_unit_variance(body_mass_index_zero_mean: pd.Series,
                                            body_mass_index_std_dev: pd.Series) -> pd.Series:
    """Zero mean unit variance value of body_mass_index"""
    return body_mass_index_zero_mean / body_mass_index_std_dev