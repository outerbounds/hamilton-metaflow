import pandas as pd


def age_mean(age: pd.Series) -> float:
    """Average of age"""
    return age.mean()


def age_zero_mean(age: pd.Series, age_mean: float) -> pd.Series:
    """Zero mean of age"""
    return age - age_mean


def age_std_dev(age: pd.Series) -> pd.Series:
    """Standard deviation of age."""
    return age.std()


def age_zero_mean_unit_variance(age_zero_mean: pd.Series, age_std_dev: pd.Series) -> pd.Series:
    """Zero mean unit variance value of age"""
    return age_zero_mean / age_std_dev


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


def transportation_expense_mean(transportation_expense: pd.Series) -> float:
    """Average transportation_expense"""
    return transportation_expense.mean()


def transportation_expense_zero_mean(transportation_expense: pd.Series, transportation_expense_mean: float) -> pd.Series:
    """Zero mean of transportation_expense"""
    return transportation_expense - transportation_expense_mean


def transportation_expense_std_dev(transportation_expense: pd.Series) -> pd.Series:
    """Standard deviation of transportation_expense."""
    return transportation_expense.std()


def transportation_expense_zero_mean_unit_variance(transportation_expense_zero_mean: pd.Series, transportation_expense_std_dev: pd.Series) -> pd.Series:
    """Zero mean unit variance value of transportation_expense"""
    return transportation_expense_zero_mean / transportation_expense_std_dev


def distance_from_residence_to_work_mean(distance_from_residence_to_work: pd.Series) -> float:
    """Average distance_from_residence_to_work"""
    return distance_from_residence_to_work.mean()


def distance_from_residence_to_work_zero_mean(distance_from_residence_to_work: pd.Series, distance_from_residence_to_work_mean: float) -> pd.Series:
    """Zero mean of distance_from_residence_to_work"""
    return distance_from_residence_to_work - distance_from_residence_to_work_mean


def distance_from_residence_to_work_std_dev(distance_from_residence_to_work: pd.Series) -> pd.Series:
    """Standard deviation of distance_from_residence_to_work."""
    return distance_from_residence_to_work.std()


def distance_from_residence_to_work_zero_mean_unit_variance(distance_from_residence_to_work_zero_mean: pd.Series, distance_from_residence_to_work_std_dev: pd.Series) -> pd.Series:
    """Zero mean unit variance value of distance_from_residence_to_work"""
    return distance_from_residence_to_work_zero_mean / distance_from_residence_to_work_std_dev


def service_time_mean(service_time: pd.Series) -> float:
    """Average service_time"""
    return service_time.mean()


def service_time_zero_mean(service_time: pd.Series, service_time_mean: float) -> pd.Series:
    """Zero mean of service_time"""
    return service_time - service_time_mean


def service_time_std_dev(service_time: pd.Series) -> pd.Series:
    """Standard deviation of service_time."""
    return service_time.std()


def service_time_zero_mean_unit_variance(service_time_zero_mean: pd.Series, service_time_std_dev: pd.Series) -> pd.Series:
    """Zero mean unit variance value of service_time"""
    return service_time_zero_mean / service_time_std_dev


def work_load_average_per_day_mean(work_load_average_per_day: pd.Series) -> float:
    """Average work_load_average_per_day"""
    return work_load_average_per_day.mean()


def work_load_average_per_day_zero_mean(work_load_average_per_day: pd.Series, work_load_average_per_day_mean: float) -> pd.Series:
    """Zero mean of work_load_average_per_day"""
    return work_load_average_per_day - work_load_average_per_day_mean


def work_load_average_per_day_std_dev(work_load_average_per_day: pd.Series) -> pd.Series:
    """Standard deviation of work_load_average_per_day."""
    return work_load_average_per_day.std()


def work_load_average_per_day_zero_mean_unit_variance(work_load_average_per_day_zero_mean: pd.Series, work_load_average_per_day_std_dev: pd.Series) -> pd.Series:
    """Zero mean unit variance value of work_load_average_per_day"""
    return work_load_average_per_day_zero_mean / work_load_average_per_day_std_dev


def hit_target_mean(hit_target: pd.Series) -> float:
    """Average hit_target"""
    return hit_target.mean()


def hit_target_zero_mean(hit_target: pd.Series, hit_target_mean: float) -> pd.Series:
    """Zero mean of hit_target"""
    return hit_target - hit_target_mean


def hit_target_std_dev(hit_target: pd.Series) -> pd.Series:
    """Standard deviation of hit_target."""
    return hit_target.std()


def hit_target_zero_mean_unit_variance(hit_target_zero_mean: pd.Series, hit_target_std_dev: pd.Series) -> pd.Series:
    """Zero mean unit variance value of hit_target"""
    return hit_target_zero_mean / hit_target_std_dev

