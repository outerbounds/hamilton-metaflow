from hamilton.function_modifiers import extract_columns
import pandas as pd


@extract_columns('seasons_1', 'seasons_2', 'seasons_3', 'seasons_4')
def seasons_encoder(seasons: pd.Series) -> pd.DataFrame:
    """One hot encodes seasons into 4 dimensions:
    1 - first season
    2 - second season
    3 - third season
    4 - fourth season
    """
    return pd.get_dummies(seasons, prefix='seasons')


@extract_columns('education_1', 'education_2', 'education_3', 'education_4')
def education_encoder(education: pd.Series) -> pd.DataFrame:
    """One hot encodes education into 4 dimensions
    1 - high school
    2 - graduate
    3 - masters
    4 - PhD
    """
    return pd.get_dummies(education, prefix='education')


@extract_columns('day_of_the_week_2',
                 'day_of_the_week_3',
                 'day_of_the_week_4',
                 'day_of_the_week_5',
                 'day_of_the_week_6')
def day_of_week_encoder(day_of_the_week: pd.Series) -> pd.DataFrame:
    """One hot encodes day of week into five dimensions -- Saturday & Sunday weren't present.
    1 - Sunday, 2 - Monday, 3 - Tuesday, 4 - Wednesday, 5 - Thursday, 6 - Friday, 7 - Saturday.
    """
    return pd.get_dummies(day_of_the_week, prefix='day_of_the_week')


def has_children(son: pd.Series) -> pd.Series:
    """Single variable that says whether someone has any children or not."""
    return pd.Series(son > 0, son.index).astype(int)


def has_pet(pet: pd.Series) -> pd.Series:
    """Single variable that says whether someone has any pets or not."""
    return pd.Series(pet > 0, pet.index).astype(int)


def is_summer(month_of_absence: pd.Series) -> pd.Series:
    """Is it summer in Brazil? i.e. months of December, January, February."""
    summer = month_of_absence.isin([1, 2, 12])
    return pd.Series(summer, index=month_of_absence.index).astype(int)
