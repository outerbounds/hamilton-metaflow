import numpy as np

from hamilton.function_modifiers import extract_columns
import pandas as pd

data_columns = ['id', 'reason_for_absence', 'month_of_absence', 'day_of_the_week', 'seasons', 'transportation_expense',
                'distance_from_residence_to_work', 'service_time', 'age', 'work_load_average_per_day', 'hit_target',
                'disciplinary_failure', 'education', 'son', 'social_drinker', 'social_smoker', 'pet', 'weight',
                'height', 'body_mass_index', 'absenteeism_time_in_hours']



@extract_columns(*data_columns)
def raw_data(location: str) -> pd.DataFrame:
    df = pd.read_csv(location, sep=";")
    # rename columns to be valid hamilton names -- and lower case it
    df.columns = [c.strip().replace('/', '_per_').replace(' ', '_').lower() for c in df.columns]
    # create proper index -- ID-Month-Day;
    index = df['id'].astype(np.str) + '-' + df['month_of_absence'].astype(np.str) + '-' + df['day_of_the_week'].astype(np.str)
    df.index = index
    return df


if __name__ == '__main__':
    d = raw_data('data/Absenteeism_at_work.csv')
    print(d.to_string())
    # print(pd.get_dummies(d.seasons, prefix='seasons'))
