"""This code should go in metaflow"""
import pandas as pd
from hamilton import driver

import data_loader
import feature_logic
import normalized_features

# TBD how we want to model this in metaflow
dr = driver.Driver({"location": "data/Absenteeism_at_work.csv"},
                   data_loader, feature_logic, normalized_features)

columns_to_exclude = {'id', 'reason_for_absence', 'month_of_absence', 'day_of_the_week'}

# should curate an actual list of features rather than doing this
features_wanted = [n.name for n in dr.list_available_variables()
                   if n.name not in columns_to_exclude and n.type == pd.Series]

featurized_df = dr.execute(features_wanted)

print(featurized_df.to_string())

