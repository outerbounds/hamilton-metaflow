from metaflow import FlowSpec, step, Parameter, card, current, conda_base, batch
from metaflow.cards import Image
import os

@conda_base(libraries={"xgboost":"1.5.1", "matplotlib":"3.5.1", 
                       "scikit-learn":"1.0.2", "pandas":"1.4.2",
                       "sf-hamilton":"1.7.0", "numpy":"1.22.3",
                       "skrebate":"0.62", "python-graphviz":"0.20"},
            python="3.9.12")
class FeatureSelectionAndClassification(FlowSpec):

    '''
    Flow description:
        Featurize dataset using Hamilton
        Run feature selection process (three branches)
        Use policy (in `feature_importance_merge` step) to select top K features for subset
        Branch on full dataset and top K feature subset dataset
        Create MLR, XGBoost, Neural Net, ExtraTrees model for each dataset (4*2=8 models total)
        Merge all modeling results in a dataframe that can be accessed using run.data.results.
    '''

    number_of_subset_features = Parameter("number_of_subset_features", default=10) 
    graphviz_flag = Parameter("graphviz_flag", default=True) # true if system graphviz is installed 

    @card
    @step
    def start(self): 
        '''
        Show distribution of class labels in matplotlib figure.
        '''
        import pandas as pd
        import numpy as np
        from flow_utilities import plot_labels, encode_labels, Config
        raw_data = pd.read_csv(Config.RAW_DATA_LOCATION, sep=";")
        self.raw_data_features = raw_data.drop(columns=[Config.LABEL_COL_NAME])
        self.labels = raw_data[Config.LABEL_COL_NAME].apply(encode_labels)
        self.raw_data_features.to_csv(Config.RAW_FEATURES_LOCATION, index=False, sep=";")
        figure = plot_labels(labels = self.labels.value_counts().values, raw_data = raw_data)
        current.card.append(Image.from_matplotlib(figure))
        self.next(self.featurize_and_split)

    @card
    @step
    def featurize_and_split(self):
        '''
        Transform and document features using sf-hamilton.
        '''
        import pandas as pd
        from hamilton import driver
        import data_loader
        import feature_logic
        import normalized_features
        from sklearn.model_selection import train_test_split
        from flow_utilities import hamilton_viz, Config
        dr = driver.Driver({"location": Config.RAW_FEATURES_LOCATION},
                           data_loader, feature_logic, normalized_features)
        features_wanted = [n.name for n in dr.list_available_variables()
                           if n.name not in Config.EXCLUDED_COLS and n.type == pd.Series]
        self.full_featurized_data = dr.execute(features_wanted)
        if self.graphviz_flag:
            current.card.append(Image.from_matplotlib(hamilton_viz(dr, features_wanted)))
        self.train_x_full, self.validation_x_full, self.train_y_full, self.validation_y_full = train_test_split(
            self.full_featurized_data, self.labels, test_size=0.2, random_state=Config.RANDOM_STATE)
        self.next(self.relief_based_feature_selection, 
            self.correlation_based_feature_selection,
            self.info_gain_feature_selection)

    @step 
    def relief_based_feature_selection(self):
        '''
        Compute feature importance based on RBFS weights.
        https://arxiv.org/abs/1711.08477
        '''
        from skrebate import ReliefF
        import pandas as pd
        self.rbfs = ReliefF()
        self.rbfs.fit(self.train_x_full.values, self.train_y_full.values)
        self.rbfs_scores = dict(zip(list(self.train_x_full.columns), self.rbfs.feature_importances_))
        self.next(self.feature_importance_merge)

    @step 
    def correlation_based_feature_selection(self):
        '''
        Compute feature importance based on a low correlation with other features.
        '''
        from flow_utilities import cbfs
        feature_names, correlation_when_removed = cbfs(self.train_x_full, N=10)
        self.cbfs_scores = dict(zip(feature_names, correlation_when_removed))
        self.next(self.feature_importance_merge)

    @step 
    def info_gain_feature_selection(self):
        '''
        Compute feature importance based on information gain.
        '''
        from sklearn.feature_selection import SelectKBest, mutual_info_classif
        selector = SelectKBest(mutual_info_classif, k=10)
        selected_features = selector.fit_transform(self.train_x_full, self.train_y_full)
        idxs = selector.get_support(indices=True)
        self.igfs_scores = dict(zip(self.train_x_full.columns[idxs], selector.scores_[idxs]))
        self.next(self.feature_importance_merge)

    @card
    @step
    def feature_importance_merge(self, inputs):
        '''
        Merge feature importance steps. Select top k features.
        '''
        import pandas as pd
        from hamilton import driver
        import data_loader
        import feature_logic
        import normalized_features
        from sklearn.model_selection import train_test_split
        from flow_utilities import hamilton_viz, Config
        import numpy as np

        # resolve ambiguity for metaflow join step, can choose any of feature selection steps
        self.train_x_full = inputs.relief_based_feature_selection.train_x_full
        self.validation_x_full = inputs.relief_based_feature_selection.validation_x_full
        self.train_y_full = inputs.relief_based_feature_selection.train_y_full
        self.validation_y_full = inputs.relief_based_feature_selection.validation_y_full
        self.merge_artifacts(inputs)

        # find top k features
        top_k_idxs = self.rbfs.top_features_[:self.number_of_subset_features] # policy: use top k according to RBFS
        top_k_features = np.array(list(self.rbfs_scores.keys()))[[top_k_idxs]]
        top_k_importance_values = np.array(list(self.rbfs_scores.values()))[[top_k_idxs]]

        # use Hamilton to select top k features
        dr = driver.Driver({"location": Config.RAW_FEATURES_LOCATION},
                           data_loader, feature_logic, normalized_features)
        self.select_featurized_data = dr.execute(top_k_features)
        if self.graphviz_flag:
            current.card.append(Image.from_matplotlib(hamilton_viz(dr, top_k_features, feature_set="selected")))
        self.train_x_select, self.validation_x_select, self.train_y_select, self.validation_y_select = train_test_split(
            self.select_featurized_data, self.labels, test_size=0.2, random_state=Config.RANDOM_STATE)
        self.next(self.dataset_split)

    @step
    def dataset_split(self):
        '''create two branches for the full dataset features and the top K features dataset'''
        self.data_split = [
            (self.train_x_full, self.validation_x_full, self.train_y_full, self.validation_y_full, "full"),
            (self.train_x_select, self.validation_x_select, self.train_y_select, self.validation_y_select, "selected")
        ]
        self.next(self.classifiers_dag, foreach="data_split")

    @step
    def classifiers_dag(self):
        self.train_x = self.input[0]
        self.valid_x = self.input[1]
        self.train_y = self.input[2]
        self.valid_y = self.input[3]
        self.dataset_name = self.input[4]
        self.next(self.multinomial_logistic_regression, 
            self.xgboost,
            self.neural_net,
            self.extra_trees)

    @step 
    def multinomial_logistic_regression(self):
        '''Fit and score sklearn.linear_model.LogisticRegression models'''
        from sklearn.linear_model import LogisticRegression
        from flow_utilities import fit_and_score_multiclass_model, Config
        self.model_name = "Multinomial Logistic Regression"
        params = {"penalty": "l2", "solver":"lbfgs", "random_state": Config.RANDOM_STATE,
                  "n_jobs": 1, "multi_class": "multinomial"}
        self.model, self.scores = fit_and_score_multiclass_model(
            LogisticRegression(**params), 
            self.train_x, self.train_y, self.valid_x, self.valid_y
        )
        self.params = params
        self.next(self.gather_model_scores)

    @card
    @step 
    def xgboost(self):
        '''Fit and score xgboost.XGBClassifier models'''
        from xgboost import XGBClassifier
        import matplotlib.pyplot as plt
        from flow_utilities import fit_and_score_multiclass_model, plot_xgb_importances, Config
        self.model_name = "XGBoost"
        params = {"n_estimators": 1000, "max_depth":10, "random_state": Config.RANDOM_STATE,
                  "n_jobs": 1, "learning_rate": .1, "objective": "multi:softmax"}
        self.model, self.scores = fit_and_score_multiclass_model(
            XGBClassifier(**params), 
            self.train_x, self.train_y, self.valid_x, self.valid_y
        )
        self.params = params
        figure = plot_xgb_importances(self.model)
        current.card.append(Image.from_matplotlib(figure))
        self.next(self.gather_model_scores)

    @step 
    def neural_net(self):
        '''Fit and score neural network'''
        from sklearn.neural_network import MLPClassifier
        from flow_utilities import fit_and_score_multiclass_model, Config
        self.model_name = "Neural Network"
        params = {"learning_rate_init": 0.01, "learning_rate": "adaptive", 
                  "hidden_layer_sizes": 50, "random_state": Config.RANDOM_STATE}
        self.model, self.scores = fit_and_score_multiclass_model(
            MLPClassifier(**params),
            self.train_x, self.train_y, self.valid_x, self.valid_y
        )
        self.params = params
        self.next(self.gather_model_scores)

    @step 
    def extra_trees(self):
        '''Fit and score extra trees model'''
        from sklearn.ensemble import ExtraTreesClassifier
        from flow_utilities import fit_and_score_multiclass_model, Config
        self.model_name = "ExtraTrees"
        params = {"criterion": "gini", "max_depth": 8, "random_state": Config.RANDOM_STATE}
        self.model, self.scores = fit_and_score_multiclass_model(
            ExtraTreesClassifier(**params), 
            self.train_x, self.train_y, self.valid_x, self.valid_y
        )
        self.params = params
        self.next(self.gather_model_scores)

    @step
    def gather_model_scores(self, inputs):
        '''for each model, gather configuration and metric scores'''
        import pandas as pd
        import json
        results = { 
            "model name": [],
            "model params": [],
            "dataset name": []
        }
        metrics = {
            "accuracy": [], 
            "macro-weighted precision": [],
            "macro-weighted recall": [],
            "macro-weighted f1": [],
            "training time": [],
            "prediction time": []
        }
        for modeling_step in inputs:
            results['model name'].append(modeling_step.model_name)
            results['model params'].append(json.dumps(modeling_step.params))
            results['dataset name'].append(modeling_step.dataset_name)
            for metric_name in metrics.keys():
                metrics[metric_name].append(modeling_step.scores[metric_name])
        results |= metrics
        self.results_df = pd.DataFrame(results)
        self.next(self.gather_across_datasets)

    @step
    def gather_across_datasets(self, inputs):
        '''merge results dataframes from `gather_model_scores` for each dataset split'''
        import pandas as pd
        self.results = pd.DataFrame()
        for dataset_split in inputs:
            self.results = pd.concat([self.results, dataset_split.results_df], axis=0)
        self.next(self.end)

    @step
    def end(self):
        print("FeatureSelectionAndClassification flow is complete!")

if __name__ == "__main__":
    FeatureSelectionAndClassification()