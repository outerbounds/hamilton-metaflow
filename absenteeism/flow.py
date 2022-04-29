from metaflow import FlowSpec, step, Parameter, card, current, conda, batch
from metaflow.cards import Image
from flow_utilities import Config, pip
import functools


class FeatureSelectionAndClassification(FlowSpec):

    '''
    flow description
        task A: Featurize raw data using hamilton.
        task B: Compare and visualize feature selection methods. 
        task C: Compare and visualize classification modeling approaches.

        Task B's feature selection flow is inspired by section 3.2 of: 
            http://ijece.iaescore.com/index.php/IJECE/article/view/25232/15114

        Task C's prediction task is the one described in:
            https://www.researchgate.net/publication/358900589_PREDICTION_OF_ABSENTEEISM_AT_WORK_WITH_MULTINOMIAL_LOGISTIC_REGRESSION_MODEL
    '''

    @conda(disabled=True)
    @card
    @step
    def start(self): 
        '''
        Show distribution of class labels in matplotlib figure.
        '''
        import pandas as pd
        import numpy as np
        from flow_utilities import plot_labels, encode_labels
        raw_data = pd.read_csv(Config.RAW_DATA_LOCATION, sep=";")
        self.raw_data_features = raw_data.drop(columns=[Config.LABEL_COL_NAME])
        self.labels = raw_data[Config.LABEL_COL_NAME].apply(encode_labels)
        self.raw_data_features.to_csv(Config.RAW_FEATURES_LOCATION, index=False, sep=";")
        figure = plot_labels(labels = self.labels.value_counts().values, raw_data = raw_data)
        current.card.append(Image.from_matplotlib(figure))
        self.next(self.featurize_and_split)

    @card
    @conda(disabled=True)
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
        from flow_utilities import hamilton_viz
        dr = driver.Driver({"location": Config.RAW_FEATURES_LOCATION},
                           data_loader, feature_logic, normalized_features)
        features_wanted = [n.name for n in dr.list_available_variables()
                           if n.name not in Config.EXCLUDED_COLS and n.type == pd.Series]
        self.full_featurized_data = dr.execute(features_wanted)
        # NOTE: Metaflow card's Image viewer forces everything into a square view.
        # this makes card viewer somewhat ineffective for Hamilton DAG which is much wider than tall. 
        # If we are feeling fancy and want to spend more hours on this we could 
        # map hamilton.graph.FunctionGraph to a metaflow.cards.DAG type in cards/ui/src/types.js  
        current.card.append(Image.from_matplotlib(hamilton_viz(dr, features_wanted)))
        self.train_x_full, self.validation_x_full, self.train_y_full, self.validation_y_full = train_test_split(
            self.full_featurized_data, self.labels, test_size=0.2, random_state=Config.RANDOM_STATE)
        self.next(self.relief_based_feature_selection, 
            self.correlation_based_feature_selection,
            self.info_gain_feature_selection)

    @conda(disabled=True)
    @step 
    def relief_based_feature_selection(self):
        '''
        Compute feature importance based on RBFS weights.
        https://arxiv.org/abs/1711.08477
        '''
        # TODO figure out how to use skrebate
        from skrebate import ReliefF
        import pandas as pd
        from copy import deepcopy
        rbfs = ReliefF()
        rbfs.fit(self.train_x_full.values, self.train_y_full.values)
        self.rbfs = {name:score for name, score in zip(list(self.train_x_full.columns), 
            rbfs.feature_importances_)}
        self.next(self.feature_importance_merge)

    @conda(disabled=True)
    @step 
    def correlation_based_feature_selection(self):
        '''
        Compute feature importance based on a low correlation with other features.
        '''
        from flow_utilities import cbfs
        feature_names, correlation_when_removed = cbfs(self.train_x_full, N=10)
        self.cbfs = {name:score for name, score in zip(feature_names, correlation_when_removed)}
        self.next(self.feature_importance_merge)

    @conda(disabled=True)
    @step 
    def info_gain_feature_selection(self):
        '''
        Compute feature importance based on information gain.
        '''
        from sklearn.feature_selection import SelectKBest, mutual_info_classif
        selector = SelectKBest(mutual_info_classif, k=10)
        selected_features = selector.fit_transform(self.train_x_full, self.train_y_full)
        idxs = selector.get_support(indices=True)
        self.igfs = {name:score for name,score in zip(self.train_x_full.columns[idxs], 
            selector.scores_[idxs])}
        self.next(self.feature_importance_merge)

    @conda(disabled=True)
    @step
    def feature_importance_merge(self, inputs):
        '''
        Merge feature importance steps.
        '''
        # TODO: setup partial dataset flow based on feature selection policy that merges above results.
        # resolve ambiguity for metaflow join step, can choose any of feature selection steps
        self.train_x_full = inputs.relief_based_feature_selection.train_x_full
        self.validation_x_full = inputs.relief_based_feature_selection.validation_x_full
        self.train_y_full = inputs.relief_based_feature_selection.train_y_full
        self.validation_y_full = inputs.relief_based_feature_selection.validation_y_full
        self.merge_artifacts(inputs)
        self.next(self.classifiers_dag)

    @conda(disabled=True)
    @step
    def classifiers_dag(self):
        self.next(self.multinomial_logistic_regression, 
            self.xgboost,
            self.neural_net,
            self.automl)

    @conda(disabled=True)
    @step 
    def multinomial_logistic_regression(self):
        '''sklearn.linear_model.LogisticRegression models'''
        # TODO: tune params
        from sklearn.linear_model import LogisticRegression
        from time import time
        from flow_utilities import fit_and_score_multiclass_model
        params = {"penalty": "l2", "solver":"lbfgs", "random_state": Config.RANDOM_STATE,
                  "n_jobs": 1, "multi_class": "multinomial"}
        self.mlr_model, self.mlr_scores = fit_and_score_multiclass_model(LogisticRegression(**params), 
            self.train_x_full, self.train_y_full, self.validation_x_full, self.validation_y_full)
        self.next(self.visualize_model_scores)

    @conda(libraries={"xgboost":"1.5.1", "matplotlib":"3.5.1", 
        "scikit-learn":"1.0.2", "pandas":"1.4.2"}, python="3.9.12")
    @card
    @step 
    def xgboost(self):
        '''xgboost.XGBClassifier models'''
        # TODO: tune params
        from xgboost import XGBClassifier
        import matplotlib.pyplot as plt
        from flow_utilities import fit_and_score_multiclass_model, plot_xgb_importances
        params = {"n_estimators": 1000, "max_depth":10, "random_state": Config.RANDOM_STATE,
                  "n_jobs": 1, "learning_rate": .1, "objective": "multi:softmax"}
        self.xgb_model, self.xgb_scores = fit_and_score_multiclass_model(XGBClassifier(**params), 
            self.train_x_full, self.train_y_full, self.validation_x_full, self.validation_y_full)
        figure = plot_xgb_importances(self.xgb_model)
        current.card.append(Image.from_matplotlib(figure))
        self.next(self.visualize_model_scores)

    #@batch(cpu=4)
    @conda(libraries={"pip":"22.0.4", "scikit-learn":"1.0.2", "pandas":"1.4.2"}, python="3.9.12") 
    @pip(libraries={"torch":"1.11.0", "skorch":"0.11.0"})
    @step 
    def neural_net(self):
        '''fit torch model using skorch interface'''
        # TODO: tune params (in flow_utilities.py)
        from skorch import NeuralNetClassifier
        import numpy as np
        from flow_utilities import fit_and_score_multiclass_model
        from torch_model import SkorchModule
        params = {"module": SkorchModule(num_input_feats=self.train_x_full.shape[1]), 
            "max_epochs":10, "lr":0.1, "iterator_train__shuffle": True, "verbose":0}
        self.nn_model, self.nn_scores = fit_and_score_multiclass_model(NeuralNetClassifier(**params), 
            self.train_x_full.values.astype(np.float32), self.train_y_full.values.astype(np.int64), 
            self.validation_x_full.values.astype(np.float32), self.validation_y_full.values.astype(np.int64))
        self.next(self.visualize_model_scores)

    #@batch(cpu=12)
    @conda(libraries={"tpot-full":"0.11.7"}, python="3.9.12")
    @step 
    def automl(self):
        '''tpot.TPOTClassifier'''
        from tpot import TPOTClassifier
        from flow_utilities import fit_and_score_multiclass_model
        params = {"generations":20, "population_size":5, "cv":5, 
                  "random_state":Config.RANDOM_STATE, "verbosity":2}
        tpot_model, self.tpot_scores = fit_and_score_multiclass_model(TPOTClassifier(**params), 
            self.train_x_full, self.train_y_full, self.validation_x_full, self.validation_y_full)
        tpot_model.export(Config.TPOT_SCRIPT_DESTINATION)
        self.next(self.visualize_model_scores)

    @conda(disabled=True)
    @step
    def visualize_model_scores(self, inputs):
        '''
        Produce metaflow card to view model performance results in.
        Result Dimensions to Display
            Flow Step
            Model Type
            Metric scores: accuracy, precision, recall, roc, training time, prediction time.
            Full Feature Set vs. Subset performance
        TODO: consider addign model interpretability libraries (e.g., LIME, Shap).
        '''
        print(f"MLR: {inputs.multinomial_logistic_regression.mlr_scores}")
        print(f"XGBoost: {inputs.xgboost.xgb_scores}")
        print(f"Neural Net: {inputs.neural_net.nn_scores}")
        self.merge_artifacts(inputs)
        self.next(self.end)

    @conda(disabled=True)
    @step
    def end(self):
        pass

if __name__ == "__main__":
    FeatureSelectionAndClassification()