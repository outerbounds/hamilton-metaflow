from time import time
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.stats import pearsonr
import itertools
import pandas as pd


RANDOM_STATE = 0
RAW_DATA_LOCATION = "./data/Absenteeism_at_work.csv"
LABEL_COL_NAME = "Absenteeism time in hours"
RAW_FEATURES_LOCATION = "./data/raw_features.csv"
TPOT_SCRIPT_DESTINATION = "./tpot_optimized_pipeline.py"


def label_encoding(x):
    '''
    encode labels for classification task described in:
        https://www.researchgate.net/publication/358900589_PREDICTION_OF_ABSENTEEISM_AT_WORK_WITH_MULTINOMIAL_LOGISTIC_REGRESSION_MODEL
    '''
    if x == 0:
        return 0
    elif 1 <= x <= 15: 
        return 1
    elif 16 <= x <= 120:
        return 2
    else: 
        raise ValueError(f"Data point with > 120 hours of absenteeism does not fit label encoding.")


def plot_labels(labels, raw_data):
    figure = plt.figure(figsize=(16,6))
    # show classes for task from paper
    buckets = ["0 hours", "1-15 hours", "16-120 hours"]
    ax1 = figure.add_subplot(121)
    ax1.bar(x=buckets, height=labels)
    ax1.set_xlabel("Class Labels", fontsize=18)
    # show label distribution graphic from paper
    buckets = [0, 1, 2, 3, 4, 5, 7, 8, 16, 24, 32, 48, 56, 64, 80, 104, 112, 120]; employee_count = []
    for num_absent_hours in buckets:
        employee_count.append((raw_data[LABEL_COL_NAME] == num_absent_hours).sum())
    ax2 = figure.add_subplot(122)
    buckets = [str(b) for b in buckets]
    ticks = np.arange(len(buckets))
    ax2.bar(x=ticks, height=employee_count)
    ax2.set_xticks(ticks, labels=buckets, rotation=35, fontsize=10)
    ax2.set_xlabel("Hours Absent", fontsize=18)
    figure.suptitle("Label Distribution", fontsize=24)
    return figure


def get_correlation_matrix(df):
    '''
    https://stackoverflow.com/questions/33997753/calculating-pairwise-correlation-among-all-columns
    '''
    df.corr().mean().index()


def cbfs(features:pd.DataFrame, N=15):
    '''
    simplified version of:
        https://www.hindawi.com/journals/complexity/2018/2520706/    
    '''
    fj_colname = features.corr().abs().mean().idxmin()
    F_s = [fj_colname]
    Cij = 0
    corr_value_when_removed = []
    while len(F_s) < N:
        vec = features.corr().filter(F_s, axis=0).drop(columns=F_s, axis=1).abs().mean()
        fj_colname = vec.idxmin()
        Cij = vec.min()
        F_s.append(fj_colname)
        corr_value_when_removed.append(Cij)
    return F_s, corr_value_when_removed
        

def plot_xgb_importances(booster):
    from xgboost import plot_importance
    figure, (ax1, ax2) = plt.subplots(2, 1, figsize=(20,16))
    plt.draw()
    plot_importance(booster = booster, ax=ax1, importance_type="weight")
    ax1.set_xlabel("Number times feature in tree", fontsize=18)
    ax1.tick_params(axis='y', labelrotation=40, labelsize=12)
    ax1.set_title("")
    plot_importance(booster = booster, ax=ax2, importance_type="gain")
    ax2.set_xlabel("Average gain of splits using feature", fontsize=18)
    ax2.tick_params(axis='y', labelrotation=40, labelsize=12)
    ax2.set_title("")
    figure.suptitle("XGBoost Feature Importance Calculations", fontsize=24)
    figure.tight_layout()
    return figure


def fit_and_score_multiclass_model(model, train_x, train_y, valid_x, valid_y):
    t0_train = time()
    model.fit(train_x, train_y)
    tf_train = time()
    t0_valid = time()
    preds = model.predict(valid_x)
    tf_valid = time()
    return model, {
        "accuracy": accuracy_score(valid_y, preds),
        # Note: micro f1 is same as accuracy it class > 2 setting
        "macro-weighted precision": precision_score(valid_y, preds, average="macro"),
        "macro-weighted recall": recall_score(valid_y, preds, average="macro"),
        "macro-weighted f1": f1_score(valid_y, preds, average="macro"),
        'training time': tf_train - t0_train,
        "prediction time": tf_valid - t0_valid
    }


class SkorchModule(nn.Module):
    def __init__(self, num_input_feats=59, num_units=10, nonlin=F.relu, num_classes=3):
        super(SkorchModule, self).__init__()

        self.dense0 = nn.Linear(num_input_feats, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(num_units, 10)
        self.output = nn.Linear(10, num_classes)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = F.relu(self.dense1(X))
        X = F.softmax(self.output(X))
        return X