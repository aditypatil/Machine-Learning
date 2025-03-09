"""
Accuracy = (TP+TN)/(TP+FP+TN+FN) (correct/total)
Precision = (TP)/(FP+TP) (of the predicted positives, what's correct prediction rate)
Recall = (TP)/(TP+FN) (of the original psitives, what's the correct prediction rate)
F1 = 2*(Precision*Recall)/(Precision+Recall) (Balancing Precision and Recall)
AUC => get prediction values and plot TPR vs FPR by varying the cutoff for classification (0 is bad model, 0.5 is random model, 1 is good model) 
Logloss = y log(y_hat) + (1-y) log(1-y_hat)
KS score = 
"""
from sklearn import metrics as skmetrics
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

class ClassificationMetrics:
    def __init__(self):
        self.metrics = {
            "accuracy": self._accuracy,
            "f1": self._f1,
            "recall": self._recall,
            "precision": self._precision,
            "auc": self._auc,
            "logloss": self._logloss,
            "ks": self._ks_score
        }
        
    
    def __call__(self, metric, y_true, y_pred, y_proba=None):
        if metric not in self.metrics:
            raise Exception("Metric not implemented")
        if metric in ["auc", "logloss", "ks"]:
            if y_proba is not None:
                return self.metrics[metric](y_true=y_true, y_pred=y_proba)
            else:
                raise Exception(f"y_proba cannot be None for {metric}")
        return self.metrics[metric](y_true=y_true, y_pred=y_pred)

    
    @staticmethod
    def _accuracy(y_true, y_pred):
        return skmetrics.accuracy_score(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _f1(y_true, y_pred):
        return skmetrics.f1_score(y_true=y_true, y_pred=y_pred)
    
    @staticmethod
    def _recall(y_true, y_pred):
        return skmetrics.recall_score(y_true=y_true, y_pred=y_pred)
    
    @staticmethod
    def _precision(y_true, y_pred):
        return skmetrics.precision_score(y_true=y_true, y_pred=y_pred)
    
    @staticmethod
    def _auc(y_true, y_pred):
        return skmetrics.roc_auc_score(y_true=y_true, y_score=y_pred)
    
    @staticmethod
    def _logloss(y_true, y_pred):
        return skmetrics.log_loss(y_true=y_true, y_pred=y_pred)
    
    @staticmethod
    def _ks_score(y_true, y_pred):
        pos_scores = y_pred[y_true == 1]
        neg_scores = y_pred[y_true == 0]
        ks_stat, _ = ks_2samp(pos_scores, neg_scores)
        return ks_stat