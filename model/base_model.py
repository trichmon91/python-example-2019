import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

class BaseModel:
    def __init__(self,data, labels):
        self.model = None
        self.is_scipy_model = False
        self.training_data = data
        self.training_labels = labels
        
    def train(self):
        if self.is_scipy_model:
            self.model.fit(self.training_data, self.training_labels)
        else:
            raise NotImplementedError("Train method not implemented for this model")
                        
        
    def predict_odds(self, data):
        if self.is_scipy_model:
            return self.model.predict_log_proba(self.training_data)[:,1]
        else:
            raise NotImplementedError("Predict odds method not implemented for base model")
        
    def plot_roc(self, predicted_odds=None, gt_labels=None):
        if predicted_odds == None:
            predicted_odds = self.predict_odds(self.training_data)
            if gt_labels != None:
                raise Exception('Must pass in both predicted_odds and gt_labels or Neither.')
            gt_labels = self.training_labels
        fpr,tpr,thresh = roc_curve(gt_labels, predicted_odds, pos_label=1)
        self.visualize_roc(fpr, tpr)
        
    def visualize_roc(self, fpr, tpr):
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve')
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()
        
    def bayesian_risk(self):
        raise NotImplementedError("Implementation Pending")
        
    def Neyman_Pearson(self):
        raise NotImplementedError("Implementation Pending")