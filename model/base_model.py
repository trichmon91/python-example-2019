import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

class BaseModel:
    def __init__(self,data, labels, priors):
        self.model = None
        self.is_scipy_model = False
        self.training_data = data
        self.training_labels = labels
        self.priors = priors
        # self.costs = [0, 0.05, -1, 0]  # C00(TN), C10(FP), C11(TP), C01(FN) #toptimal
        self.costs = [0, 0.05, -0.25, 1.25]  # C00(TN), C10(FP), C11(TP), C01(FN  #tsepsis
        
    def train(self):
        if self.is_scipy_model:
            self.model.fit(self.training_data, self.training_labels)
        else:
            raise NotImplementedError("Train method not implemented for this model")
                        
        
    def predict_odds(self, data):
        if self.is_scipy_model:
            seps_probs = self.model.predict_log_proba(self.training_data)[:, 1]
            non_probs = self.model.predict_log_proba(self.training_data)[:, 0]
            return seps_probs - non_probs
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
        plt.xlabel(r'$P_F$')
        plt.ylabel(r'$P_D$')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()
        
    def bayesian_risk(self, cost=None, predicted_odds=None, gt_labels=None):
        p0 = self.priors[0]  # np.sum(self.training_labels == 0) / np.size(self.training_labels)
        p1 = self.priors[1]  # np.sum(self.training_labels == 1) / np.size(self.training_labels)
        if cost is None:
            cost = self.costs
        slope = (cost[1] - cost[0])/(cost[3] - cost[2]) * (p0/p1)
        thresh = np.log((cost[1] - cost[0])/(cost[3] - cost[2]))
        if predicted_odds is None:
            predicted_odds = self.predict_odds(self.training_data)
            if gt_labels is not None:
                raise Exception('Must pass in both predicted_odds and gt_labels or Neither.')
            gt_labels = self.training_labels
        tpr = np.sum(predicted_odds[gt_labels == 1] > thresh) / np.sum(gt_labels == 1)
        fpr = np.sum(predicted_odds[gt_labels == 0] > thresh) / np.sum(gt_labels == 0)
        # total prob of error
        prob_e = p0 * fpr + p1 * (1 - tpr)
        return tpr, fpr, prob_e, slope


    def Neyman_Pearson(self):
        raise NotImplementedError("Implementation Pending")