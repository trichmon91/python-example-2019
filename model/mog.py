from model.base_model import BaseModel
from sklearn import mixture
import numpy as np


class mog(BaseModel):

    def __init__(self, data, labels):
        super().__init__(data, labels)
        self.is_scipy_model = False
        self.seps_model = None
        self.non_model = None
        self.predicted_odds = None
        self.seps_data = data[labels==1, :]
        self.non_data = data[labels==0, :]
        self.labels = labels

    def best_model(self, data, n_components_range):
        lowest_bic = np.infty
        bic = []
        cv_types = ['spherical', 'tied', 'diag', 'full']
        for cv_type in cv_types:
            for n_components in n_components_range:
                # Fit a Gaussian mixture with EM
                gmm = mixture.GaussianMixture(n_components=n_components,
                                              covariance_type=cv_type)
                gmm.fit(self.seps_data)
                bic.append(gmm.bic(self.seps_data))
                if bic[-1] < lowest_bic:
                    lowest_bic = bic[-1]
                    best_gmm = gmm
        return best_gmm

    def train(self):
        n_components_range = range(1, 10)
        self.seps_model = self.best_model(self.seps_data, n_components_range)
        self.non_model = self.best_model(self.non_data, n_components_range)
        self.model = self.seps_model

    def predict_odds(self, data):
        seps_probs = self.seps_model.score_samples(data)
        non_probs = self.non_model.score_samples(data)
        # predictions = probs > 0
        # correct_pred = predictions == model.labels
        # pos_pred = np.sum(correct_pred[model.labels == 1])
        # tpr = pos_pred / np.sum(model.labels == 1)
        # fals_pos = predictions[model.labels == 0] == 1
        # fpr = np.sum(fals_pos) / np.sum(model.labels == 0)
        return seps_probs - non_probs
