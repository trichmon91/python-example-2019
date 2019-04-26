from model.base_model import BaseModel
from sklearn.naive_bayes import GaussianNB

class NaiveBayes(BaseModel):
    
    def __init__(self, data, labels, priors):
        super().__init__(data, labels, priors)
        self.model = GaussianNB(priors=priors)  #[(5.7 - 1.7)/5.7, 1.7/5.7])
        self.is_scipy_model = True
    