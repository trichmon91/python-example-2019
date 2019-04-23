from model.base_model import BaseModel
from sklearn.naive_bayes import GaussianNB

class NaiveBayes(BaseModel):
    
    def __init__(self, data, labels):
        super().__init__(data, labels)
        self.model = GaussianNB()
        self.is_scipy_model = True
    