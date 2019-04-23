from model.base_model import BaseModel
from sklearn.linear_model import LogisticRegression

class LogReg(BaseModel):
    
    #we can take more args for sklearn logreg params. For now, hardcode.
    def __init__(self, data, labels):
        super().__init__(data, labels)
        self.model = LogisticRegression(C=1e5, solver='lbfgs')
        self.is_scipy_model = True
    