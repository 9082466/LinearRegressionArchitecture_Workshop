import numpy as np

class ModelTraining:
    def __init__(self, model):
        self.model = model

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def compute_residuals(self, X, y):
        predictions = self.model.predict(X)
        residuals = y - predictions
        return residuals
