import joblib

class TrainedMLModel:
    def __init__(self, model=None):
        self.model = model

    def save(self, path="model.pkl"):
        joblib.dump(self.model, path)

    def load(self, path="model.pkl"):
        self.model = joblib.load(path)
        return self.model

    def predict(self, X):
        return self.model.predict(X)
