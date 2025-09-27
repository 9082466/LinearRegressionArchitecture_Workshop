import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class ModelEvaluationValidation:
    def __init__(self):
        self.thresholds = {}

    def evaluate(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {"MSE": mse, "R2": r2}

    def compute_thresholds(self, residuals, lower_pct=5, upper_pct=95):
        self.thresholds = {
            "MinC": np.percentile(residuals, lower_pct),
            "MaxC": np.percentile(residuals, upper_pct)
        }
        return self.thresholds