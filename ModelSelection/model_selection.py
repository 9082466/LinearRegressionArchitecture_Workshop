from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

class ModelSelection:
    def __init__(self, model_type="linear"):
        self.model_type = model_type

    def get_model(self):
        if self.model_type == "linear":
            return LinearRegression()
        elif self.model_type == "rf":
            return RandomForestRegressor()
        else:
            raise ValueError("Unsupported model type")
