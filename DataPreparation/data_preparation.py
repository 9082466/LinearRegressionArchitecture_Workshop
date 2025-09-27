from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

class DataPreparation:
    def __init__(self):
        self.stdscaler = StandardScaler()
        self.minmax = MinMaxScaler()

    def clean(self, df):
        df = df.dropna().copy()
        return df

    def transform(self, df, features):
        df_std = df.copy()
        df_norm = df.copy()
        df_std[features] = self.stdscaler.fit_transform(df_std[features])
        df_norm[features] = self.minmax.fit_transform(df_norm[features])
        return df_std, df_norm

    def split(self, df, target, test_size=0.2, random_state=42):
        X = df.drop(columns=[target])
        y = df[target]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
