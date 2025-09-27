import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataExtractionAnalysis:
    def __init__(self, data_path=None, connection=None):
        self.data_path = data_path
        self.connection = connection
        self.df = None

    def load_data(self):
        if self.data_path:
            self.df = pd.read_csv(self.data_path)
        elif self.connection is not None:
            query = "SELECT * FROM streaming_table;"
            self.df = pd.read_sql(query, self.connection)
        return self.df

    def basic_summary(self):
        print(self.df.info())
        print(self.df.describe())

    def plot_raw_signals(self, columns):
        self.df[columns].plot(figsize=(12,4))
        plt.title("Raw Sensor Signals")
        plt.show()
