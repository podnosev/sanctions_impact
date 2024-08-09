import numpy as np
import pandas as pd
import scipy.stats as stats


''' parameters '''

class MetricCalc:
    col = "Цена"

    def __init__(self, col: str):
        self.col = col

    def mean(self, df: pd.DataFrame):
        return df[self.col].mean()

    def median(self, df: pd.DataFrame):
        return df[self.col].median()

    def max(self, df: pd.DataFrame):
        return df[self.col].max()

    def min(self, df: pd.DataFrame):
        return df[self.col].min()

    def std(self, df: pd.DataFrame):
        return df[self.col].std()

    def dispersion(self, df: pd.DataFrame):
        return self.std(df)**2

    def asymmetry(self, df: pd.DataFrame):
        tmp = []
        mean = df[self.col].mean()
        for value in df[self.col]:
            tmp.append((value - mean) ** 3)
        return np.mean(tmp) / (self.std(df) ** 3)

    def excess(self, df: pd.DataFrame):
        tmp = []
        mean = df[self.col].mean()
        for value in df[self.col]:
            tmp.append((value - mean) ** 4)
        return np.mean(tmp) / (self.std(df) ** 4) - 3

    def drawdown(self, df: pd.DataFrame, point: int):
        if point == 0:
            return 0
        flag = 0
        pointed = df.at[point, self.col]
        cur = df.at[point, self.col]
        prev = df.at[point - 1, self.col]
        for i in range(1, point):
            if (prev > cur and flag == 0):
                flag = 1
            if (prev <= cur and flag == 1):
                return pointed - cur
            cur = prev
            prev = df.at[point - i - 1, self.col]
        return pointed - prev

    def jarque_bera(self, df: pd.DataFrame):
        return stats.jarque_bera(df[self.col])