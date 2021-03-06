from PyQt5.QtWidgets import *
import pandas as pd
import numpy as np
import default_process as dp

class DataContainer():
    def __init__(self):

        factor_characters_df = pd.read_csv('multifactor_characters_dt.csv', index_col=0)
        self.original_factor_characters_df = factor_characters_df.copy()
        self.factor_weight_df = factor_characters_df.iloc[:, :14]
        self.factor_value_df = factor_characters_df.iloc[:, 14:]
        # 로우 데이터 정규화
        self.normalized_factor_value_df = self.normalized_df(df=self.factor_value_df)
        # 기준 1
        self.normalized_factor_value_df['risk averse-aggresive'] = -self.normalized_factor_value_df['SD']
        # 기준 2
        self.normalized_factor_value_df['passive-active'] = -self.normalized_factor_value_df['TrackingError']
        # 기준 3
        self.normalized_factor_value_df['winratio-big jump'] = self.normalized_factor_value_df['UpsideFrequency'] - \
                                                               self.normalized_factor_value_df['MaxReturn']

        # 기준 4
        self.normalized_factor_value_df['value-growth'] = (self.normalized_factor_value_df['Value_f_mean'] +
                                                          self.normalized_factor_value_df['Contrarian_f_mean']) - \
                                                          (self.normalized_factor_value_df['Growth_f_mean'] +
                                                          self.normalized_factor_value_df['Momentum_f_mean'])

        # 기준 5
        self.normalized_factor_value_df['dividend-investment'] = self.normalized_factor_value_df['Dividend_f_mean'] - \
                                                               self.normalized_factor_value_df['Investment_f_mean']

    def normalized_df(self, df: pd.DataFrame) -> pd.DataFrame:
        normalized_df = df.apply(lambda x: self.calculation_normalized(x), axis=0)
        return normalized_df

    def calculation_normalized(self, data_list: list) -> pd.Series:
        data_list = pd.Series(data_list)
        if data_list.std() == 0:
            normal = data_list * 0 + 1
        else:
            normal = (data_list - data_list.mean()) / data_list.std()
        return normal

if __name__ == '__main__':
    data_container = DataContainer()

