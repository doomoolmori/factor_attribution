import pandas as pd
import numpy as np
from itertools import *


class Calculation():
    def __init__(self):
        self.char_list = ['Price', 'Dividend', 'Momentum', 'Investment', 'Risk', 'WinRatio', 'Market']
        self.char_factor_mapping_dict = {'Price': ['Value_f_mean'],
                                    'Dividend': ['Dividend_f_mean'],
                                    'Momentum': ['Momentum_f_mean'],
                                    'Investment': ['Investment_f_mean'],
                                    'Risk': ['SD'],
                                    'WinRatio': ['UpsideFrequency'],
                                    'Market': ['TrackingError']}
        # 낮은 score에 더 높은 비중을 줘야하는 것
        score_inverse_list = ['SD',
                              'TrackingError']
        factor_characters_df = pd.read_csv('multifactor_characters_dt.csv', index_col=0)
        for factor in score_inverse_list:
            factor_characters_df = self.value_inverse(factor=factor, factor_characters_df=factor_characters_df)
        self.factor_characters_df = factor_characters_df
        self.factor_weight_df = factor_characters_df.iloc[:, :14]

    def value_inverse(self, factor: str, factor_characters_df: pd.DataFrame) -> pd.DataFrame:
        factor_characters_df[factor] = 1 / factor_characters_df[factor]
        return factor_characters_df

    def calculation_normalized(self, data_list: list) -> pd.Series:
        data_list = pd.Series(data_list)
        if data_list.std() == 0:
            normal = data_list * 0 + 1
        else:
            normal = (data_list - data_list.mean()) / data_list.std()
        return normal

    def make_attribution_and_factor_weight_array(self, factor_weight_df: pd.DataFrame,
                                                 qcut_df: pd.DataFrame, quantile: int) -> np.array:
        # axis 0 : attribution, axis 1 : factor weight, axis 2 : quantile array 생성
        result_array = []
        for i in range(quantile):
            same_q_df = (qcut_df == i) * 1
            temp = factor_weight_df.T @ same_q_df.values
            temp = temp / np.reshape(same_q_df.sum(0).values, (1, len(same_q_df.columns)))
            result_array.append(temp.values)
        result_array = np.array(result_array)
        result_array = np.swapaxes(result_array, 0, 2)
        return result_array

    def make_normalized_ca(self, ca_list:list):
        ca_list = np.array(ca_list)
        if np.ndim(ca_list) == 1:
            result = ca_list / np.abs(ca_list).max()
            return result
        elif np.ndim(ca_list) == 2:
            ca_list = np.array(ca_list)
            ca_list[np.abs(ca_list).sum(1) == 0, :] += 1
            result = ca_list / np.reshape(np.abs(ca_list).max(1), (len(ca_list), 1))
            return result

    def make_normalized_factor_df(self, factor_characters_df: pd.DataFrame) -> pd.DataFrame:
        using_factor_attribution = [x[0] for x in self.char_factor_mapping_dict.values()]
        normalized_factor_df = factor_characters_df[using_factor_attribution].\
                               apply(lambda x: self.calculation_normalized(x), axis=0)
        return normalized_factor_df

    def asv_quantile_average_weight(self, asv: pd.DataFrame, quantile: int) -> np.array:
        qcut_df = asv.apply(lambda x: pd.qcut(x, quantile, labels=False))
        qcut_df = (quantile - 1) - qcut_df
        result_array = self.make_attribution_and_factor_weight_array(self.factor_weight_df, qcut_df, quantile)
        return result_array

    def quantile_weight_beta(self, average_weight_df: pd.DataFrame, quantile: int) -> pd.DataFrame:
        Y = average_weight_df
        X = np.array([x + 1 for x in range(len(Y.columns))][::-1])
        cov = pd.DataFrame(np.cov(Y, X))
        beta = cov.iloc[-1, :-1] / np.diag(cov)[-1]
        expected = pd.DataFrame(np.ones(np.shape(Y)) * X * np.reshape(beta.values, (len(beta), 1)))
        t_value = beta.values / ((Y.values - expected.values).std(1) / np.sqrt(quantile - 1))
        beta_df = pd.concat([beta, pd.DataFrame(t_value)], 1)
        beta_df.columns = ['beta', 't_value']
        beta_df.index = average_weight_df.index
        beta_df['contain'] = 0
        beta_df['contain'].loc[Y.index[t_value > 0.5]] = 1
        return beta_df

    def re_filtering(self, beta_df: pd.DataFrame, normalized_ca: list) -> pd.DataFrame:
        # 3단계 하위 필터링 하는 단계
        # TODO 더 스무스한 로직이 필요 우선은 포문으로 돌려놈..
        filtered_factor = beta_df['contain'][beta_df['contain'] == 1].index  # Y.index[t_value > 0.5]
        dropped_factor = beta_df['contain'][beta_df['contain'] == 0].index
        query = ''
        for x in dropped_factor:
            if ' ' in x:
                query += (f'`{x}` == 0 and ')
            else:
                query += (f'{x} == 0 and ')
        self.filtered_factor_characters_df = self.factor_characters_df.query(query[:-4])
        filtered_normalized_factor_df = self.make_normalized_factor_df(factor_characters_df=
                                                                       self.filtered_factor_characters_df)
        abs_unique_score = np.sort(np.unique(np.abs(normalized_ca)))[::-1]
        re_filtered_normalized_factor_df = filtered_normalized_factor_df.copy()
        self.cutting_dict = {}
        for abs_score in abs_unique_score:
            print(abs_score)
            temp_list = []
            temp_abs_binary_score = np.abs(normalized_ca) == abs_score
            for i, binary_score in enumerate(temp_abs_binary_score):
                if binary_score == True and normalized_ca[i] >= 0:
                    join_idx = re_filtered_normalized_factor_df.iloc[:, i].sort_values(ascending=False) \
                        [0:int(len(re_filtered_normalized_factor_df) * 0.8)].index
                    temp_list.append(join_idx)
                elif binary_score == True and normalized_ca[i] < 0:
                    join_idx = (-re_filtered_normalized_factor_df.iloc[:, i]).sort_values(ascending=False) \
                        [0:int(len(re_filtered_normalized_factor_df) * 0.8)].index
                    temp_list.append(join_idx)
                else:
                    join_idx = re_filtered_normalized_factor_df.index
                self.cutting_dict[self.char_list[i]] = join_idx
            temp_idx = temp_list[0]
            if len(temp_list) > 0:
                for i in range(1, len(temp_list)):
                    temp_idx = temp_idx.intersection(temp_list[i])
            re_filtered_normalized_factor_df = re_filtered_normalized_factor_df.loc[temp_idx.copy()]
        return re_filtered_normalized_factor_df

    def init_setting(self, ca_list: list) -> dict:
        self.normalized_ca = self.make_normalized_ca(ca_list)
        self.normalized_factor_df = self.make_normalized_factor_df(factor_characters_df=self.factor_characters_df)
        self.asv = pd.DataFrame(self.normalized_factor_df.values @
                                                     self.normalized_ca)


    def make_result_dict(self, ca_list:list) -> dict:
        # 1 단계 average_weight_df 를 구하는 단계
        result = {}

        self.init_setting(ca_list)
        quantile = 10

        ## 1단계 특성, 팩터비중, 특성 퀀타일 어레이 생성
        quantile_weight_array = self.asv_quantile_average_weight(self.asv, quantile)

        ## for 문 돌면서 모든 경우 처리
        average_weight_df = pd.DataFrame(quantile_weight_array[0, :, :])
        average_weight_df.index = self.factor_weight_df.columns
        average_weight_df.columns = [f'{x + 1}_q' for x in range(quantile)]

        ## 2단계 특성점수(퀀타일)와 팩터비중간 리그레션
        quantile_weight_beta_df = self.quantile_weight_beta(average_weight_df, quantile)

        ## 3단계 특성 순위별로 필터링 하는 단계
        re_filtered_normalized_factor_df = self.re_filtering(quantile_weight_beta_df, self.normalized_ca)

        ## 4단계 절대적으로 필터링 하는 단계
        final_df = self.filtered_factor_characters_df.loc[re_filtered_normalized_factor_df.index].copy()
        final_df = final_df.sort_values(by='Sharpe', ascending=False)

        result['average_weight_df'] = average_weight_df
        result['beta_df'] = quantile_weight_beta_df
        for key in self.cutting_dict.keys():
            print(key)
            self.filtered_factor_characters_df[f'{key}_join'] = 0
            self.filtered_factor_characters_df[f'{key}_join'].loc[self.cutting_dict[key]] = 1
        result['filtered_factor_df'] = self.filtered_factor_characters_df
        result['final_factor_df'] = final_df

        key = result.keys()
        with pd.ExcelWriter(f'out/{ca_list}.xlsx') as writer:
            for key_name in key:
                result[key_name].to_excel(writer, sheet_name=key_name)
        return result


    def make_result_all(self) -> dict:
        # 1 단계 average_weight_df 를 구하는 단계
        score_set = [-2, -1, 0, 1, 2]
        total_ca = list(product(score_set, repeat=len(self.char_list)))
        total_ca = np.array(total_ca)
        total_ca[np.abs(total_ca).sum(1) == 0, :] += 1
        total_normalized_ca = total_ca / np.reshape(np.abs(total_ca).max(1),
                                                          (len(total_ca), 1))
        normalized_factor_df = self.make_normalized_factor_df(factor_characters_df=self.factor_characters_df)
        asv = pd.DataFrame(normalized_factor_df.values @ total_normalized_ca.T)

        quantile = 10
        qcut_df = asv.apply(lambda x: pd.qcut(x, quantile, labels=False), axis=0)
        qcut_df = (quantile - 1) - qcut_df
        quantile_weight_array = self.make_attribution_and_factor_weight_array(self.factor_weight_df, qcut_df, quantile)

        ## for 문 돌면서 모든 경우 처리
        average_weight_df = pd.DataFrame(quantile_weight_array[0, :, :])
        average_weight_df.index = self.factor_weight_df.columns
        average_weight_df.columns = [f'{x + 1}_q' for x in range(quantile)]

        ## 2단계 특성점수(퀀타일)와 팩터비중간 리그레션
        quantile_weight_beta_df = self.quantile_weight_beta(average_weight_df, quantile)

        ## 3단계 특성 순위별로 필터링 하는 단계
        re_filtered_normalized_factor_df = self.re_filtering(quantile_weight_beta_df, self.normalized_ca)

        ## 4단계 절대적으로 필터링 하는 단계
        final_df = self.filtered_factor_characters_df.loc[re_filtered_normalized_factor_df.index].copy()
        final_df = final_df.sort_values(by='Sharpe', ascending=False)


if __name__ == '__main__':
    calculation = Calculation()
    ca_list = np.array([-2, -2, -2, -2, -2, -2, -2])
    #calculation.make_result_all()
    calculation.make_result_dict(ca_list)
    # import time
    # start = time.time()
    #print("time :", time.time() - start)
