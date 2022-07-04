import pandas as pd
import numpy as np

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

    def make_scaled_attribution_score_list(self, attribution_score_list:list):
        attribution_score_list = np.array(attribution_score_list)
        return attribution_score_list / np.abs(attribution_score_list).max()

    def make_normalized_factor_df(self, factor_characters_df: pd.DataFrame) -> pd.DataFrame:
        using_factor_attribution = [x[0] for x in self.char_factor_mapping_dict.values()]
        normalized_factor_df = factor_characters_df[using_factor_attribution].\
                               apply(lambda x: self.calculation_normalized(x), axis=0)
        return normalized_factor_df

    def make_result_dict(self, attribution_score_list:list) -> dict:
        
        # 1 단계 average_weight_df 를 구하는 단계
        result = {}
        scaled_attribution_score_list = self.make_scaled_attribution_score_list(attribution_score_list)
        normalized_factor_df = self.make_normalized_factor_df(factor_characters_df=self.factor_characters_df)
        attribution_score_vector = normalized_factor_df.values @ scaled_attribution_score_list

        attribution_score_vector_rank = pd.DataFrame(attribution_score_vector).rank()
        quantile = 10
        qcut_df = attribution_score_vector_rank.apply(lambda x: pd.qcut(x, quantile, labels=False))
        qcut_df = (quantile - 1) - qcut_df

        result_array = self.make_attribution_and_factor_weight_array(self.factor_weight_df, qcut_df, quantile)
        average_weight_df = pd.DataFrame(result_array[0, :, :])
        average_weight_df.index = self.factor_weight_df.columns
        average_weight_df.columns = [f'{x + 1}_q' for x in range(quantile)]
        result['average_weight_df'] = average_weight_df


        # 2단계 quantile, average_weight 베타를 구하는 단계
        Y = average_weight_df
        X = np.array([x + 1 for x in range(len(Y.columns))][::-1])
        cov = pd.DataFrame(np.cov(Y, X))
        beta = cov.loc[len(cov) - 1][:-1]/np.diag(cov)[-1]
        expected = pd.DataFrame(np.ones(np.shape(Y)) * X * np.reshape(beta.values, (len(beta), 1)))
        t_value = beta.values/((Y.values - expected.values).std(1)/np.sqrt(quantile - 1))
        beta_df = pd.concat([beta, pd.DataFrame(t_value)], 1)
        beta_df.columns = ['beta', 't_value']
        beta_df.index = average_weight_df.index
        beta_df['contain'] = 0
        beta_df['contain'].loc[Y.index[t_value > 0.5]] = 1
        result['beta_df'] = beta_df

        filtered_factor = beta_df['contain'][beta_df['contain'] == 1].index#Y.index[t_value > 0.5]
        dropped_factor = beta_df['contain'][beta_df['contain'] == 0].index

        query = ''
        for x in dropped_factor:
            if ' ' in x:
                query += (f'`{x}` == 0 and ')
            else:
                query += (f'{x} == 0 and ')
        filtered_factor_characters_df = self.factor_characters_df.query(query[:-4])
        filtered_normalized_factor_df = self.make_normalized_factor_df(factor_characters_df=
                                                                       filtered_factor_characters_df)

        # 3단계 하위 필터링 하는 단계
        # TODO 더 스무스한 로직이 필요 우선은 포문으로 돌려놈..
        abs_unique_score = np.sort(np.unique(np.abs(scaled_attribution_score_list)))[::-1]
        abs_filtered_normalized_factor_df = filtered_normalized_factor_df.copy()
        cutting_dict = {}
        for abs_score in abs_unique_score:
            print(abs_score)
            temp_list = []
            temp_abs_binary_score = np.abs(scaled_attribution_score_list) == abs_score
            for i, binary_score in enumerate(temp_abs_binary_score):
                if binary_score == True and scaled_attribution_score_list[i] >= 0:
                    join_idx = abs_filtered_normalized_factor_df.iloc[:, i].sort_values(ascending=False)\
                               [0:int(len(abs_filtered_normalized_factor_df) * 0.8)].index
                    temp_list.append(join_idx)
                elif binary_score == True and scaled_attribution_score_list[i] < 0:
                    join_idx = (-abs_filtered_normalized_factor_df.iloc[:, i]).sort_values(ascending=False)\
                               [0:int(len(abs_filtered_normalized_factor_df) * 0.8)].index
                    temp_list.append(join_idx)
                else:
                    join_idx = abs_filtered_normalized_factor_df.index
                cutting_dict[self.char_list[i]] = join_idx

            temp_idx = temp_list[0]
            if len(temp_list) > 0:
                for i in range(1, len(temp_list)):
                    temp_idx = temp_idx.intersection(temp_list[i])
            abs_filtered_normalized_factor_df = abs_filtered_normalized_factor_df.loc[temp_idx.copy()]

        for key in cutting_dict.keys():
            print(key)
            filtered_factor_characters_df[f'{key}_join'] = 0
            filtered_factor_characters_df[f'{key}_join'].loc[cutting_dict[key]] = 1
        result['filtered_factor_df'] = filtered_factor_characters_df


        ## 4단계 절대적으로 필터링 하는 단계
        final_df = filtered_factor_characters_df.loc[abs_filtered_normalized_factor_df.index].copy()
        final_df = final_df.sort_values(by='Sharpe', ascending=False)
        result['final_factor_df'] = final_df


        top_10_index = attribution_score_vector_rank.sort_values(0)[-10:].index
        top_10_df = self.factor_characters_df.loc[self.factor_characters_df.index[top_10_index[::-1]]]
        result['top_10_df'] = top_10_df

        bottom_10_index = attribution_score_vector_rank.sort_values(0)[:10].index
        bottom_10_df = self.factor_characters_df.loc[self.factor_characters_df.index[bottom_10_index]]
        result['bottom_10_df'] = bottom_10_df



        key = result.keys()
        with pd.ExcelWriter(f'out/{attribution_score_list}.xlsx') as writer:
            for key_name in key:
                result[key_name].to_excel(writer, sheet_name=key_name)
        return result

    """
        def get_char_score_dict(self, char_list: list) -> dict:
            score_dict = {}
            for char in char_list:
                score = int(input(f"{char}_score : "))
                assert 0 <= score & score <= 4, '0 <= score <= 4'
                score_dict[f'{char}'] = score
            return score_dict

        def calculation_zero_to_one(self, data_list: list) -> pd.Series:
            data_list = pd.Series(data_list)
            zero_to_one = data_list / data_list.max()
            return zero_to_one

        def calculation_char_score(self, char_zero_to_one_score_dict: dict,
                                   char_factor_mapping_dict: dict,
                                   factor_characters_df: pd.DataFrame) -> pd.Series:
            temp_list = []
            for char in char_zero_to_one_score_dict.keys():
                char_zero_to_one_score = char_zero_to_one_score_dict[char]
                char_factor_list = char_factor_mapping_dict[char]
                factor_normalized_score = factor_characters_df[char_factor_list]. \
                    apply(lambda x: self.calculation_normalized(x), axis=0).mean(1)
                temp_list.append(factor_normalized_score * char_zero_to_one_score)
            return pd.concat(temp_list, 1).sum(1)
        """



if __name__ == '__main__':
    calculation = Calculation()
    #import time
    #start = time.time()
    attribution_score_list = np.array([-2, -2, -2, -2, -2, -2, -2])
    calculation.make_result_dict(attribution_score_list)
    #print("time :", time.time() - start)
