import pandas as pd
import numpy as np

class Calculation():
    def __init__(self):
        self.char_list = ['Price', 'Dividend', 'Momentum', 'Investment', 'Risk', 'WinRatio', 'Market']
        self.char_factor_mapping_dict = {'Price': ['Value.1'],
                                    'Dividend': ['Dividend'],
                                    'Momentum': ['Momentum.1'],
                                    'Investment': ['Investment.1'],
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
        self.factor_weight_df = factor_characters_df.iloc[:, :15]

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

        """
        ###
        Y = average_weight_df
        X = np.array([x + 1 for x in range(len(Y.columns))][::-1])
        cv = pd.DataFrame(np.cov(Y, X))
        beta = cv.loc[len(cv) - 1][:-1]/np.diag(cv)[-1]
        expected = pd.DataFrame(np.ones(np.shape(Y)) * X * np.reshape(beta.values, (15, 1)))
        t_value = beta.values/((Y.values - expected.values).std(1)/np.sqrt(9))

        #p_value[p_value < 0] = np.NAN
        filtered_factor = Y.index[t_value > 0.5]
        dropped_factor = Y.index.drop(filtered_factor)

        query = ''
        for x in dropped_factor:
            if ' ' in x:
                query += (f'`{x}` == 0 and ')
            else:
                query += (f'{x} == 0 and ')

        filtered_factor_characters_df = self.factor_characters_df.query(query[:-4])
        filtered_normalized_factor_df = self.make_normalized_factor_df(factor_characters_df=
                                                                       filtered_factor_characters_df)


        unique_score = np.unique(np.abs(scaled_attribution_score_list))
        unique_score.sort()
        unique_score = unique_score[::-1]
        for score in unique_score:
            print(score)



        np.abs(scaled_attribution_score_list)

        ## 상대 필터링

        100 * (0.8 ** 2)

        ## 절대 필터링


        result['average_weight_df'] = average_weight_df

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
    attribution_score_list = np.array([-2, 0, 0, 0, 0, 0, 0])
    calculation.make_result_dict(attribution_score_list)
    import numpy as np

    np.array([1, 2, 3, 4, 2]) / 4