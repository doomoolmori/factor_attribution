import pandas as pd
import numpy as np


def get_char_score_dict(char_list: list) -> dict:
    score_dict = {}
    for char in char_list:
        score = int(input(f"{char}_score : "))
        assert 0 <= score & score <= 4, '0 <= score <= 4'
        score_dict[f'{char}'] = score
    return score_dict


def value_inverse(factor: str, factor_characters_df: pd.DataFrame) -> pd.DataFrame:
    factor_characters_df[factor] = 1 / factor_characters_df[factor]
    return factor_characters_df


def calculation_zero_to_one(data_list: list) -> pd.Series:
    data_list = pd.Series(data_list)
    zero_to_one = data_list / data_list.max()
    return zero_to_one


def calculation_normalized(data_list: list) -> pd.Series:
    data_list = pd.Series(data_list)
    if data_list.std() == 0:
        normal = data_list * 0 + 1
    else:
        normal = (data_list - data_list.mean()) / data_list.std()
    return normal


def calculation_char_score(char_zero_to_one_score_dict: dict,
                           char_factor_mapping_dict: dict,
                           factor_characters_df: pd.DataFrame) -> pd.Series:
    temp_list = []
    for char in char_zero_to_one_score_dict.keys():
        char_zero_to_one_score = char_zero_to_one_score_dict[char]
        char_factor_list = char_factor_mapping_dict[char]
        factor_normalized_score = factor_characters_df[char_factor_list]. \
            apply(lambda x: calculation_normalized(x), axis=0).mean(1)
        temp_list.append(factor_normalized_score * char_zero_to_one_score)
    return pd.concat(temp_list, 1).sum(1)


if __name__ == '__main__':
    char_list = ['Price', 'Dividend', 'Momentum', 'Investment', 'Risk', 'WinRatio', 'Market']

    char_factor_mapping_dict = {'Price': ['Value.1'],
                                'Dividend': ['Dividend'],
                                'Momentum': ['Momentum.1'],
                                'Investment': ['Investment.1'],
                                'Risk': ['SD'],
                                         #'MinReturn',
                                         #'AverageDrawdown.Factor',
                                         #'maxDrawdown',
                                         #'PainIndex.Factor',
                                         #'AverageLength.Factor',
                                         #'CDD',
                                         #'VaR.Factor',
                                         #'CVaR.Factor'],
                                'WinRatio': ['UpsideFrequency'],
                                'Market': ['TrackingError']}

    # 낮은 score에 더 높은 비중을 줘야하는 것
    score_inverse_list = ['SD',
                          #'AverageDrawdown.Factor',
                          #'maxDrawdown',
                          #'PainIndex.Factor',
                          #'AverageLength.Factor',
                          #'CDD',
                          'TrackingError']
                          #'Beta']

    factor_characters_df = pd.read_csv('multifactor_characters_dt.csv', index_col=0)
    for factor in score_inverse_list:
        factor_characters_df = value_inverse(factor=factor, factor_characters_df=factor_characters_df)


    char_score_dict = get_char_score_dict(char_list)
    char_score_value = list(char_score_dict.values())
    char_zero_to_one_score_value = calculation_zero_to_one(char_score_value)
    char_zero_to_one_score_dict = dict(zip(char_score_dict.keys(),
                                           char_zero_to_one_score_value))

    for i in range(0, 199):
        score_sum = calculation_char_score(char_zero_to_one_score_dict,
                                           char_factor_mapping_dict,
                                           factor_characters_df)
        print(i)

    result = factor_characters_df.iloc[np.argmax(score_sum), :]
    print(result[:15])




    from itertools import *

    factor_weight_df = factor_characters_df.iloc[:, :15]

    score_set = [0, 1, 2, 3, 4]
    total_score_set = list(product(score_set, repeat=len(char_list)))
    total_score_set = np.array(total_score_set)
    total_score_set = total_score_set[1:, :]

    ## X
    total_normalized_score_set = total_score_set/np.reshape(total_score_set.max(1), (len(total_score_set), 1))
    X = total_normalized_score_set

    ## Y
    n_top = 30
    n_bottom = 30
    attribution_list = [x[0] for x in char_factor_mapping_dict.values()]

    normalized_factor_df = factor_characters_df[attribution_list].apply(lambda x: calculation_normalized(x), axis=0)

    temp = X @ normalized_factor_df.T.values
    rank_df = pd.DataFrame(temp).rank(1, ascending=False)
    top_switch_df = (rank_df <= 30) * 1
    bottom_switch_df = (rank_df >= len(rank_df.columns) - 30) * 1


    top_Y = (top_switch_df.values @ factor_weight_df).values/n_top
    bottom_Y = (bottom_switch_df.values @ factor_weight_df).values/n_bottom

    """
    top_delta = pd.DataFrame(np.linalg.pinv(X) @ top_Y)
    top_delta.columns = factor_weight_df.columns
    top_delta.index = char_list
    top_delta.to_csv('top_delta.csv')
    
    
    bottom_delta = pd.DataFrame(np.linalg.pinv(X) @ bottom_Y)
    bottom_delta.columns = factor_weight_df.columns
    bottom_delta.index = char_list
    bottom_delta.to_csv('bottom_delta.csv')
    """

    """
    delta = pd.DataFrame(np.linalg.pinv(top_Y) @ X)
    delta.columns = char_list
    delta.index = factor_weight_df.columns
    delta['percent'] = np.mean(top_Y, 0)
    delta.to_csv('delta_t.csv')
    
    
    
    delta = pd.DataFrame(np.linalg.pinv(bottom_Y) @ X)
    delta.columns = char_list
    delta.index = factor_weight_df.columns
    delta['percent'] = np.mean(bottom_Y, 0)
    delta.to_csv('delta_b.csv')
    
    
    ###
    import torch
    
    learning_rate = 1e-3
    
    x_array = torch.tensor(X)
    delta_df = torch.randn((len(char_list), len(factor_weight_df.columns)), dtype=torch.float64, requires_grad=True)
    y_array = torch.tensor(top_Y)
    
    
    for t in range(10000):
    
        y_pred = torch.matmul(x_array, delta_df)
    
        loss = (y_array - y_pred).pow(2).sum(1).sqrt().mean()
        #norm_loss = ((loss)/loss.std()).sum()
        loss.backward()
    
        print(t, loss.item())
        with torch.no_grad():
            delta_df -= learning_rate * delta_df.grad
    
            # 가중치 갱신 후에는 변화도를 직접 0으로 만듭니다.
            delta_df.grad = None
    
    
    pd.DataFrame(delta_df.to_array).to_csv('top_delta.csv')
    """