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
"""
import polars as pl

def rank(_exp, method='average', ascending=True):
    # Fill nans so as not to affect ranking
    fill = np.Inf if ascending else -np.Inf
    reverse = False if ascending else True
    tmp = pl.when(_exp.is_not_null()).then(_exp).otherwise(fill).rank(reverse=reverse, method=method)
    # Plug nans back in
    exp = pl.when(_exp.is_not_null()).then(tmp).otherwise(_exp)
    return exp

import polars as pl
data = pl.read_csv('multifactor_characters_dt.csv', dtype={'sedol': pl.Utf8}, quote_char="'", low_memory=False)


raw_data_name = '2022-05-27_cosmos-univ-with-factors_with-finval_global_monthly.csv'
universe = 'Univ_KOSPI&KOSDAQ'
sector = 'Sector_1'
raw_data = pl.read_csv(raw_data_name, quote_char="'", low_memory=False, dtype={'sedol': pl.Utf8})
raw_data = raw_data.filter((pl.col(universe) == 1)).sort('date_')


value_dict = {'DeepValue': True,
              'Inverse PEG ( PEG =PER / 12 month forward eps growth)': True,
              'Price/Cash Flow - Current': False,
              'Price/Earnings Ratio - Close': False,
              'Price/Earnings Ratio - Current': False,
              'Price/Sales': False,
              'StableValue': True}

value = 0
for value_name in value_dict.keys():
    temp_df = raw_data.pivot(index='date_', columns='infocode', values=value_name)
    trans_matrix = temp_df.select(pl.all().exclude('date_')).transpose()
    value += trans_matrix.select(rank(pl.all(), ascending=value_dict[value_name])).transpose().to_numpy()

momentum = 0
for momentum_name in momentum_dict.keys():
    temp_df = raw_data.pivot(index='date_', columns='infocode', values=momentum_name)
    trans_matrix = temp_df.select(pl.all().exclude('date_')).transpose()
    momentum += trans_matrix.select(rank(pl.all(), ascending=momentum_dict[momentum_name])).transpose().to_numpy()

value = pd.DataFrame(value).rank(1).to_numpy()
momentum = pd.DataFrame(momentum).rank(1).to_numpy()

available = (~np.isnan(value) * ~np.isnan(momentum)) * 1.0

value[np.isnan(value)] = 0
momentum[np.isnan(momentum)] = 0

import torch
import numpy as np

0.0179 - 0.2 * (0.0047) + 0.2 * (-0.0019)
# value랑 momentum

weight = torch.tensor([0.50, 0.50], requires_grad=True)
# nan = torch.tensor([[1., 1.],
#                    [0, 1.]], requires_grad=True)
nan = torch.tensor(available, requires_grad=True)

# factor 계산할 때 없으면 엄청 낮은 값을 넣자
# factor_score1 = torch.tensor([[2., 3.],
#                              [-1, 1.]], requires_grad=True)  # 최초 Tensor 객체
factor_score1 = torch.tensor(value, requires_grad=True)  # 최초 Tensor 객체
factor_score2 = torch.tensor(momentum, requires_grad=True)  # 최초 Tensor 객체

# factor_score2 = torch.tensor([[2.2, 1.3],
#                              [-1, 10.1]], requires_grad=True)  # 최초 Tensor 객체

pct_change = torch.tensor(adj_pct_change.fillna(0).to_numpy(), requires_grad=True)

# pct_change = torch.tensor([[0.2, -0.3],
#                           [0.5, 0.1]], requires_grad=True)  # 최초 Tensor 객체

rank_sum = factor_score1 * weight[0] + factor_score2 * weight[1]

soft_max_weight = torch.softmax(rank_sum, dim=1) * nan
soft_max_weight = torch.multiply(soft_max_weight, torch.reshape(1 / (soft_max_weight).sum(1), (234, 1)))

returns = torch.matmul(soft_max_weight, pct_change.T)

total = torch.diag(returns)
sum_end = total.mean()
# sum_std = total.std()
# result = sum_end/sum_std
sum_end.backward()
print(sum_end)
print(weight.grad)
"""

##

