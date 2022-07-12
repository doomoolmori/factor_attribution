import pandas as pd
import numpy as np


using_factor = ['Bankruptcy Score_f_mean', 'Contrarian_f_mean', 'Dividend_f_mean', 'Earning Momentum_f_mean',
             'Earnings Quality_f_mean', 'Financial Quality_f_mean', 'Growth_f_mean', 'Information Uncertainty_f_mean',
             'Investment_f_mean', 'Leverage_f_mean', 'Liquidity_f_mean', 'Momentum_f_mean', 'Size_f_mean',
             'Value_f_mean',
             'V1_f_mean',
             'sector_centralization',
             'Return', 'SD', 'Sharpe', 'MinReturn', 'MaxReturn',
             'UpsideFrequency', 'UpCapture', 'DownCapture', 'UpNumber', 'DownNumber', 'UpPercent', 'DownPercent',
             'AverageDrawdown.Factor', 'maxDrawdown',
             'TrackingError', 'PainIndex.Factor',
             'AverageLength.Factor', 'AverageRecovery.Factor',
             'CDD', 'VaR.Factor', 'CVaR.Factor',
             'Alpha', 'Beta', 'Beta.Bull', 'Beta.Bear',
             'turnover',
             'cpi_beta', 'cpi_beta.bull', 'cpi_beta.bear',
             'realrate_beta', 'realrate_beta.bull', 'realrate_beta.bear']


def calculation_normalized(data_list: list) -> pd.Series:
    data_list = pd.Series(data_list)
    if data_list.std() == 0:
        normal = data_list * 0 + 1
    else:
        normal = (data_list - data_list.mean()) / data_list.std()
    return normal

raw_df = pd.read_csv('multifactor_characters_dt.csv', index_col=0)
factor_df = raw_df[using_factor]
normalized_factor_df = factor_df.apply(lambda x: calculation_normalized(x), axis=0)

X = normalized_factor_df
from sklearn.decomposition import PCA
pca = PCA(n_components=15) # 주성분을 몇개로 할지 결정
printcipalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data=printcipalComponents)
exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)
(X @ pca.components_.T) - (X @ pca.components_.T).mean(0)

result = pd.DataFrame(pca.components_.T)
result.index = normalized_factor_df.columns
#result.to_csv('pca.csv')

