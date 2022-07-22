import data_process
from itertools import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import cm
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import pandas as pd
import numpy as np

using_factor = ['Bankruptcy Score_f_mean', 'Contrarian_f_mean', 'Dividend_f_mean', 'Earning Momentum_f_mean',
                'Earnings Quality_f_mean', 'Financial Quality_f_mean', 'Growth_f_mean',
                'Information Uncertainty_f_mean',
                'Investment_f_mean', 'Leverage_f_mean', 'Liquidity_f_mean', 'Momentum_f_mean', 'Size_f_mean',
                'Value_f_mean',
                'V1_f_mean', 'Academic & Educational Services_f_mean',
                'Basic Materials_f_mean', 'Consumer Non-Cyclicals_f_mean',
                'Cyclical Consumer Goods & Services_f_mean', 'Energy_f_mean',
                'Financials_f_mean', 'Healthcare_f_mean',
                'Industrials_f_mean', 'Real Estate_f_mean',
                'Technology_f_mean', 'Utilities_f_mean',
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

box_plot_factor = ['Bankruptcy Score_f_mean', 'Contrarian_f_mean', 'Dividend_f_mean', 'Earning Momentum_f_mean',
                   'Earnings Quality_f_mean', 'Financial Quality_f_mean', 'Growth_f_mean',
                   'Information Uncertainty_f_mean',
                   'Investment_f_mean', 'Leverage_f_mean', 'Liquidity_f_mean', 'Momentum_f_mean', 'Size_f_mean',
                   'Value_f_mean']

lever_list = ['risk averse-aggresive',
              'passive-active',
              'winratio-big jump',
              'value-growth',
              'dividend-investment']


class LeverTest(QMainWindow):
    def __init__(self, data_container):
        super().__init__()
        self.data_container = data_container
        self.lever_calculation = LeverCalculation(data_container)
        self.setGeometry(100, 200, 500, 400)
        self.setWindowTitle("PyQt")

        self.make_lever_combo_box_dict()

        self.setting_btn = QPushButton("setting", self)
        self.setting_btn.move(350, 150)
        self.setting_btn.clicked.connect(self.setting_dialog_open)

    def make_lever_combo_box_dict(self):
        self.lever_combo_box_dict = {}
        for i, lever in enumerate(lever_list):
            self.lever_combo_box_dict[f'{lever}_box'] = QComboBox(self)
            self.lever_combo_box_dict[f'{lever}_box'].setGeometry(10, 60 * (i + 1), 300, 30)
            for i in range(1, len(lever_list) + 1):
                self.lever_combo_box_dict[f'{lever}_box'].addItem(f'{lever}_attribution : {i}')
            self.lever_combo_box_dict[f'{lever}_box'].setCurrentText(f'{lever}_attribution : 3')

    def close_widget(self):
        try:
            self.box_widget.close()
        except:
            pass

    def setting_dialog_open(self):
        self.close_widget()
        ca_list = []
        for lever in lever_list:
            ca_list.append(int(self.lever_combo_box_dict[f'{lever}_box'].currentText().split(' : ')[-1]))
        ca_list = np.array(ca_list)
        if np.abs(ca_list).sum() == 0:
            ca_list += 1
        self.ca_list = ca_list
        print(ca_list)

        calculation_result = {}
        calculation_result['normalized_factor_df'] = self.data_container.normalized_factor_value_df
        lever_intersect_idx = self.lever_calculation.make_lever_intersection_and_valuesum_idx(ca_list=ca_list)
        calculation_result['final_factor_df'] = self.data_container.normalized_factor_value_df.loc[lever_intersect_idx]

        self.box_widget = QWidget()
        self.score_box_plot(widget=self.box_widget, calculation_result=calculation_result)

    ### Box_and_scatter
    def score_box_plot(self, widget, calculation_result):
        self.box_plot_canvas(widget)
        self.plot_score_box(widget, calculation_result)

    def box_plot_canvas(self, widget):
        widget.setWindowTitle("box_plot")
        widget.setGeometry(500, 100, 2000, 1000)

        canvas_box0 = FigureCanvas(Figure(figsize=(4, 3)))
        vbox = QVBoxLayout(widget)
        vbox.addWidget(canvas_box0)
        widget.box0 = canvas_box0.figure.subplots()

        canvas_box1 = FigureCanvas(Figure(figsize=(4, 3)))
        vbox.addWidget(canvas_box1)
        widget.box1 = canvas_box1.figure.subplots()
        widget.show()

    def plot_score_box(self, widget, result):
        df = result['normalized_factor_df']
        final_df = result['final_factor_df']

        self._plot_one_box(widget=widget.box0,
                           total_df=df,
                           sample_df=final_df,
                           label=lever_list)

        self._plot_one_box(widget=widget.box1,
                           total_df=df,
                           sample_df=final_df,
                           label=box_plot_factor)
        title = ''
        for i, name in enumerate(lever_list):
            title = f'{title}  {name}:{self.ca_list[i]}'
        widget.box0.set_title(title)

    def _plot_one_box(self, widget, total_df: pd.DataFrame, sample_df: pd.DataFrame, label: list):
        total_df = total_df[label]
        sample_df = sample_df[label]
        widget.boxplot(np.array(total_df))
        for i, score_name in enumerate(label):
            final_score = total_df[score_name].loc[sample_df.index]
            y = np.array([final_score.mean() for x in range(5)])
            x = np.array([i + 0.98, i + 0.99, i + 1, i + 1.01, i + 1.02])
            widget.plot(x, y, 'r.', alpha=1)
        widget.set_xticklabels(labels=label, fontsize=6)


class LeverCalculation():
    def __init__(self, data_container):
        self.data_container = data_container
        self.normalized_factor_df = self.data_container.normalized_factor_value_df
        # normalized_factor_df을 5퀀타일로 나눈 df
        qcut_df = self.normalized_factor_df.apply(lambda x: pd.qcut(x, 5, labels=False, duplicates='drop'))
        self.qcut_df = len(lever_list) - qcut_df
    
    # 모든 공간을 쳐내는 방식
    # (https://www.notion.so/8d62df11c324497e99f5b4b0e3e52fd9#299f66991e1f42b18658328c1e3c9c68) 의 최초로직
    def make_lever_intersection_idx(self, ca_list):
        score_dict = dict(zip(lever_list, ca_list))
        # level 마다 해당하는 전략인덱스를 저장하기 위한 리스트
        lever_universe_list = []
        # 레버들을 돌면서 공간을 쳐낸다.
        for lever in score_dict.keys():
            score = score_dict[lever]
            if lever == 'risk averse-aggresive':
                # 리스크인 경우 해당 스코어 이하의 전략들을 사용한다.
                idx = self.qcut_df[lever][(self.qcut_df[lever] <= score)].index
            else:
                # 이외의 영우 해당 스코어 -1 , +1 범위의 전략들을 사용한다.
                idx = self.qcut_df[lever][(self.qcut_df[lever] >= score - 1) & (self.qcut_df[lever] <= score + 1)].index
            lever_universe_list.append(list(idx))
        # 레버 유니버스간의 교집합을 구한다.
        intersect_idx = list(set(lever_universe_list[0]) &
                             set(lever_universe_list[1]) &
                             set(lever_universe_list[2]) &
                             set(lever_universe_list[3]) &
                             set(lever_universe_list[4]))
        # 최종 df를 구한다.
        return intersect_idx
    
    # 리스크 위주로 쳐내고, 나머지는 value썸하는 방식 
    # (https://www.notion.so/8d62df11c324497e99f5b4b0e3e52fd9#299f66991e1f42b18658328c1e3c9c68) 의 3번 방안
    def make_lever_intersection_and_valuesum_idx(self, ca_list):
        score_dict = dict(zip(lever_list, ca_list))
        # level 마다 해당하는 전략인덱스를 저장하기 위한 리스트
        lever_universe_list = []
        # 레버들을 돌면서 공간을 쳐낸다.
        standard_score = (np.array([5, 4, 3, 2, 1]) - np.array([5, 4, 3, 2, 1]).mean())/np.array([5, 4, 3, 2, 1]).std()
        epsilon = 0.0001
        for lever in score_dict.keys():
            score = score_dict[lever]
            if lever == 'risk averse-aggresive':
                # 리스크인 경우 해당 스코어 이하의 전략들을 사용한다.
                abs_risk_idx = self.qcut_df[lever][(self.qcut_df[lever] <= score)].index
            elif lever == 'passive-active':
                shift_down = 0
                relative_risk_idx = self.qcut_df[lever][(self.qcut_df[lever] == score)].index
                risk_idx = list(set(abs_risk_idx) & set(relative_risk_idx))
                while len(risk_idx) == 0:
                    shift_down += 1
                    relative_risk_idx = self.qcut_df[lever][(self.qcut_df[lever] == score - shift_down)].index
                    risk_idx = list(set(abs_risk_idx) & set(relative_risk_idx))
                    print(shift_down)
            else:
                multiply = (standard_score[score - 1] + epsilon)
                lever_universe_list.append(self.normalized_factor_df[lever].loc[risk_idx] * multiply)
                # 이외의 영우 해당 스코어 -1 , +1 범위의 전략들을 사용한다.
        score_vector = pd.concat(lever_universe_list, 1).sum(1).sort_values()[::-1]
        intersect_and_valuesum_idx = score_vector.index[:20]
        # 최종 df를 구한다.
        return intersect_and_valuesum_idx


    # 리스크로 처내고 가장 가까운 공간을 찾는 방안.
    # (https://www.notion.so/8d62df11c324497e99f5b4b0e3e52fd9#299f66991e1f42b18658328c1e3c9c68) 의 4번 방안
    def make_lever_intersection_and_near_idx(self, ca_list):
        score_dict = dict(zip(lever_list, ca_list))
        null_space_df = self.make_null_space_df()
        abs_risk_space_df = null_space_df[null_space_df['risk averse-aggresive'] == score_dict['risk averse-aggresive']]
        abs_risk_space_df = abs_risk_space_df[abs_risk_space_df['null'] == 0]

        for i, lever in enumerate(score_dict.keys()):
            if lever == 'risk averse-aggresive':
                diff = np.array(abs_risk_space_df[lever_list[i:]]) - ca_list[i:]
                distance = np.sqrt((diff ** 2).sum(1))
            else:
                diff = np.array(abs_risk_space_df[lever_list[i]]) - ca_list[i]
                distance = np.sqrt((diff ** 2))
            abs_risk_space_df = abs_risk_space_df[distance == distance.min()]

        null_space_filtered_ca_list = abs_risk_space_df[lever_list].values[0]
        near_intersect_idx = self.make_lever_intersection_idx(ca_list=null_space_filtered_ca_list)
        return near_intersect_idx

    def make_all_score_space(self):
        all_score_space = list(product([1, 2, 3, 4, 5], repeat=len(lever_list)))
        return all_score_space

    def make_null_space_df(self):
        try:
            space_df = pd.read_csv('null_space.csv', index_col=0)
        except:
            all_score_space = self.make_all_score_space()
            null_space = []
            for i in range(len(all_score_space)):
                print(i)
                temp_idx = self.make_lever_intersection_idx(ca_list=all_score_space[i])
                if len(temp_idx) == 0:
                    null_space.append(i)
            space_df = pd.DataFrame(all_score_space)
            space_df.columns = lever_list
            space_df['null'] = 0
            space_df['null'].loc[null_space] = 1
            space_df.to_csv('null_space.csv')
        return space_df

if __name__ == '__main__':

    import sys
    app = QApplication(sys.argv)
    data_container = data_process.DataContainer()
    window = LeverTest(data_container)
    window.show()
    app.exec_()

    """
    data_container = data_process.DataContainer()
    lever_calculation = LeverCalculation(data_container)
    lever_calculation.make_null_space_df()
    """






