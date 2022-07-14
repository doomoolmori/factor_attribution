from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import cm
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import pandas as pd
import numpy as np
import widget_plot as wp

using_factor = ['Bankruptcy Score_f_mean', 'Contrarian_f_mean', 'Dividend_f_mean', 'Earning Momentum_f_mean',
             'Earnings Quality_f_mean', 'Financial Quality_f_mean', 'Growth_f_mean', 'Information Uncertainty_f_mean',
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
             'Earnings Quality_f_mean', 'Financial Quality_f_mean', 'Growth_f_mean', 'Information Uncertainty_f_mean',
             'Investment_f_mean', 'Leverage_f_mean', 'Liquidity_f_mean', 'Momentum_f_mean', 'Size_f_mean',
             'Value_f_mean']


class Attribution(QMainWindow):
    def __init__(self, calculation):
        super().__init__()

        self.attribution_dict = {
            'risk averse-aggresive': 'risk averse-aggresive',
            'value-growth': 'value-growth',
            'passive-active' : 'passive-active',
            'winratio-big jump' : 'winratio-big jump'
        }

        self.calculation = calculation
        self.quantile = 10
        self.factor_data_setting()
        self.setGeometry(100, 200, 500, 600)

        self.make_combo_box_list()
        self.combo_box_list_set()

        self.make_lock_btn_list()
        self.lock_btn_list_set()

        self.clear_btn = QPushButton("clear", self)
        self.clear_btn.setGeometry(350, 200, 50, 30)
        self.clear_btn.clicked.connect(self.clear_)

        self.setting_btn = QPushButton("set", self)
        self.setting_btn.setGeometry(350, 100, 50, 30)
        self.setting_btn.clicked.connect(self.setting_)

        self.show()


    def factor_data_setting(self):
        self.df = self.calculation.original_factor_characters_df
        self.qcut_df = self.df.apply(lambda x: pd.qcut(x, self.quantile, labels=False, duplicates='drop'))
        self.qcut_df += 1
        self.first_standard_df = 0
        self.second_standard_df = 0
        self.third_standard_df = 0

    def clear_(self):
        self.factor_data_setting()
        print(self.third_standard_df)
        for number, lock in enumerate(self.standard_lock_list):
            lock.setEnabled(True)

        for number, combo in enumerate(self.standard_combo_list):
            number = number + 1
            combo.clear()
            self.combo_box_set(combo, number=number)

    def make_combo_box_list(self):
        self.standard_grid_1_combo = QComboBox(self)
        self.standard_1_combo = QComboBox(self)

        self.standard_grid_2_combo = QComboBox(self)
        self.standard_2_combo = QComboBox(self)

        self.standard_grid_3_combo = QComboBox(self)
        self.standard_3_combo = QComboBox(self)

        self.standard_grid_4_combo = QComboBox(self)
        self.standard_4_combo = QComboBox(self)

        self.standard_grid_combo_list = [self.standard_grid_1_combo, self.standard_grid_2_combo, self.standard_grid_3_combo, self.standard_grid_4_combo]
        self.standard_combo_list = [self.standard_1_combo, self.standard_2_combo, self.standard_3_combo, self.standard_4_combo]

    def combo_box_list_set(self):
        for number, combo in enumerate(self.standard_grid_combo_list):
            number = number + 1
            self.combo_grid_set(combo, number=number)

        for number, combo in enumerate(self.standard_combo_list):
            number = number + 1
            self.combo_box_set(combo, number=number)

    def combo_grid_set(self, widget, number):
        widget.clear()
        # widget.addItem(f' 특성({list(self.attribution_dict.keys())[number-1]}) {number} : 무관')
        for i in range(len(list(self.attribution_dict.keys()))):
            widget.addItem(f' 특성({list(self.attribution_dict.keys())[i]}) {number} : {i}')
        # widget.setCurrentText(f' 특성({list(self.attribution_dict.keys())[number-1]}) {number} : 무관')
        widget.setGeometry(10, 100 * number - 30, 250, 30)

    def changed_print(self, value):
        print("changed_print")
        print(value)
        changed_combo_grid_index = int(value.split(' : ')[0].split(')')[-1])
        if changed_combo_grid_index == 1:
            self.combo_box_set(self.standard_combo_list[0], number=changed_combo_grid_index)
        elif changed_combo_grid_index == 2:
            self.combo_box_set(self.standard_combo_list[1], number=changed_combo_grid_index)
        elif changed_combo_grid_index == 3:
            self.combo_box_set(self.standard_combo_list[2], number=changed_combo_grid_index)
        elif changed_combo_grid_index == 4:
            self.combo_box_set(self.standard_combo_list[3], number=changed_combo_grid_index)
        print("changed_combo_grid_index")
        print(changed_combo_grid_index)

    def combo_grid_box_event(self):
        self.standard_grid_1_combo.currentTextChanged.connect(self.changed_print)
        self.standard_grid_2_combo.currentTextChanged.connect(self.changed_print)
        self.standard_grid_3_combo.currentTextChanged.connect(self.changed_print)
        self.standard_grid_4_combo.currentTextChanged.connect(self.changed_print)

    def combo_box_set(self, widget, number, range=[x for x in range(1, 10 + 1)]):
        widget.clear()
        grid_i = self.standard_grid_combo_list[number - 1].currentText().split(': ')[-1]
        print("grid_i")
        print(grid_i)
        #widget.addItem(f' 특성({list(self.attribution_dict.keys())[number-1]}) {number} : 무관')
        for i in range:
            widget.addItem(f' 특성({list(self.attribution_dict.keys())[int(grid_i)]}) {number} : {i}')
        #widget.setCurrentText(f' 특성({list(self.attribution_dict.keys())[number-1]}) {number} : 무관')
        widget.setGeometry(10, 100 * number, 250, 30)



    def make_lock_btn_list(self):
        self.standard_1_lock = QPushButton('1-lock', self)
        self.standard_2_lock = QPushButton('2-lock', self)
        self.standard_3_lock = QPushButton('3-lock', self)
        self.standard_4_lock = QPushButton('4-lock', self)

        self.standard_lock_list = [self.standard_1_lock, self.standard_2_lock, self.standard_3_lock, self.standard_4_lock]

    def lock_btn_list_set(self):
        for number, lock in enumerate(self.standard_lock_list):
            number = number + 1
            self.lock_btn_set(lock, number=number)

    def lock_btn_set(self, widget, number):
        widget.setGeometry(260, 100 * number, 50, 30)
        widget.clicked.connect(self.btn_locking)


    def btn_locking(self):
        event = self.sender()
        number = int(event.text().split('-')[0])
        self.standard_lock_list[number - 1].setDisabled(True)
        self.locking_item_calculation(number=number)
        #self.locking_combo(number=number)


    def locking_item_calculation(self, number):
        score = self.standard_combo_list[number - 1].currentText().split(' : ')[-1]
        print("--- locking_item_calculation -- ")
        dict_key_value = int( self.standard_grid_combo_list[number - 1].currentText().split(' : ')[-1] )
        key_list = list(self.attribution_dict.keys())
        first_standard = self.attribution_dict[key_list[dict_key_value]]
        second_standard = self.attribution_dict[key_list[dict_key_value]]
        third_standard = self.attribution_dict[key_list[dict_key_value]]
        fourth_standard = self.attribution_dict[key_list[dict_key_value]]
        print("locked key :  " + str(key_list[dict_key_value]) )

        if number == 1:
            if score == '무관':
                self.first_standard_df = self.qcut_df.copy()
                self.first_original_df = self.df.copy()
            elif int(score) <= 5:
                score = int(score)
                self.first_standard_df = self.qcut_df[self.qcut_df[first_standard] <= 5].copy()
                self.first_original_df = self.df[self.qcut_df[first_standard] <= 5].copy()

            else:
                self.first_standard_df = self.qcut_df[self.qcut_df[first_standard] > 5].copy()
                self.first_original_df = self.df[self.qcut_df[first_standard] > 5].copy()

            print(self.first_standard_df)
            self.first_score = score

            self.first_standard_df = self.first_original_df.apply(lambda x: pd.qcut(x, self.quantile, labels=False, duplicates='drop'))
            self.first_standard_df += 1

        elif number == 2:
            if score == '무관':
                self.second_standard_df = self.first_standard_df.copy()
                self.second_original_df = self.first_original_df.copy()
            elif int(score) <= 5:
                self.second_standard_df = self.first_standard_df[self.first_standard_df[second_standard] <= 5].copy()
                self.second_original_df = self.first_original_df[self.first_standard_df[second_standard] <= 5].copy()

            else:
                self.second_standard_df = self.first_standard_df[self.first_standard_df[second_standard] > 5].copy()
                self.second_original_df = self.first_original_df[self.first_standard_df[second_standard] > 5].copy()

            print(self.second_standard_df)
            self.second_score = score
            self.second_standard_df = self.second_original_df.apply(lambda x: pd.qcut(x, self.quantile, labels=False, duplicates='drop'))
            self.second_standard_df += 1


        elif number == 3:
            if score == '무관':
                self.third_standard_df = self.second_standard_df.copy()
                self.third_original_df = self.second_original_df.copy()

            elif int(score) <= 5:
                self.third_standard_df = self.second_standard_df[self.second_standard_df[third_standard] <= int(5)].copy()
                self.third_original_df = self.second_original_df[self.second_standard_df[third_standard] <= int(5)].copy()

            else:
                self.third_standard_df = self.second_standard_df[self.second_standard_df[third_standard] > int(5)].copy()
                self.third_original_df = self.second_original_df[self.second_standard_df[third_standard] > int(5)].copy()

            print(self.third_standard_df)
            self.third_score = score
            self.third_standard_df = self.third_original_df.apply(lambda x: pd.qcut(x, self.quantile, labels=False, duplicates='drop'))
            self.third_standard_df += 1


        elif number == 4:
            if score == '무관':
                self.fourth_standard_df = self.third_standard_df.copy()
                self.fourth_original_df = self.third_original_df.copy()

            elif int(score) <= 5:
                self.fourth_standard_df = self.third_standard_df[self.third_standard_df[fourth_standard] <= int(5)].copy()
                self.fourth_original_df = self.third_original_df[self.third_standard_df[fourth_standard] <= int(5)].copy()

            else:
                self.fourth_standard_df = self.third_standard_df[self.third_standard_df[fourth_standard] > int(5)].copy()
                self.fourth_original_df = self.third_original_df[self.third_standard_df[fourth_standard] > int(5)].copy()

            print(self.fourth_standard_df)
            self.fourth_score = score

    def locking_combo(self, number):
        key_list = list(self.attribution_dict.keys())
        first_standard = self.attribution_dict[key_list[0]]
        second_standard = self.attribution_dict[key_list[1]]
        third_standard = self.attribution_dict[key_list[2]]
        fourth_standard = self.attribution_dict[key_list[3]]

        if number == 1:
            first_possible = self.first_standard_df[first_standard].unique()
            first_possible.sort()
            self.combo_box_set(self.standard_combo_list[0], number=1, range=first_possible)
            set_name = f' 특성({list(self.attribution_dict.keys())[0]}) 1 : {self.first_score}'
            self.standard_combo_list[0].setCurrentText(set_name)

            second_possible = self.first_standard_df[second_standard].unique()
            second_possible.sort()
            self.combo_box_set(self.standard_combo_list[1], number=2, range=second_possible)
            set_name = f' 특성({list(self.attribution_dict.keys())[1]}) 2 : 무관'
            self.standard_combo_list[1].setCurrentText(set_name)

            third_possible = self.first_standard_df[third_standard].unique()
            third_possible.sort()
            self.combo_box_set(self.standard_combo_list[2], number=3, range=third_possible)
            set_name = f' 특성({list(self.attribution_dict.keys())[2]}) 3 : 무관'
            self.standard_combo_list[2].setCurrentText(set_name)

            fourth_possible = self.first_standard_df[fourth_standard].unique()
            fourth_possible.sort()
            self.combo_box_set(self.standard_combo_list[3], number=4, range=fourth_possible)
            set_name = f' 특성({list(self.attribution_dict.keys())[3]}) 3 : 무관'
            self.standard_combo_list[3].setCurrentText(set_name)

        elif number == 2:
            second_possible = self.second_standard_df[second_standard].unique()
            second_possible.sort()
            self.combo_box_set(self.standard_combo_list[1], number=2, range=second_possible)
            set_name = f' 특성({list(self.attribution_dict.keys())[1]}) 2 : {self.second_score}'
            self.standard_combo_list[1].setCurrentText(set_name)

            third_possible = self.second_standard_df[third_standard].unique()
            third_possible.sort()
            self.combo_box_set(self.standard_combo_list[2], number=3, range=third_possible)
            set_name = f' 특성({list(self.attribution_dict.keys())[2]}) 3 : 무관'
            self.standard_combo_list[2].setCurrentText(set_name)

            fourth_possible = self.second_standard_df[fourth_standard].unique()
            fourth_possible.sort()
            self.combo_box_set(self.standard_combo_list[3], number=4, range=fourth_possible)
            set_name = f' 특성({list(self.attribution_dict.keys())[3]}) 3 : 무관'
            self.standard_combo_list[3].setCurrentText(set_name)

        elif number == 3:
            third_possible = self.third_standard_df[third_standard].unique()
            third_possible.sort()
            self.combo_box_set(self.standard_combo_list[2], number=3, range=third_possible)
            set_name = f' 특성({list(self.attribution_dict.keys())[2]}) 3 : {self.third_score}'
            self.standard_combo_list[2].setCurrentText(set_name)

            fourth_possible = self.third_standard_df[fourth_standard].unique()
            fourth_possible.sort()
            self.combo_box_set(self.standard_combo_list[3], number=4, range=fourth_possible)
            set_name = f' 특성({list(self.attribution_dict.keys())[3]}) 3 : 무관'
            self.standard_combo_list[3].setCurrentText(set_name)

        elif number == 4:
            fourth_possible = self.fourth_standard_df[fourth_standard].unique()
            fourth_possible.sort()
            self.combo_box_set(self.standard_combo_list[3], number=4, range=fourth_possible)
            set_name = f' 특성({list(self.attribution_dict.keys())[3]}) 3 : 무관'
            self.standard_combo_list[3].setCurrentText(set_name)

    def get_standard_name(self, number):
        dict_key_value = int(self.standard_grid_combo_list[number - 1].currentText().split(' : ')[-1])
        key_list = list(self.attribution_dict.keys())
        standard = self.attribution_dict[key_list[dict_key_value]]
        return standard

    def final_score_calculation(self):
        score_sum_list = []
        weight_list = [1, 0.85, 0.7, 0.55]
        standard_score_list = np.array([int(self.first_score), int(self.second_score),
                                        int(self.third_score), int(self.fourth_score)])

        if standard_score_list.std() != 0:
            standard_score_list = (standard_score_list - standard_score_list.sum()) / standard_score_list.std()

        score_df = self.fourth_original_df.apply(lambda x: self.calculation.calculation_normalized(x), axis=0)
        for i, number in enumerate([1, 2, 3, 4]):
            score_sum_list.append(score_df[self.get_standard_name(number)] \
                                  * weight_list[i] * (int(standard_score_list[i])))
        
        #score top 30개 고름
        total_idx = pd.concat(score_sum_list, 1).sum(1).sort_values()[::-1][:30]

        #self.calculation.asv_4 = pd.concat(score_sum_list, 1).sum(1).sort_values()[::-1]
        #self.calculation.fourth_original_df = self.fourth_original_df
        return total_idx.index



    def setting_(self):
        if len(self.fourth_standard_df) > 0:
            total_idx = self.final_score_calculation()
            self.calculation.final_df = self.calculation.original_factor_characters_df.loc[total_idx]
        self.w = wp.MyWindow(self.calculation)
        self.w.show()