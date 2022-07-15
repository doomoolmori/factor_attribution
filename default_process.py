import pandas as pd
import numpy as np
from itertools import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import cm
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import pandas as pd
import numpy as np
import widget_plot as wp
import optional_process as op


# widget 파트
class DefaultProcessWidget(QMainWindow):
    def __init__(self, data_container):
        super().__init__()
        self.data_container = data_container
        self.default_process_calculation = DefaultProcessCalculation(data_container)
        self.setGeometry(100, 200, 500, 600)

        self.default_process_calculation.default_data_setting()

        self.make_standard_grid_combo_list()
        self.standard_grid_combo_list_set()

        self.make_standard_score_combo_list()
        self.standard_score_combo_list_set()
        self.grid_combo_box_event()

        self.make_standard_lock_btn_list()
        self.standard_lock_btn_list_set()

        self.clear_btn = QPushButton("clear", self)
        self.clear_btn.setGeometry(350, 200, 50, 30)
        self.clear_btn.clicked.connect(self.clear_)

        self.setting_btn = QPushButton("set", self)
        self.setting_btn.setGeometry(350, 100, 50, 30)
        self.setting_btn.clicked.connect(self.setting_)
        self.show()

    def clear_(self):
        self.default_process_calculation.default_data_setting()
        for number, lock in enumerate(self.standard_lock_list):
            lock.setEnabled(True)

        for number, score in enumerate(self.standard_score_combo_list):
            number = number + 1
            score.clear()
            self.score_combo_set(score, number=number)

    def setting_(self):
        self.default_process_calculation.total_score_vector_calculation()
        self.w = op.OptionalProcessWidget(self.data_container)
        self.w.show()

    def make_standard_score_combo_list(self):
        self.standard_1_score_combo = QComboBox(self)
        self.standard_2_score_combo = QComboBox(self)
        self.standard_3_score_combo = QComboBox(self)
        self.standard_4_score_combo = QComboBox(self)
        self.standard_score_combo_list = [self.standard_1_score_combo, self.standard_2_score_combo,
                                          self.standard_3_score_combo, self.standard_4_score_combo]

    def make_standard_grid_combo_list(self):
        self.standard_1_grid_combo = QComboBox(self)
        self.standard_2_grid_combo = QComboBox(self)
        self.standard_3_grid_combo = QComboBox(self)
        self.standard_4_grid_combo = QComboBox(self)
        self.standard_grid_combo_list = [self.standard_1_grid_combo, self.standard_2_grid_combo,
                                         self.standard_3_grid_combo, self.standard_4_grid_combo]

    def standard_score_combo_list_set(self):
        for number, combo in enumerate(self.standard_score_combo_list):
            number = number + 1
            self.score_combo_set(combo, number=number)

    def standard_grid_combo_list_set(self):
        for number, combo in enumerate(self.standard_grid_combo_list):
            number = number + 1
            self.grid_combo_set(combo, number=number)

    def grid_combo_set(self, widget, number):
        widget.clear()
        attribution_name_list = list(self.default_process_calculation.default_attribution_dict.keys())
        for i in range(len(attribution_name_list)):
            widget.addItem(f' 특성({attribution_name_list[i]}) {number} : {i}')
        widget.setGeometry(10, 100 * number - 30, 250, 30)

    def score_combo_set(self, widget, number):
        widget.clear()
        grid_i = self.standard_grid_combo_list[number - 1].currentText().split(': ')[-1]
        print("grid_i")
        print(grid_i)
        attribution_name_list = list(self.default_process_calculation.default_attribution_dict.keys())
        range_ = [x for x in range(1, self.default_process_calculation.quantile + 1)]
        for i in range_:
            widget.addItem(f' 특성({attribution_name_list[int(grid_i)]}) {number} : {i}')
        widget.setGeometry(10, 100 * number, 250, 30)

    def changed_print(self, value):
        print("changed_print")
        print(value)
        changed_combo_grid_index = int(value.split(' : ')[0].split(')')[-1])
        self.score_combo_set(widget=self.standard_score_combo_list[changed_combo_grid_index - 1],
                             number=changed_combo_grid_index)
        print("changed_combo_grid_index")
        print(changed_combo_grid_index)

    def grid_combo_box_event(self):
        self.standard_1_grid_combo.currentTextChanged.connect(self.changed_print)
        self.standard_2_grid_combo.currentTextChanged.connect(self.changed_print)
        self.standard_3_grid_combo.currentTextChanged.connect(self.changed_print)
        self.standard_4_grid_combo.currentTextChanged.connect(self.changed_print)

    def make_standard_lock_btn_list(self):
        self.standard_1_lock = QPushButton('1-lock', self)
        self.standard_2_lock = QPushButton('2-lock', self)
        self.standard_3_lock = QPushButton('3-lock', self)
        self.standard_4_lock = QPushButton('4-lock', self)

        self.standard_lock_list = [self.standard_1_lock, self.standard_2_lock,
                                   self.standard_3_lock, self.standard_4_lock]

    def standard_lock_btn_list_set(self):
        for number, lock in enumerate(self.standard_lock_list):
            number = number + 1
            self.lock_btn_set(lock, number=number)

    def lock_btn_set(self, widget, number):
        widget.setGeometry(260, 100 * number, 50, 30)
        widget.clicked.connect(self.btn_locking)

    def btn_locking(self):
        event = self.sender()
        number = int(event.text().split('-')[0])
        score = int(self.standard_score_combo_list[number - 1].currentText().split(' : ')[-1])
        key_index = int(self.standard_grid_combo_list[number - 1].currentText().split(' : ')[-1])
        self.standard_lock_list[number - 1].setDisabled(True)
        self.default_process_calculation.locking_item_calculation(number=number,
                                                                  score=score,
                                                                  key_index=key_index)
####


# calculation 파트
class DefaultProcessCalculation():
    def __init__(self, data_container):
        self.data_container = data_container
        self.quantile = 10
        self.default_attribution_dict = {
            'risk averse-aggresive': 'risk averse-aggresive',
            'value-growth': 'value-growth',
            'passive-active': 'passive-active',
            'winratio-big jump': 'winratio-big jump'
        }

    def default_data_setting(self):
        self.df = self.data_container.normalized_factor_value_df
        self.qcut_df = self.df.apply(lambda x: pd.qcut(x, self.quantile, labels=False, duplicates='drop'))
        self.qcut_df += 1
        self.first_standard_df = 0
        self.second_standard_df = 0
        self.third_standard_df = 0
        self.fourth_standard = 0

    def locking_item_calculation(self, number: int, score: int, key_index: int):
        print("--- locking_item_calculation -- ")
        key_list = list(self.default_attribution_dict.keys())
        self.first_standard = self.default_attribution_dict[key_list[key_index]]
        self.second_standard = self.default_attribution_dict[key_list[key_index]]
        self.third_standard = self.default_attribution_dict[key_list[key_index]]
        self.fourth_standard = self.default_attribution_dict[key_list[key_index]]
        print("locked key :  " + str(key_list[key_index]))

        if number == 1:
            if score == '무관':
                self.first_standard_df = self.qcut_df.copy()
                self.first_original_df = self.df.copy()
            elif int(score) <= 5:
                score = int(score)
                self.first_standard_df = self.qcut_df[self.qcut_df[self.first_standard] <= 5].copy()
                self.first_original_df = self.df[self.qcut_df[self.first_standard] <= 5].copy()

            else:
                self.first_standard_df = self.qcut_df[self.qcut_df[self.first_standard] > 5].copy()
                self.first_original_df = self.df[self.qcut_df[self.first_standard] > 5].copy()

            print(self.first_standard_df)
            self.first_score = score

            self.first_standard_df = self.first_original_df.apply(
                lambda x: pd.qcut(x, self.quantile, labels=False, duplicates='drop'))
            self.first_standard_df += 1

        elif number == 2:
            if score == '무관':
                self.second_standard_df = self.first_standard_df.copy()
                self.second_original_df = self.first_original_df.copy()
            elif int(score) <= 5:
                self.second_standard_df = self.first_standard_df[
                    self.first_standard_df[self.second_standard] <= 5].copy()
                self.second_original_df = self.first_original_df[
                    self.first_standard_df[self.second_standard] <= 5].copy()

            else:
                self.second_standard_df = self.first_standard_df[
                    self.first_standard_df[self.second_standard] > 5].copy()
                self.second_original_df = self.first_original_df[
                    self.first_standard_df[self.second_standard] > 5].copy()

            print(self.second_standard_df)
            self.second_score = score
            self.second_standard_df = self.second_original_df.apply(
                lambda x: pd.qcut(x, self.quantile, labels=False, duplicates='drop'))
            self.second_standard_df += 1


        elif number == 3:
            if score == '무관':
                self.third_standard_df = self.second_standard_df.copy()
                self.third_original_df = self.second_original_df.copy()

            elif int(score) <= 5:
                self.third_standard_df = self.second_standard_df[
                    self.second_standard_df[self.third_standard] <= int(5)].copy()
                self.third_original_df = self.second_original_df[
                    self.second_standard_df[self.third_standard] <= int(5)].copy()

            else:
                self.third_standard_df = self.second_standard_df[
                    self.second_standard_df[self.third_standard] > int(5)].copy()
                self.third_original_df = self.second_original_df[
                    self.second_standard_df[self.third_standard] > int(5)].copy()

            print(self.third_standard_df)
            self.third_score = score
            self.third_standard_df = self.third_original_df.apply(
                lambda x: pd.qcut(x, self.quantile, labels=False, duplicates='drop'))
            self.third_standard_df += 1

        elif number == 4:
            if score == '무관':
                self.fourth_standard_df = self.third_standard_df.copy()
                self.fourth_original_df = self.third_original_df.copy()

            elif int(score) <= 5:
                self.fourth_standard_df = self.third_standard_df[
                    self.third_standard_df[self.fourth_standard] <= int(5)].copy()
                self.fourth_original_df = self.third_original_df[
                    self.third_standard_df[self.fourth_standard] <= int(5)].copy()

            else:
                self.fourth_standard_df = self.third_standard_df[
                    self.third_standard_df[self.fourth_standard] > int(5)].copy()
                self.fourth_original_df = self.third_original_df[
                    self.third_standard_df[self.fourth_standard] > int(5)].copy()

            print(self.fourth_standard_df)
            self.fourth_score = score

    def total_score_vector_calculation(self):
        score_sum_list = []
        weight_list = [1, 0.85, 0.7, 0.6]
        standard_score_list = np.array([int(self.first_score), int(self.second_score),
                                        int(self.third_score), int(self.fourth_score)])
        standard_list = np.array([self.first_standard, self.second_standard,
                                  self.third_standard, self.fourth_standard])
        if standard_score_list.std() != 0:
            standard_score_list = (standard_score_list - standard_score_list.sum()) / standard_score_list.std()
        score_df = self.fourth_original_df
        for i, number in enumerate([1, 2, 3, 4]):
            score_sum_list.append(score_df[standard_list[i]] * weight_list[i] * (int(standard_score_list[i])))

        self.data_container.default_process_tsv = pd.concat(score_sum_list, 1).sum(1).sort_values()[::-1]
        self.data_container.default_process_df = self.fourth_original_df
        #print(self.data_container.default_process_df)
