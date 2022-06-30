import sys
from matplotlib.backends.backend_qt5agg import FigureCanvas as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import cm
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import pandas as pd
import numpy as np


class MyWindow(QMainWindow):
    def __init__(self, Calculation):
        super().__init__()
        self.calculation = Calculation

        self.setGeometry(100, 200, 500, 300)
        self.setWindowTitle("PyQt")
        self.setWindowIcon(QIcon("icon.png"))

        self.price_box = QComboBox(self)
        self.price_box.setGeometry(10, 60, 200, 30)

        self.dividend_box = QComboBox(self)
        self.dividend_box.setGeometry(10, 100, 200, 30)

        self.momentum_box = QComboBox(self)
        self.momentum_box.setGeometry(10, 140, 200, 30)

        self.investment_box = QComboBox(self)
        self.investment_box.setGeometry(10, 180, 200, 30)

        self.risk_box = QComboBox(self)
        self.risk_box.setGeometry(10, 220, 200, 30)

        self.winratio_box = QComboBox(self)
        self.winratio_box.setGeometry(10, 260, 200, 30)

        self.market_box = QComboBox(self)
        self.market_box.setGeometry(10, 300, 200, 30)

        for i in range(0, 5):
            self.price_box.addItem(f'price_attribution : {i}')
            self.dividend_box.addItem(f'dividend_attribution : {i}')
            self.momentum_box.addItem(f'momentum_attribution : {i}')
            self.investment_box.addItem(f'investment_attribution : {i}')
            self.risk_box.addItem(f'risk_attribution : {i}')
            self.winratio_box.addItem(f'winratio_attribution : {i}')
            self.market_box.addItem(f'market_attribution : {i}')

        setting_btn = QPushButton("setting", self)
        setting_btn.move(300, 150)
        setting_btn.clicked.connect(self.setting_dialog_open)


    def table_open(self):
        self.table_dialog = QWidget()
        self.table_dialog.setGeometry(600, 100, 1200, 800)
        self.table_dialog.setWindowTitle("Table")

        self.table_dialog.table = QTableWidget(self.table_dialog)
        self.table_dialog.table.resize(1200, 800)
        self.table_dialog.table.setRowCount(65)
        self.table_dialog.table.setColumnCount(10)
        self.table_dialog.show()


    def create_table_widget(self, widget, df):
        widget.setRowCount(len(df.index))
        widget.setColumnCount(len(df.columns))
        widget.setHorizontalHeaderLabels(df.columns)
        widget.setVerticalHeaderLabels(np.array(df.index, dtype=str))

        for row_index, row in enumerate(df.index):
            for col_index, column in enumerate(df.columns):
                value = df.loc[row][column]
                item = QTableWidgetItem(str(value))
                widget.setItem(row_index, col_index, item)

    def top_10(self, df):
        self.table_top10 = QWidget()
        self.table_top10.setGeometry(600, 100, 1200, 800)
        self.table_top10.setWindowTitle("top10Table")
        self.table_top10.table = QTableWidget(self.table_top10)
        self.table_top10.table.resize(1200, 800)
        self.create_table_widget(self.table_top10.table, df)
        self.table_top10.show()

    def bottom_10(self, df):
        self.table_bottom10 = QWidget()
        self.table_bottom10.setGeometry(600, 100, 1200, 800)
        self.table_bottom10.setWindowTitle("top10Table")
        self.table_bottom10.table = QTableWidget(self.table_bottom10)
        self.table_bottom10.table.resize(1200, 800)
        self.create_table_widget(self.table_bottom10.table, df)
        self.table_bottom10.show()

    def table_(self, widget, df, name):
        widget.setGeometry(600, 100, 1200, 800)
        widget.setWindowTitle(name)
        widget.table = QTableWidget(widget)
        widget.table.resize(1200, 800)
        self.create_table_widget(widget.table, df)
        widget.show()

    def setting_dialog_open(self):
        attribution_score_list = np.array([int(self.price_box.currentText().split(' : ')[-1]), \
                                           int(self.dividend_box.currentText().split(' : ')[-1]), \
                                           int(self.momentum_box.currentText().split(' : ')[-1]), \
                                           int(self.investment_box.currentText().split(' : ')[-1]), \
                                           int(self.risk_box.currentText().split(' : ')[-1]), \
                                           int(self.winratio_box.currentText().split(' : ')[-1]), \
                                           int(self.market_box.currentText().split(' : ')[-1])])

        if attribution_score_list.sum() == 0:
            attribution_score_list += 1
        print(attribution_score_list)
        result = self.calculation.make_result_dict(attribution_score_list)
        self.surface_plot(result['average_weight_df'])

        self.table_top_10 = QWidget()
        self.table_bottom_10 = QWidget()

        self.table_(self.table_top_10, result['top_10_df'], 'top')
        self.table_(self.table_bottom_10, result['bottom_10_df'], 'bottom')

        #self.bottom_10(result['bottom_10_df'])

    def surface_plot(self, df):
        self.surface_dialog = QWidget()
        bar_width = 0.1
        year = list(df.index)
        index = np.arange(len(df))

        self.surface_dialog.setWindowTitle("surface")
        self.surface_dialog.setGeometry(500, 100, 1200, 800)

        canvas_bar = FigureCanvas(Figure(figsize=(4, 3)))
        vbox = QVBoxLayout(self.surface_dialog)
        vbox.addWidget(canvas_bar)

        self.surface_dialog.ax = canvas_bar.figure.subplots()
        for i, q_name in enumerate(df.columns):
            self.surface_dialog.ax.bar(index + i * bar_width, df[q_name], bar_width, alpha=0.8,  label=q_name)

        self.surface_dialog.ax.set_xticks(np.arange(bar_width, len(df) + bar_width, 1))
        self.surface_dialog.ax.set_xticklabels(labels=year, fontsize=8)
        self.surface_dialog.ax.set_xlabel('factor')
        self.surface_dialog.ax.set_ylabel('average_weight')
        self.surface_dialog.ax.legend()

        canvas_surface = FigureCanvas(Figure(figsize=(4, 3)))
        vbox.addWidget(canvas_surface)

        self.surface_dialog.ax1 = canvas_surface.figure.add_subplot(1, 1, 1, projection='3d')
        self.surface_dialog.ax1.clear()

        X, Y = np.meshgrid(np.array([x for x in range(0, len(df.columns))]),
                           np.array([x for x in range(0, len(df))]))

        self.surface_dialog.ax1.plot_surface(Y, X, df.values, cmap=cm.gist_rainbow)
        self.surface_dialog.ax1.set_xticks(np.arange(bar_width, 15 + bar_width, 1))
        self.surface_dialog.ax1.set_xticklabels(list(df.index), fontsize=6)
        self.surface_dialog.ax1.set_yticks(np.arange(bar_width, 5 + bar_width, 1))
        self.surface_dialog.ax1.set_yticklabels(list(df.columns), fontsize=6)
        self.surface_dialog.show()

    def btn_surface(self):
        result = pd.read_csv('BTC.csv', index_col=0)
        X, Y = np.meshgrid(np.array(result.columns, dtype=int) / 10, np.array(result.index, dtype=int))
        self.surface_dialog.ax[0].plot_surface(X, Y, result.values, cmap=cm.gist_rainbow)
        self.surface_dialog.ax[1].plot_surface(X, Y, result.values, cmap=cm.gist_rainbow)
        self.surface_dialog.ax[2].plot_surface(X, Y, result.values, cmap=cm.gist_rainbow)


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
        return attribution_score_list / attribution_score_list.max()

    def make_normalized_factor_df(self) -> pd.DataFrame:
        using_factor_attribution = [x[0] for x in self.char_factor_mapping_dict.values()]
        normalized_factor_df = self.factor_characters_df[using_factor_attribution].\
                               apply(lambda x: self.calculation_normalized(x), axis=0)
        return normalized_factor_df

    def make_result_dict(self, attribution_score_list:list) -> dict:
        result = {}
        scaled_attribution_score_list = self.make_scaled_attribution_score_list(attribution_score_list)
        normalized_factor_df = self.make_normalized_factor_df()
        attribution_score_vector = normalized_factor_df.values @ scaled_attribution_score_list

        attribution_score_vector_rank = pd.DataFrame(attribution_score_vector).rank()
        quantile = 5
        qcut_df = attribution_score_vector_rank.apply(lambda x: pd.qcut(x, quantile, labels=False))
        qcut_df = (quantile - 1) - qcut_df

        result_array = self.make_attribution_and_factor_weight_array(self.factor_weight_df, qcut_df, quantile)
        average_weight_df = pd.DataFrame(result_array[0, :, :])
        average_weight_df.index = self.factor_weight_df.columns
        average_weight_df.columns = [f'{x + 1}_q' for x in range(quantile)]
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



    app = QApplication(sys.argv)
    window = MyWindow(calculation)
    window.show()
    app.exec_()

