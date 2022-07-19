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



class OptionalProcessWidget(QMainWindow):
    def __init__(self, data_container):
        super().__init__()

        self.opc = OptionalProcessCalculation(data_container)
        print(self.opc.data_container.default_process_df)
        self.setGeometry(100, 200, 500, 400)
        self.setWindowTitle("PyQt")

        self.make_optional_combo_box_dict()
        self.make_default_top_n_btn()
        self.make_optional_top_n_btn()

        self.setting_btn = QPushButton("setting", self)
        self.setting_btn.move(300, 200)
        self.setting_btn.clicked.connect(self.setting_dialog_open)

    def make_optional_combo_box_dict(self):
        self.optional_combo_box_dict = {}
        for i, optional in enumerate(self.opc.char_list):
            self.optional_combo_box_dict[f'{optional}_box'] = QComboBox(self)
            self.optional_combo_box_dict[f'{optional}_box'].setGeometry(10, 60 * (i + 1), 200, 30)
            for i in range(-2, 3):
                self.optional_combo_box_dict[f'{optional}_box'].addItem(f'{optional}_attribution : {i}')
            self.optional_combo_box_dict[f'{optional}_box'].setCurrentText(f'{optional}_attribution : 0')

    def make_default_top_n_btn(self):
        self.default_top_N = QComboBox(self)
        self.default_top_N.setGeometry(300, 100, 200, 30)
        for i in range(1, 11):
            self.default_top_N.addItem(f'default_top_N : {i * 50}')
        self.default_top_N.setCurrentText(f'default_top_N : 100')

    def make_optional_top_n_btn(self):
        self.optional_top_N = QComboBox(self)
        self.optional_top_N.setGeometry(300, 150, 200, 30)
        for i in range(1, 11):
            self.optional_top_N.addItem(f'optional_top_N : {i * 10}')
        self.optional_top_N.setCurrentText(f'optional_top_N : 100')


    def close_widget(self):
        try:
            self.surface_and_bar_widget.close()
        except:
            pass
        try:
            self.box_widget.close()
        except:
            pass
        try:
            self.scatter_widget.close()
        except:
            pass
        try:
            self.table_widget.close()
        except:
            pass

    def setting_dialog_open(self):
        self.close_widget()

        ca_list = []
        for optional in self.opc.char_list:
            ca_list.append(int(self.optional_combo_box_dict[f'{optional}_box'].currentText().split(' : ')[-1]))
        ca_list = np.array(ca_list)
        if np.abs(ca_list).sum() == 0:
            ca_list += 1
        self.ca_list = ca_list
        default_top_N = int(self.default_top_N.currentText().split(' : ')[-1])
        optional_top_N = int(self.optional_top_N.currentText().split(' : ')[-1])

        print(ca_list)
        calculation_result = self.opc.make_result_dict(ca_list=ca_list,
                                                       default_top_n=default_top_N,
                                                       optional_top_n=optional_top_N)

        self.surface_and_bar_widget = QWidget()
        self.box_widget = QWidget()
        self.scatter_widget = QWidget()
        self.table_widget = QWidget()

        self.weight_surface_and_bar(widget=self.surface_and_bar_widget, calculation_result=calculation_result)
        self.score_box_plot(widget=self.box_widget, calculation_result=calculation_result)
        self.score_scatter_plot(widget=self.scatter_widget, calculation_result=calculation_result)
        self.final_table(widget=self.table_widget, calculation_result=calculation_result)

    ### Surface_and_bar
    def weight_surface_and_bar(self, widget, calculation_result):
        df = calculation_result['average_weight_df']
        self.surface_and_bar_canvas(widget)
        self.plot_weight_surface(widget, df)
        self.plot_weight_bar(widget, df)

    def surface_and_bar_canvas(self, widget):
        widget.setWindowTitle("surface")
        widget.setGeometry(500, 100, 2000, 1000)

        canvas_bar = FigureCanvas(Figure(figsize=(4, 3)))
        vbox = QVBoxLayout(widget)
        vbox.addWidget(canvas_bar)
        widget.weight_bar = canvas_bar.figure.subplots()

        canvas_surface = FigureCanvas(Figure(figsize=(4, 3)))
        vbox.addWidget(canvas_surface)
        widget.weight_surface = canvas_surface.figure.add_subplot(1, 1, 1, projection='3d')
        widget.show()

    def plot_weight_surface(self, widget, df):
        bar_width = 0.5 / (len(df.columns))  # 0.1
        widget.weight_surface.clear()
        X, Y = np.meshgrid(np.array([x for x in range(0, len(df.columns))]),
                           np.array([x for x in range(0, len(df))]))

        widget.weight_surface.plot_surface(Y, X, df.values, cmap=cm.gist_rainbow)
        widget.weight_surface.set_xticks(np.arange(bar_width, 14 + bar_width, 1))
        widget.weight_surface.set_xticklabels(list(df.index), fontsize=6)
        widget.weight_surface.set_yticks(np.arange(bar_width, len(df.columns) + bar_width, 1))
        widget.weight_surface.set_yticklabels(list(df.columns), fontsize=6)

    def plot_weight_bar(self, widget, df):
        bar_width = 0.5 / (len(df.columns))  # 0.1
        year = list(df.index)
        index = np.arange(len(df))
        for i, q_name in enumerate(df.columns):
            widget.weight_bar.bar(index + i * bar_width, df[q_name], bar_width, alpha=0.8, label=q_name)

        widget.weight_bar.set_xticks(np.arange(bar_width, len(df) + bar_width, 1))
        widget.weight_bar.set_xticklabels(labels=year, fontsize=8)
        widget.weight_bar.set_xlabel('factor')
        widget.weight_bar.set_ylabel('average_weight')
        widget.weight_bar.legend()

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
        df = result['normalized_factor_df'][self.opc.char_factor_mapping_dict.values()]
        final_df = result['final_factor_df'][self.opc.char_factor_mapping_dict.values()]
        widget.box0.boxplot(np.array(df))
        for i, score_name in enumerate(self.opc.char_list):
            final_score = df[self.opc.char_factor_mapping_dict[score_name]].loc[final_df.index]
            y = np.array([final_score.mean() for x in range(5)])
            x = np.array([i + 0.98, i + 0.99, i + 1, i + 1.01, i + 1.02])
            widget.box0.plot(x, y, 'r.', alpha=1)
        widget.box0.set_xticklabels(labels=self.opc.char_list, fontsize=8)
        title = ''
        for i, name in enumerate(self.opc.char_list):
            title = f'{title}  {name}:{self.ca_list[i]}'
        widget.box0.set_title(title)

        df1 = result['normalized_factor_df'][box_plot_factor]
        widget.box1.boxplot(np.array(df1))
        for i, score_name in enumerate(df1.columns):
            final_score = df1[score_name].loc[final_df.index]
            y = np.array([final_score.mean() for x in range(5)])
            x = np.array([i + 0.98, i + 0.99, i + 1, i + 1.01, i + 1.02])
            widget.box1.plot(x, y, 'r.', alpha=1)
        title_list = []
        for name in box_plot_factor:
            title_list.append(name.split('_')[0])
        widget.box1.set_xticklabels(labels=title_list, fontsize=8)

    def score_scatter_plot(self, widget, calculation_result):
        self.three_d_scatter_canvas(widget)
        self.plot_score_scatter(widget, calculation_result)

    def three_d_scatter_canvas(self, widget):
        widget.setWindowTitle("3d scatter")
        widget.setGeometry(500, 100, 2000, 1000)

        canvas_scatter = FigureCanvas(Figure(figsize=(4, 3)))
        vbox = QVBoxLayout(widget)
        vbox.addWidget(canvas_scatter)
        widget.scatter = [canvas_scatter.figure.add_subplot(1, 2, 1, projection='3d'),
                          canvas_scatter.figure.add_subplot(1, 2, 2, projection='3d')]

        widget.x0_box = QComboBox(widget)
        widget.x0_box.setGeometry(10, 10, 200, 30)
        widget.y0_box = QComboBox(widget)
        widget.y0_box.setGeometry(10, 60, 200, 30)
        widget.z0_box = QComboBox(widget)
        widget.z0_box.setGeometry(10, 110, 200, 30)

        widget.x1_box = QComboBox(widget)
        widget.x1_box.setGeometry(1000, 10, 200, 30)
        widget.y1_box = QComboBox(widget)
        widget.y1_box.setGeometry(1000, 60, 200, 30)
        widget.z1_box = QComboBox(widget)
        widget.z1_box.setGeometry(1000, 110, 200, 30)

        for factor_name in using_factor:
            widget.x0_box.addItem(f'x : {factor_name}')
            widget.y0_box.addItem(f'y : {factor_name}')
            widget.z0_box.addItem(f'z : {factor_name}')
            widget.x1_box.addItem(f'x : {factor_name}')
            widget.y1_box.addItem(f'y : {factor_name}')
            widget.z1_box.addItem(f'z : {factor_name}')

        widget.x0_box.setCurrentText(f'x : Beta')
        widget.y0_box.setCurrentText(f'y : TrackingError')
        widget.z0_box.setCurrentText(f'z : Alpha')
        widget.x1_box.setCurrentText(f'x : Return')
        widget.y1_box.setCurrentText(f'y : SD')
        widget.z1_box.setCurrentText(f'z : maxDrawdown')

        widget.scatter0_btn = QPushButton("scatter0", widget)
        widget.scatter0_btn.move(310, 10)
        widget.scatter1_btn = QPushButton("scatter1", widget)
        widget.scatter1_btn.move(1310, 10)
        widget.show()

    def plot_score_scatter(self, widget, result):
        final_df = result['final_factor_df']
        widget.scatter0_btn.clicked.connect(lambda: self.scatter_3D(final_df=final_df, plot_number=0, widget=widget))
        widget.scatter1_btn.clicked.connect(lambda: self.scatter_3D(final_df=final_df, plot_number=1, widget=widget))

    def scatter_3D(self, final_df: pd.DataFrame, plot_number: int, widget):
        widget.scatter[plot_number].clear()
        if plot_number == 0:
            x = widget.x0_box.currentText().split(' : ')[-1]
            y = widget.y0_box.currentText().split(' : ')[-1]
            z = widget.z0_box.currentText().split(' : ')[-1]
        elif plot_number == 1:
            x = widget.x1_box.currentText().split(' : ')[-1]
            y = widget.y1_box.currentText().split(' : ')[-1]
            z = widget.z1_box.currentText().split(' : ')[-1]
        X = self.opc.data_container.normalized_factor_value_df[x]
        Y = self.opc.data_container.normalized_factor_value_df[y]
        Z = self.opc.data_container.normalized_factor_value_df[z]

        all_x = np.array(X)
        all_y = np.array(Y)
        all_z = np.array(Z)
        widget.scatter[plot_number].scatter(all_x, all_y, all_z, marker='o', s=5, cmap='Blues', alpha=0.1)

        final_x = np.array(X.loc[final_df.index])
        final_y = np.array(Y.loc[final_df.index])
        final_z = np.array(Z.loc[final_df.index])

        widget.scatter[plot_number].scatter(final_x, final_y, final_z, marker='o', s=40, cmap='Greens')
        for i in range(len(final_df.index)):
            widget.scatter[plot_number].text3D(final_x[i], final_y[i], final_z[i],
                                               str(i) + " : " + str(final_df.index[i]))

        widget.scatter[plot_number].set_xlabel(f'{x}')
        widget.scatter[plot_number].set_ylabel(f'{y}')
        widget.scatter[plot_number].set_zlabel(f'{z}')

    ### table
    def final_table(self, widget, calculation_result):
        df = calculation_result['final_factor_df']
        self.table_canvas(widget)
        self.create_table_widget(widget.table, df)

    def table_canvas(self, widget):
        widget.setGeometry(600, 100, 1200, 800)
        widget.setWindowTitle('final_df')
        widget.table = QTableWidget(widget)
        widget.table.resize(2000, 1000)
        widget.show()

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


class OptionalProcessCalculation():
    def __init__(self, data_container):

        self.data_container = data_container

        self.char_factor_mapping_dict = {'Value': 'Value_f_mean',
                                         'Dividend': 'Dividend_f_mean',
                                         'Momentum': 'Momentum_f_mean',
                                         'Investment': 'Investment_f_mean',
                                         'Risk': 'SD',
                                         'WinRatio': 'UpsideFrequency',
                                         'Market': 'TrackingError'}
        self.char_list = list(self.char_factor_mapping_dict.keys())


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

    def make_normalized_ca(self, ca_list: list):
        ca_list = np.array(ca_list)
        if np.ndim(ca_list) == 1:
            result = ca_list / np.abs(ca_list).max()
            return result
        elif np.ndim(ca_list) == 2:
            ca_list = np.array(ca_list)
            ca_list[np.abs(ca_list).sum(1) == 0, :] += 1
            result = ca_list / np.reshape(np.abs(ca_list).max(1), (len(ca_list), 1))
            return result

    def asv_quantile_average_weight(self, factor_weight_df: pd.DataFrame, asv: pd.DataFrame, quantile: int) -> np.array:
        qcut_df = asv.apply(lambda x: pd.qcut(x, quantile, labels=False, duplicates='drop'))
        qcut_df = (quantile - 1) - qcut_df
        result_array = self.make_attribution_and_factor_weight_array(factor_weight_df, qcut_df, quantile)
        return result_array

    def make_average_weight_df(self, weight_df: pd.DataFrame, score_vector: pd.DataFrame, quantile: int = 10):
        quantile_weight_array = self.asv_quantile_average_weight(weight_df, score_vector, quantile)
        average_weight_df = pd.DataFrame(quantile_weight_array[0, :, :])
        average_weight_df.index = self.data_container.factor_weight_df.columns
        average_weight_df.columns = [f'{x + 1}_q' for x in range(quantile)]
        return average_weight_df

    def make_result_dict(self, ca_list: list, default_top_n: int = 100, optional_top_n: int = 100) -> dict:
        # 1 단계 average_weight_df 를 구하는 단계
        result = {}
        quantile = 10
        default_top_n_score_vector = (self.data_container.default_process_tsv.sort_values()[::-1]).iloc[:default_top_n]
        default_top_n_index = default_top_n_score_vector.index
        default_top_n_weight_df = self.data_container.factor_weight_df.loc[default_top_n_index]
        default_top_n_factor_df = self.data_container.normalized_factor_value_df.loc[default_top_n_index]

        default_top_n_average_weight_df = self.make_average_weight_df(weight_df=default_top_n_weight_df,
                                                                      score_vector=pd.DataFrame(default_top_n_score_vector),
                                                                      quantile=quantile)

        optional_top_n_score_vector = ((default_top_n_factor_df[self.char_factor_mapping_dict.values()] @ \
                                      self.make_normalized_ca(ca_list)).sort_values()[::-1]).iloc[:optional_top_n]
        optional_top_n_index = optional_top_n_score_vector.index
        optional_top_n_weight_df = self.data_container.factor_weight_df.loc[optional_top_n_index]
        optional_top_n_factor_df = self.data_container.normalized_factor_value_df.loc[optional_top_n_index]
        ## TODO 최종값 필터링 하는 어떠한 로직이 들어와야함
        
        
        result['average_weight_df'] = default_top_n_average_weight_df
        result['final_factor_df'] = optional_top_n_factor_df
        result['normalized_factor_df'] = self.data_container.normalized_factor_value_df

        key = result.keys()
        with pd.ExcelWriter(f'out/{ca_list}.xlsx') as writer:
            for key_name in key:
                result[key_name].to_excel(writer, sheet_name=key_name)

        return result


if __name__ == '__main__':
    aa = OptionalProcessCalculation()
