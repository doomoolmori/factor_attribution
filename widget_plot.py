from matplotlib.backends.backend_qt5agg import FigureCanvas as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import cm
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import pandas as pd
import numpy as np

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

class MyWindow(QMainWindow):
    def __init__(self, calculation):
        super().__init__()
        self.calculation = calculation
        self.setGeometry(100, 200, 500, 400)
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
        for i in range(-2, 3):
            self.price_box.addItem(f'value_attribution : {i}')
            self.dividend_box.addItem(f'dividend_attribution : {i}')
            self.momentum_box.addItem(f'momentum_attribution : {i}')
            self.investment_box.addItem(f'investment_attribution : {i}')
            self.risk_box.addItem(f'risk_attribution : {i}')
            self.winratio_box.addItem(f'winratio_attribution : {i}')
            self.market_box.addItem(f'market_attribution : {i}')
        self.price_box.setCurrentText(f'value_attribution : 0')
        self.dividend_box.setCurrentText(f'dividend_attribution : 0')
        self.momentum_box.setCurrentText(f'momentum_attribution : 0')
        self.investment_box.setCurrentText(f'investment_attribution : 0')
        self.risk_box.setCurrentText(f'risk_attribution : 0')
        self.winratio_box.setCurrentText(f'winratio_attribution : 0')
        self.market_box.setCurrentText(f'market_attribution : 0')
        setting_btn = QPushButton("setting", self)
        setting_btn.move(300, 150)
        setting_btn.clicked.connect(self.setting_dialog_open)

    def setting_dialog_open(self):
        ca_list = np.array([int(self.price_box.currentText().split(' : ')[-1]), \
                            int(self.dividend_box.currentText().split(' : ')[-1]), \
                            int(self.momentum_box.currentText().split(' : ')[-1]), \
                            int(self.investment_box.currentText().split(' : ')[-1]), \
                            int(self.risk_box.currentText().split(' : ')[-1]), \
                            int(self.winratio_box.currentText().split(' : ')[-1]), \
                            int(self.market_box.currentText().split(' : ')[-1])])
        if ca_list.sum() == 0:
            ca_list += 1
        self.ca_list = ca_list
        print(ca_list)
        calculation_result = self.calculation.make_result_dict(ca_list=ca_list, top_N=30)

        self.surface_and_bar_widget = QWidget()
        self.box_and_scatter_widget = QWidget()
        self.table_widget = QWidget()

        self.weight_surface_and_bar(widget=self.surface_and_bar_widget, calculation_result=calculation_result)
        self.score_box_and_scatter(widget=self.box_and_scatter_widget, calculation_result=calculation_result)
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
    def score_box_and_scatter(self, widget, calculation_result):
        self.box_and_scatter_canvas(widget)
        self.plot_score_box(widget, calculation_result)
        self.plot_score_scatter(widget, calculation_result)

    def box_and_scatter_canvas(self, widget):
        widget.setWindowTitle("box_scatter")
        widget.setGeometry(500, 100, 2000, 1000)

        canvas_box = FigureCanvas(Figure(figsize=(4, 3)))
        vbox = QVBoxLayout(widget)
        vbox.addWidget(canvas_box)
        widget.box = canvas_box.figure.subplots()

        canvas_scatter = FigureCanvas(Figure(figsize=(4, 3)))
        vbox.addWidget(canvas_scatter)
        widget.scatter = [canvas_scatter.figure.add_subplot(1, 2, 1, projection='3d'),
                          canvas_scatter.figure.add_subplot(1, 2, 2, projection='3d')]

        widget.x0_box = QComboBox(widget)
        widget.x0_box.setGeometry(10, 500, 200, 30)
        widget.y0_box = QComboBox(widget)
        widget.y0_box.setGeometry(10, 550, 200, 30)
        widget.z0_box = QComboBox(widget)
        widget.z0_box.setGeometry(10, 600, 200, 30)

        widget.x1_box = QComboBox(widget)
        widget.x1_box.setGeometry(1000, 500, 200, 30)
        widget.y1_box = QComboBox(widget)
        widget.y1_box.setGeometry(1000, 550, 200, 30)
        widget.z1_box = QComboBox(widget)
        widget.z1_box.setGeometry(1000, 600, 200, 30)

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
        widget.scatter0_btn.move(310, 500)
        widget.scatter1_btn = QPushButton("scatter1", widget)
        widget.scatter1_btn.move(1310, 500)
        widget.show()

    def plot_score_box(self, widget, result):
        df = result['normalized_factor_df']
        final_df = result['final_factor_df']
        widget.box.boxplot(np.array(df))
        for i, score_name in enumerate(self.calculation.char_list):
            final_score = df[self.calculation.char_factor_mapping_dict[score_name]].loc[final_df.index]
            y = np.array([final_score.mean() for x in range(5)])
            x = np.array([i + 0.98, i + 0.99, i + 1, i + 1.01, i + 1.02])
            widget.box.plot(x, y, 'r.', alpha=1)
        widget.box.set_xticklabels(labels=self.calculation.char_list, fontsize=8)
        title = ''
        for i, name in enumerate(self.calculation.char_list):
            title = f'{title}  {name}:{self.ca_list[i]}'
        widget.box.set_title(title)

    def plot_score_scatter(self, widget, result):
        final_df = result['final_factor_df']
        widget.scatter0_btn.clicked.connect(lambda: self.scatter_3D(final_df=final_df, plot_number=0, widget=widget))
        widget.scatter1_btn.clicked.connect(lambda: self.scatter_3D(final_df=final_df, plot_number=1, widget=widget))


    def scatter_3D(self, final_df:pd.DataFrame, plot_number:int, widget):
        widget.scatter[plot_number].clear()
        if plot_number == 0:
            x = widget.x0_box.currentText().split(' : ')[-1]
            y = widget.y0_box.currentText().split(' : ')[-1]
            z = widget.z0_box.currentText().split(' : ')[-1]
        elif plot_number == 1:
            x = widget.x1_box.currentText().split(' : ')[-1]
            y = widget.y1_box.currentText().split(' : ')[-1]
            z = widget.z1_box.currentText().split(' : ')[-1]
        X = self.calculation.original_factor_characters_df[x]
        X = (X - X.mean()) / X.std()
        Y = self.calculation.original_factor_characters_df[y]
        Y = (Y - Y.mean()) / Y.std()
        Z = self.calculation.original_factor_characters_df[z]
        Z = (Z - Z.mean()) / Z.std()

        all_x = np.array(X)
        all_y = np.array(Y)
        all_z = np.array(Z)
        widget.scatter[plot_number].scatter(all_x, all_y, all_z, marker='o', s=5, cmap='Blues', alpha=0.1)

        final_x = np.array(X.loc[final_df.index])
        final_y = np.array(Y.loc[final_df.index])
        final_z = np.array(Z.loc[final_df.index])
        widget.scatter[plot_number].scatter(final_x, final_y, final_z, marker='o', s=40, cmap='Greens')

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




