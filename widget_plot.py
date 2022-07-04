from matplotlib.backends.backend_qt5agg import FigureCanvas as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import cm
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import pandas as pd
import numpy as np


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
            self.price_box.addItem(f'price_attribution : {i}')
            self.dividend_box.addItem(f'dividend_attribution : {i}')
            self.momentum_box.addItem(f'momentum_attribution : {i}')
            self.investment_box.addItem(f'investment_attribution : {i}')
            self.risk_box.addItem(f'risk_attribution : {i}')
            self.winratio_box.addItem(f'winratio_attribution : {i}')
            self.market_box.addItem(f'market_attribution : {i}')
        self.price_box.setCurrentText(f'price_attribution : 0')
        self.dividend_box.setCurrentText(f'dividend_attribution : 0')
        self.momentum_box.setCurrentText(f'momentum_attribution : 0')
        self.investment_box.setCurrentText(f'investment_attribution : 0')
        self.risk_box.setCurrentText(f'risk_attribution : 0')
        self.winratio_box.setCurrentText(f'winratio_attribution : 0')
        self.market_box.setCurrentText(f'market_attribution : 0')

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

        self.table_beta = QWidget()
        self.table_final = QWidget()

        self.table_(self.table_beta, result['beta_df'], 'beta')
        self.table_(self.table_final, result['final_factor_df'], 'final_df_sharp_sort')


    def surface_plot(self, df):
        self.surface_dialog = QWidget()
        bar_width = 0.5/(len(df.columns)) # 0.1
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
        self.surface_dialog.ax1.set_xticks(np.arange(bar_width, 14 + bar_width, 1))
        self.surface_dialog.ax1.set_xticklabels(list(df.index), fontsize=6)
        self.surface_dialog.ax1.set_yticks(np.arange(bar_width, len(df.columns) + bar_width, 1))
        self.surface_dialog.ax1.set_yticklabels(list(df.columns), fontsize=6)
        self.surface_dialog.show()
