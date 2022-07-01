import attribution_calculation as ac
import widget_plot
import sys
from PyQt5.QtWidgets import QApplication

if __name__ == '__main__':
    calculation = ac.Calculation()
    app = QApplication(sys.argv)
    window = widget_plot.MyWindow(calculation)
    window.show()
    app.exec_()

