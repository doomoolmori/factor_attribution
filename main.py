import attribution_calculation as ac
import widget_plot
import sys
import widget_attribution
from PyQt5.QtWidgets import QApplication

if __name__ == '__main__':
    calculation = ac.Calculation()
    app = QApplication(sys.argv)

    #window = widget_plot.MyWindow(calculation)
    window1 = widget_attribution.Attribution(calculation)
    window1.combo_grid_box_event()
    window1.show()
    app.exec_()

