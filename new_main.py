import sys
from PyQt5.QtWidgets import *
import data_process
import default_process

if __name__ == '__main__':
    app = QApplication(sys.argv)
    data_container = data_process.DataContainer()
    window = default_process.DefaultProcessWidget(data_container)
    window.show()
    app.exec_()

