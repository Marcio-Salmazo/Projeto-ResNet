import sys
from PyQt5.QtWidgets import QApplication
from Interface import Interface

# OBSERVAÇÃO:
# Tensorflow 2.10.0
# Python 9
# Numpy 1.23.5
# Scipy 1.13.1
# Protobuf 3.20.2
# Tensorboard 2.10.1

class Main:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.interface = Interface()

    def run(self):
        self.interface.show()
        self.interface.resize(800, 600)
        sys.exit(self.app.exec())

if __name__ == "__main__":
    main = Main()
    main.run()
