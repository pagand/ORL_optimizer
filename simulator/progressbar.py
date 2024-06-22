import sys
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtCore import Qt, QRect

class CustomProgressBar(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.progress = 0

    def initUI(self):
        self.setGeometry(100, 100, 300, 50)
        self.setWindowTitle('Custom Progress Bar')
        self.show()

    def setProgress(self, value):
        self.progress = value
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        self.drawProgressBar(painter)

    def drawProgressBar(self, painter):
        painter.setBrush(QColor(230, 230, 230))
        painter.drawRect(10, 10, 280, 30)

        painter.setBrush(QColor(50, 150, 250))
        progress_width = int(280 * (self.progress / 100.0))
        painter.drawRect(10, 10, progress_width, 30)

def main():
    app = QApplication(sys.argv)
    ex = CustomProgressBar()

    import time
    for i in range(101):
        ex.setProgress(i)
        app.processEvents()
        time.sleep(0.05)

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
