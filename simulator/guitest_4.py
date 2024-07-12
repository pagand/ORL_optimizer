import sys
import io
import os
import time
import csv
import math
import numpy as np
from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal, QPoint, Qt
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QMainWindow, QGridLayout, QFrame, QSizePolicy
from PyQt5.QtGui import QPainter, QPolygon, QColor, QPen
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from Speedometer import Ui_Form_spd
from Compass import Ui_Form_heading

eta_str = '00:00'
index = 0

# Reads historical best table H to N.
class readTable(QWidget):
    def __init__(self):
        super(readTable, self).__init__()

        self.filePath = 'HNtop1.csv'
        self.table = self.readTable()

    def readTable(self):
        with open(self.filePath, 'r') as file:
            table = list(csv.DictReader(file))
        return table

    # Returns information of each point from the begining
    # Everytime this is called it moves to the next point
    # It also returns next point's information as optimized value
    def update(self, current_posi, next_posi, heading):
        global index
        row = self.table[index]
        lat1 = float(row['LATITUDE'])
        long1 = float(row['LONGITUDE'])
        head1 = float(row['HEADING'])
        spd = float(row['STW'])
        sfc1 = int((float(row['ENGINE_1_FUEL_CONSUMPTION']) + float(row['ENGINE_2_FUEL_CONSUMPTION']))/2)

        next_row = self.table[index + 1]
        head2 = float(next_row['HEADING'])
        sfc2 = int((float(next_row['ENGINE_1_FUEL_CONSUMPTION']) + float(next_row['ENGINE_2_FUEL_CONSUMPTION']))/2)
        if next_row:
            lat2 = float(next_row['LATITUDE'])
            long2 = float(next_row['LONGITUDE'])

        index = index + 1
        information = [lat1, long1, head1, head2, spd, 1, sfc1, sfc2]
        return information

# Compass dial widget
class compass(QWidget):
    def __init__(self):
        super(compass, self).__init__()
        self.setFixedSize(190, 210)
        self.init_ui()

    def init_ui(self):
        self.speedometer_widget = QWidget()
        self.speedometer_ui = Ui_Form_heading()
        self.speedometer_ui.setupUi(self.speedometer_widget)

        self.indicator_new = Image.open("indicator_heading.png")

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self.speedometer_widget)

        # Add a label widget to name the dial
        self.label = QLabel('Heading')
        # self.label.setStyleSheet("background: rgb(255, 0, 0 );")
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # self.label.setFixedHeight(30)
        self.label.setAlignment(Qt.AlignTop | Qt.AlignCenter)
        layout.addWidget(self.label)

        self.setLayout(layout)

    # Rorate the 'indicator_heading.png' image to the optimal position and replace the old one
    def update(self, rot_c, rot_i):
        rot_i = -(rot_i-rot_c)

        rot_indicator = QtGui.QPixmap.fromImage(ImageQt(self.indicator_new.rotate(-int(rot_i))))
        self.speedometer_ui.indicator.setPixmap(rot_indicator)

# SFC dial widget
class sfcDial(QWidget):
    def __init__(self):
        super(sfcDial, self).__init__()
        self.setFixedSize(190, 210)
        self.init_ui()

    def init_ui(self):
        self.speedometer_widget = QWidget()
        self.speedometer_ui = Ui_Form_spd()
        self.speedometer_ui.setupUi(self.speedometer_widget)

        self.needle_new = Image.open("needle_spd.png")
        self.indicator_new = Image.open("indicator_spd.png")

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self.speedometer_widget)

        # Add a label to name the dial
        self.label = QLabel('SFC')
        # self.label.setFixedSize(80, 20)
        # self.label.setAlignment(Qt.AlignCenter)
        # self.label.setStyleSheet("background: rgb(255, 0, 0 );")
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # self.label.setFixedHeight(30)
        self.label.setAlignment(Qt.AlignTop | Qt.AlignCenter)
        layout.addWidget(self.label)


        self.setLayout(layout)

    # Rotate the 'needle_spd.png' to show current SFC and 'indicator_spd.png' to show optimal SFC and replace the old ones
    def update(self, rot_n, rot_i):
        rot_n = (rot_n/960)*240
        rot_i = (rot_i/960)*240

        rot_needle = QtGui.QPixmap.fromImage(ImageQt(self.needle_new.rotate(-int(rot_n))))
        rot_indicator = QtGui.QPixmap.fromImage(ImageQt(self.indicator_new.rotate(-int(rot_i))))
        self.speedometer_ui.needle.setPixmap(rot_needle)
        self.speedometer_ui.indicator.setPixmap(rot_indicator)

# SFC plot
class Canvas(FigureCanvas):
    def __init__(self, parent):
        # self.fig, self.ax = plt.subplots(figsize=(4.05, 1.9), dpi=100)
        self.fig, self.ax = plt.subplots(figsize=(1, 1), dpi=100)
        super().__init__(self.fig)
        self.setParent(parent)

        self.data_1 = []
        self.line_1, = self.ax.plot(self.data_1, label='Current SFC')
        self.data_2 = []
        self.line_2, = self.ax.plot(self.data_2, label='Optimal SFC')

        self.ax.set(xlabel='time (min)', ylabel='SFC (L/h)', title='SFC')
        # adjust plot margin to show axis names
        self.fig.subplots_adjust(left=0.22, right=0.9, top=0.88, bottom=0.25)
        self.ax.legend()
        self.ax.grid()

    # Add each time instant's SFC and optimal SFC into the plot
    def update_plot(self, new_value_1, new_value_2):
        self.data_1.append(new_value_1)
        self.data_2.append(new_value_2)
        self.line_1.set_ydata(self.data_1)
        self.line_1.set_xdata(np.arange(len(self.data_1)))
        self.line_2.set_ydata(self.data_2)
        self.line_2.set_xdata(np.arange(len(self.data_2)))
        self.ax.relim()
        self.ax.autoscale_view()
        self.draw()

# Make the plot into a widget
class AppDemo(QWidget):
    def __init__(self):
        super().__init__()

        self.chart = Canvas(self)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.chart)

    def add_data(self, new_value_1, new_value_2):
        self.chart.update_plot(new_value_1, new_value_2)

# Information panel widget
class infoDisplay(QWidget):
    def __init__(self):
        super(infoDisplay, self).__init__()
        self.current_page = 0
        self.setFixedSize(500, 190)
        self.init_ui()

    def init_ui(self):
        self.grid = QGridLayout()
        self.grid.setSpacing(10)

        # Create the information panel with labels
        # Need to initialize self.lables with the info we want to show
        # Then add each label widget in self.labels to the grid
        self.labels = []
        self.variables = ['Latitude', 'Longitude', 'Speed', 'Heading']
        for i in range(2):
            for j in range(4):
                if (i * 4 + j) < len(self.variables):
                    label = QLabel(self.variables[i*4+j], self)
                else:
                    label = QLabel(f'Label {i * 4 + j + 1}', self)
                label.setFrameShape(QFrame.Box)
                #label.setStyleSheet("padding: 5px; border: 1px solid white; color: white;")
                #label.setStyleSheet("padding: 5px; border: 1px solid white")
                label.setStyleSheet("padding: 5px; border: 1px solid black")
                self.grid.addWidget(label, i, j)
                self.labels.append(label)

        #self.change_page(self.current_page)

        # Create buttons for changing page
        self.prev_button = QPushButton('←', self)
        self.prev_button.clicked.connect(self.prev_page)
        self.prev_button.setStyleSheet("color: white; background-color: black; border: none; font-size: 18px;")

        self.next_button = QPushButton('→', self)
        self.next_button.clicked.connect(self.next_page)
        self.next_button.setStyleSheet("color: white; background-color: black; border: none; font-size: 18px;")

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.prev_button)
        button_layout.addStretch()
        button_layout.addWidget(self.next_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(self.grid)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    # Create a method that change self.labels instead
    # Make the change_page method only change index, not label text
    def change_page(self, page):
        start_index = page * 8
        for i in range(2):
            for j in range(4):
                if (i * 4 + j) < len(self.variables):
                    label = QLabel(self.variables[i*4+j], self)
                else:
                    label = QLabel(f'Label {i * 4 + j + 1}', self)

    def prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.change_page(self.current_page)

    def next_page(self):
        self.current_page += 1
        self.change_page(self.current_page)

# Information panel widget. Not in use
class infoDisplay_old(QWidget):
    def __init__(self):
        super(infoDisplay, self).__init__()

        self.init_ui()

    # Add opt SFC, heading, and next point coordinate
    def init_ui(self):
        self.lat = QLabel('Latitude: N/A')
        self.long = QLabel('Longitude: N/A')
        self.head = QLabel('Heading: N/A')
        self.spd = QLabel('Speed: N/A')
        self.acce = QLabel('Acceleration: N/A')
        self.sfc = QLabel('SFC: N/A')
        self.eta = QLabel('ETA: N/A')

        layout = QVBoxLayout()

        layout.addWidget(self.lat)
        layout.addWidget(self.long)
        layout.addWidget(self.head)
        layout.addWidget(self.spd)
        layout.addWidget(self.acce)
        layout.addWidget(self.sfc)
        layout.addWidget(self.eta)

        self.setLayout(layout)

    def update(self, lat, long, head, spd, acce, sfc):
        self.lat.setText(f'Latitude: {lat}')
        self.long.setText(f'Longitude: {long}')
        self.head.setText(f'Heading: {head}')
        self.spd.setText(f'Speed: {spd}')
        self.acce.setText(f'Acceration: {acce}')
        self.sfc.setText(f'SFC: {sfc}')
        self.eta.setText(f'ETA: {eta_str}')

# Number pad widget
class numPad(QWidget):
    buttonClickedSignal = pyqtSignal(str)

    def __init__(self):
        super(numPad, self).__init__()

        self.init_ui()

    # Create the number pad
    def init_ui(self):
        rows = [
            ['1', '2', '3'],
            ['4', '5', '6'],
            ['7', '8', '9'],
            ['CLR', '0', 'ENT']
        ]

        layout = QVBoxLayout()

        for row in rows:
            rowLayout = QHBoxLayout()

            for buttonText in row:
                button = QPushButton(buttonText)
                button.clicked.connect(self.buttonClicked)
                button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                rowLayout.addWidget(button)

            layout.addLayout(rowLayout)

        self.setLayout(layout)

    # Link buttonCLicked to update the ETA
    # Move each char in the ETA string 1 char forward and insert the clicked char at the end
    # Also send a signal to update other information. This function is for demonstration purposes. Need to be removed in the final version.
    def buttonClicked(self):
        sender = self.sender()
        numClicked = sender.text()

        global eta_str

        if numClicked == 'CLR':
            eta_str = '00:00'
        else:
            h = eta_str[0:2]
            m = eta_str[3:5]
            eta_str = h[1] + m[0] + ':' + m[1] + numClicked

        self.buttonClickedSignal.emit(eta_str)

# Assemble each widget into one widget
class gui(QWidget):
    def __init__(self):
        super(gui, self).__init__()

        self.init_ui()

    def init_ui(self):
        self.table = readTable()
        self.compass = compass()
        self.info = infoDisplay()
        self.pad = numPad()
        self.dial = sfcDial()
        self.plot = AppDemo()

        # For receiving the nuttonClicked signal to update other information
        self.pad.buttonClickedSignal.connect(self.updateETA)

        # layout = QHBoxLayout()
        # subLayout1 = QHBoxLayout()
        # subLayout2 = QVBoxLayout()
        # subLayout3 = QVBoxLayout()
        #
        # subLayout1.addWidget(self.compass)
        # subLayout1.addWidget(self.dial)
        # subLayout2.addLayout(subLayout1)
        # subLayout2.addWidget(self.plot)
        # subLayout3.addWidget(self.info)
        # subLayout3.addWidget(self.pad)
        # layout.addLayout(subLayout2)
        # layout.addLayout(subLayout3)
        #
        # self.setLayout(layout)

        # adjust margin, setContentsMargins
        layout = QGridLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        layout.addWidget(self.compass, 0, 0, 4, 2) # assign row and column number to elements
        layout.addWidget(self.dial, 0, 2, 4, 2)
        layout.addWidget(self.plot, 4, 0, 6, 4)
        layout.addWidget(self.info, 0, 4, 3, 2)
        layout.addWidget(self.pad, 3, 4, 7, 2)

        # layout = QGridLayout()
        # layout.setContentsMargins(10, 10, 10, 10)
        # layout.setSpacing(10)
        # layout.addWidget(self.compass, 0, 0, 4, 2)
        # layout.addWidget(self.dial, 0, 2, 4, 2)
        # layout.addWidget(self.plot, 4, 0, 8, 4)
        # layout.addWidget(self.info, 0, 4, 3, 2)
        # layout.addWidget(self.pad, 3, 4, 7, 2)

        self.setLayout(layout)
        self.setWindowTitle('GUI')
        self.setGeometry(320, 220, 800, 480)

    # For updating the arrow on the interactive map. Not in use anymore
    def update(self, current_posi, next_posi, heading):
        self.table.update(current_posi, next_posi, heading)

    # For updating information on the old information panel
    # Need to be modified to adapt the new information panel 
    def updateInfo(self, lat, long, head, spd, acce, sfc):
        self.info.update(lat, long, head, spd, acce, sfc)

    # When button clicked, update everything
    def updateETA(self, eta_str):
        information = self.table.update(1, 1, 1)
        self.compass.update(information[2], information[3])
        #self.info.update(information[0], information[1], information[2], information[4], information[5], information[6])
        #self.info.eta.setText(f'ETA: {eta_str}')
        self.plot.add_data(information[6], information[7])
        self.dial.update(information[6], information[7])

if __name__ == '__main__':
    app = QApplication(sys.argv)
    #app.setStyleSheet("QWidget { background-color: black; color: white; } QLabel { border: 1px solid white; }")
    #app.setStyleSheet("QWidget { background-color: black; color: white; }")

    window = gui()
    window.show()

    sys.exit(app.exec_())
