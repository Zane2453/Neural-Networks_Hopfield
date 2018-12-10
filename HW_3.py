import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.ticker import MultipleLocator 
from PyQt5 import QtCore, QtWidgets,QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel,QMainWindow, QTableWidget, QTableWidgetItem, QWidget, QSpinBox, QGroupBox, QSlider
from PyQt5.QtWidgets import QFormLayout, QDockWidget, QComboBox, QHBoxLayout, QPushButton, QTextEdit, QAction, QApplication, QDesktopWidget, QRadioButton
from PyQt5.QtGui import QIcon
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
import sys
import numpy as np
import random
import time

class My_Main_window(QtWidgets.QDialog):
    def __init__(self,parent=None):
        
        super(My_Main_window,self).__init__(parent)

        # set the User Interface
        self.setWindowTitle('NN')
        self.resize(750, 400)

        # set the figure (left&right)
        self.figure_1 = Figure(figsize=(4, 4), dpi=100)
        self.canvas_1 = FigureCanvas(self.figure_1)

        # draw the initial axes of left graph
        self.ax_1 = self.figure_1.add_axes([0.1,0.1,0.8,0.8])
        self.ax_1.set_xlim([0,5])
        self.ax_1.set_ylim([0,5])
        self.ax_1.plot()

        # set the group box
        self.file = QGroupBox("Choose Train and Test File")
        self.association = QGroupBox("Choose Association Way")
        self.show_data = QGroupBox("Choose Show Data")
        self.show_result = QGroupBox("Result of Association")

        # set the push button
        self.button_test = QPushButton("Test")
        self.button_train = QPushButton("Train")
        self.button_show = QPushButton("Show the Graph")
        self.button_original = QPushButton("Original Graph")
        self.button_association = QPushButton("Association Graph")

        # set the combo box
        self.combo_train = QComboBox()
        self.combo_test = QComboBox()
        self.combo_train.addItems(["Basic_Training.txt", "Bonus_Training.txt"])
        self.combo_test.addItems(["Basic_Testing.txt", "Bonus_Testing.txt"])
        
        # set the label
        self.label_train_file=QLabel()
        self.label_test_file=QLabel()
        self.label_train_data=QLabel()
        self.label_test_data=QLabel()
        self.label_correct_rate=QLabel()
        self.label_correct_text=QLabel()
        self.label_iteration_time=QLabel()
        self.label_iteration_text=QLabel()
        self.label_train_file.setText("Train File")
        self.label_test_file.setText("Test File")
        self.label_train_data.setText("Train Data No.")
        self.label_test_data.setText("Test Data No.")
        self.label_correct_rate.setText("Correct Rate")
        self.label_correct_text.setText(" -- ")
        self.label_iteration_time.setText("Iteration Time")
        self.label_iteration_text.setText(" -- ")
        self.label_correct_text.setAlignment(Qt.AlignCenter)
        self.label_iteration_text.setAlignment(Qt.AlignCenter)

        # set the button trigger
        self.button_show.clicked.connect(self.ShowResult)
        self.button_train.clicked.connect(self.TrainData)
        self.button_test.clicked.connect(self.TestData)
        self.button_original.clicked.connect(self.ShowOriginal)
        self.button_association.clicked.connect(self.ShowAssociation)

        # set the combobox trigger
        self.combo_train.activated.connect(self.SetTrainFile)
        self.combo_test.activated.connect(self.SetTestFile)
        
        # set the radio button
        self.radio_syn=QRadioButton("Synchronization")
        self.radio_asyn=QRadioButton("Asynchronization")
        self.radio_train=QRadioButton("Train Data")
        self.radio_test=QRadioButton("Test Data")
        self.radio_syn.setChecked(True)
        self.radio_asyn.setChecked(False)
        self.radio_train.setChecked(True)
        self.radio_test.setChecked(False)

        # set the spin box
        self.spin_train=QSpinBox()
        self.spin_test=QSpinBox()

        # set the layout
        layout = QtWidgets.QHBoxLayout()
        layout_left = QtWidgets.QVBoxLayout()
        layout_right = QtWidgets.QVBoxLayout()
        layout_right_down = QtWidgets.QHBoxLayout()
        file_layout = QFormLayout()
        association_layout = QFormLayout()
        show_layout = QFormLayout()
        result_layout = QFormLayout()

        file_layout.addRow(self.label_train_file, self.combo_train)
        file_layout.addRow(self.label_test_file, self.combo_test)

        association_layout.addRow(self.radio_syn, self.radio_asyn)
        association_layout.addRow(self.button_train, self.button_test)

        show_layout.addRow(self.radio_train, self.radio_test)
        show_layout.addRow(self.label_train_data, self.spin_train)
        show_layout.addRow(self.label_test_data, self.spin_test)

        result_layout.addRow(self.label_correct_rate, self.label_correct_text)
        result_layout.addRow(self.label_iteration_time, self.label_iteration_text)

        self.file.setLayout(file_layout)
        self.association.setLayout(association_layout)
        self.show_data.setLayout(show_layout)
        self.show_result.setLayout(result_layout)
        
        layout_left.addWidget(self.file)
        layout_left.addWidget(self.association)
        layout_left.addWidget(self.show_data)
        layout_left.addWidget(self.show_result)
        layout_left.addWidget(self.button_show)

        layout_right_down.addWidget(self.button_original)
        layout_right_down.addWidget(self.button_association)

        layout_right.addWidget(self.canvas_1)
        layout_right.addLayout(layout_right_down)

        layout.addLayout(layout_left,10)
        layout.addLayout(layout_right,42)
 
        self.setLayout(layout)

    # process the train input and calculate the weight & theta matrix
    def SetTrainFile(self):
        self.train_input=[]
        f=open(self.combo_train.currentText())
        self.train_count=int(1) 
        self.row=int(0)

        self.train_input.append([])
        while 1:
            line=f.readline()
            if line=="":
                break
            
            line=line[:len(line)].strip("\n")
            if len(line) == 0 :
                self.train_input.append([])
                if self.train_count == 1:
                    self.col=int(len(self.train_input[self.train_count-1])/self.row)
                self.train_count=self.train_count+1
            else :
                if self.train_count == 1:
                    self.row+=1
                for i in range(len(line)):
                    if line[i] == '1':
                        self.train_input[self.train_count-1].append(int(1))
                    else:
                        self.train_input[self.train_count-1].append(int(-1))

        self.dim = int(len(self.train_input[0]))
        self.weight = []
        self.theta = []

        for i in range(self.dim):
            self.weight.append([])
            for j in range(self.dim):
                self.weight[i].append(int(0))
                for k in range(self.train_count):
                    self.weight[i][j]=self.weight[i][j]+(self.train_input[k][i] * self.train_input[k][j])
                ## weight[i][j]=float(weight[i][j]/dim)

        for i in range(self.dim):
            self.weight[i][i]=self.weight[i][i]-self.train_count

        self.theta = np.zeros((self.dim,), dtype=np.int)
        ## self.theta = np.sum(self.weight, 1)

        self.spin_train.setRange(1,self.train_count)

    # process the test input
    def SetTestFile(self):
        self.test_input = []
        f=open(self.combo_test.currentText())
        self.test_count=int(1)

        self.test_input.append([])
        while 1:
            line=f.readline()
            if line=="":
                break
            
            line=line[:len(line)].strip("\n")
            if len(line) == 0 :
                self.test_input.append([])
                self.test_count+=1
            else :
                for i in range(len(line)):
                    if line[i] == '1':
                        self.test_input[self.test_count-1].append(int(1))
                    else:
                        self.test_input[self.test_count-1].append(int(-1))
        
        self.spin_test.setRange(1,self.test_count)

    # calculate the association of train data
    def TrainData(self):
        self.train_output = []
        self.train_time = []
        self.train_rate = []

        # Synchronization
        if self.radio_syn.isChecked() :
            for z in range(self.train_count):
                temp = np.copy(self.train_input[z])
                run_time = int(0)
                similar=int(0)
                correct_rate=float(0)
                while (similar != self.dim) :
                    similar=int(0)
                    sigmoid_data = []
                    run_time+=1
                    
                    for i in range(self.dim):
                        sigmoid_data.append(np.dot(self.weight[i], temp) - self.theta[i])
                    
                    for i in range(self.dim):
                        if sigmoid_data[i] < int(0):
                            if temp[i] == int(-1):
                                similar+=1
                            else:
                                temp[i]=int(-1)
                        elif sigmoid_data[i] > int(0):
                            if temp[i] == int(1):
                                similar+=1
                            else:
                                temp[i]=int(1)
                        else:
                            similar+=1
                    
                for i in range(self.dim):
                    if temp[i] == self.train_input[z][i]:
                        correct_rate+=1

                self.train_output.append(temp)
                self.train_time.append(run_time)
                self.train_rate.append(correct_rate/self.dim)

        # Asynchronization
        else:
            for z in range(self.train_count):
                temp = np.copy(self.train_input[z])
                run_time = int(0)
                similar=True
                correct_rate=float(0)
                while similar:
                    similar=False
                    sigmoid_data = np.copy(temp)
                    run_time+=1

                    for i in range(self.dim):
                        res = np.dot(self.weight[i], sigmoid_data) - self.theta[i]
                        if res < int(0):
                            sigmoid_data[i]=int(-1)
                        elif res > int(0):
                            sigmoid_data[i]=int(1)

                    for i in range(self.dim):
                        if sigmoid_data[i] != temp[i]:
                            similar = True
                    
                    temp=np.copy(sigmoid_data)

                for i in range(self.dim):
                    if temp[i] == self.train_input[z][i]:
                        correct_rate+=1

                self.train_output.append(temp)
                self.train_time.append(run_time)
                self.train_rate.append(correct_rate/self.dim)
    
    # calculate the association of test data
    def TestData(self):
        self.test_output = []
        self.test_time = []
        self.test_rate = []

        # Synchronization
        if self.radio_syn.isChecked() :
            for z in range(self.test_count):
                temp = np.copy(self.test_input[z])
                run_time = int(0)
                similar=int(0)
                correct_rate=float(0)
                while (similar != self.dim) :
                    similar=int(0)
                    sigmoid_data = []
                    run_time+=1
                    
                    for i in range(self.dim):
                        sigmoid_data.append(np.dot(self.weight[i], temp) - self.theta[i])
                    
                    for i in range(self.dim):
                        if sigmoid_data[i] < int(0):
                            if temp[i] == int(-1):
                                similar+=1
                            else:
                                temp[i]=int(-1)
                        elif sigmoid_data[i] > int(0):
                            if temp[i] == int(1):
                                similar+=1
                            else:
                                temp[i]=int(1)
                        else:
                            similar+=1

                for i in range(self.dim):
                    if temp[i] == self.train_input[z][i]:
                        correct_rate+=1
                
                self.test_output.append(temp)
                self.test_time.append(run_time)
                self.test_rate.append(correct_rate/self.dim)

        # Asynchronization
        else:
            for z in range(self.test_count):
                temp = np.copy(self.test_input[z])
                run_time = int(0)
                similar=True
                correct_rate=float(0)
                while similar:
                    similar=False
                    sigmoid_data = np.copy(temp)
                    run_time+=1

                    for i in range(self.dim):
                        res = np.dot(self.weight[i], sigmoid_data) - self.theta[i]
                        if res < int(0):
                            sigmoid_data[i]=int(-1)
                        elif res > int(0):
                            sigmoid_data[i]=int(1)

                    for i in range(self.dim):
                        if sigmoid_data[i] != temp[i]:
                            similar = True
                    
                    temp=np.copy(sigmoid_data)

                for i in range(self.dim):
                    if temp[i] == self.train_input[z][i]:
                        correct_rate+=1

                self.test_output.append(temp)
                self.test_time.append(run_time)
                self.test_rate.append(correct_rate/self.dim)

    # show the original graph
    def ShowOriginal(self):
        spacing = 1
        minorLocator = MultipleLocator(spacing)
        self.ax_1.cla()
        self.ax_1.set_xlim([0,self.col])
        self.ax_1.set_ylim([0,self.row])
        self.ax_1.set_title("Original Graph")
        self.ax_1.plot()
        self.ax_1.xaxis.set_minor_locator(minorLocator) 
        self.ax_1.yaxis.set_minor_locator(minorLocator) 
        self.ax_1.grid(which='minor')
        for i in range(self.row):
            for j in range(self.col):
                if self.graph_original[i*self.col + j] == int(1):
                    self.ax_1.add_patch(
                        patches.Rectangle(
                            (j, self.row-i-1),   # (x,y)
                            1,          # width
                            1,          # height
                        )
                    )
        self.canvas_1.draw()

    # show the association graph
    def ShowAssociation(self):
        spacing = 1
        minorLocator = MultipleLocator(spacing)
        self.ax_1.cla()
        self.ax_1.set_xlim([0,self.col])
        self.ax_1.set_ylim([0,self.row])
        self.ax_1.set_title("Association Graph")
        self.ax_1.plot()
        self.ax_1.xaxis.set_minor_locator(minorLocator) 
        self.ax_1.yaxis.set_minor_locator(minorLocator) 
        self.ax_1.grid(which='minor')
        for i in range(self.row):
            for j in range(self.col):
                if self.graph_result[i*self.col + j] == int(1):
                    self.ax_1.add_patch(
                        patches.Rectangle(
                            (j, self.row-i-1),   # (x,y)
                            1,          # width
                            1,          # height
                        )
                    )
        self.canvas_1.draw()

    # process the original & association graph    
    def ShowResult(self):
        index = int(0)
        self.graph_original = []
        self.graph_result = []

        if self.radio_train.isChecked():
            index = self.spin_train.value()-1
            self.label_correct_text.setText(str(self.train_rate[index]))
            self.label_iteration_text.setText(str(self.train_time[index]))
            self.graph_original = np.copy(self.train_input[index])
            self.graph_result = np.copy(self.train_output[index])
        else:
            index = self.spin_test.value()-1
            self.label_correct_text.setText(str(self.test_rate[index]))
            self.label_iteration_text.setText(str(self.test_time[index]))
            self.graph_original = np.copy(self.test_input[index])
            self.graph_result = np.copy(self.test_output[index])

        spacing = 1
        minorLocator = MultipleLocator(spacing)
        self.ax_1.cla()
        self.ax_1.set_title("Original Graph")
        self.ax_1.set_xlim([0,self.col])
        self.ax_1.set_ylim([0,self.row])
        self.ax_1.plot()
        self.ax_1.xaxis.set_minor_locator(minorLocator) 
        self.ax_1.yaxis.set_minor_locator(minorLocator) 
        self.ax_1.grid(which='minor')
        for i in range(self.row):
            for j in range(self.col):
                if self.graph_original[i*self.col + j] == int(1):
                    self.ax_1.add_patch(
                        patches.Rectangle(
                            (j, self.row-i-1),   # (x,y)
                            1,          # width
                            1,          # height
                        )
                    )
        self.canvas_1.draw()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = My_Main_window()
    main_window.show()
    app.exec()
