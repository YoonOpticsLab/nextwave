from PyQt5.QtWidgets import (QMainWindow, QLabel, QSizePolicy, QApplication, QPushButton,
                             QHBoxLayout, QVBoxLayout, QGridLayout, QScrollArea,
                             QWidget, QGroupBox, QTabWidget, QTextEdit, QSpinBox, QDoubleSpinBox, QSlider,
                             QFileDialog, QCheckBox, QDialog, QFormLayout, QDialogButtonBox, QLineEdit)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QFont
from PyQt5.QtCore import Qt, QTimer, QEvent, QLineF, QPointF, pyqtSignal 
import PyQt5.QtGui as QtGui
import PyQt5.QtCore as QtCore

#from PyQt5 import QtCore, QtWidgets

#from PyQt5.QtWidgets import QApplication, QDialog, QLabel, QVBoxLayout


import pyqtgraph as pg
from pyqtgraph.parametertree import Parameter, ParameterTree

import numpy as np
import sys
import os
import json

import matplotlib.cm as cmap

from nextwave_code import NextwaveEngineComm
from nextwave_sockets import NextwaveSocketComm

from threading import Thread

import xml.etree.ElementTree as ET

NUM_ZERN_DIALOG=20 # TODO

class ZernikeDialog(QDialog):
    def createFormGroupBox(self,titl):
        formGroupBox = QGroupBox(titl)
        #layout = QFormLayout()
        layout = QGridLayout()
        self.lines = [QLineEdit() for n in np.arange(NUM_ZERN_DIALOG)]
        self.chks = [QCheckBox() for n in np.arange(NUM_ZERN_DIALOG)]
        self.chk0 = QCheckBox()
        self.chk0.stateChanged.connect(self.chk0_changed)
        self.chk0.setChecked(True)

        layout.addWidget(self.chk0, 0, 1)
        for nZernike,le in enumerate(self.lines):
            #chk = QCsjeckBox()
            #layout.addRow(QLabel("Z%2d"%(nZernike)) , le)
            layout.addWidget(QLabel("Z%2d"%(nZernike+1)), nZernike+1, 0) #, le)
            layout.addWidget(QLabel("%+0.2f"%self.ui_parent.engine.zernikes[nZernike]), nZernike+1, 1) #, le)

            if nZernike>=2:
                layout.addWidget(self.chks[nZernike], nZernike+1, 2)
                layout.addWidget(self.lines[nZernike], nZernike+1, 3)

            self.chks[nZernike].setChecked(True)

        btnR = QPushButton("\u2192") # r
        layout.addWidget(btnR,0,2)
        btnR.clicked.connect(self.use_current )

        btnReset = QPushButton("\u21ba") # reset spinning arrow
        #btnReset = QPushButton("\u1f5d1") # trash can.. doesn't work
        layout.addWidget(btnReset,0,3)
        btnReset.clicked.connect(self.reset )

        formGroupBox.setLayout(layout)
        return formGroupBox

    def use_current(self):
        for nZernike,le in enumerate(self.lines):
            le.setText("%f"%self.ui_parent.engine.zernikes[nZernike])

    def reset(self):
        for nZernike,le in enumerate(self.lines):
            le.setText("")

    def chk0_changed(self):
        print(self.ui_parent.engine.zernikes[4] )

    def mycall(self):
        zs = [str( l1.text()) for l1 in self.lines]
        zs = [0 if z1=='' else float(z1) for z1 in zs]
        self.callback(zs)

    def handleClick(self,button):
        role=self.buttonBox.buttonRole(button)
        if role==QDialogButtonBox.ApplyRole:
            self.mycall()

    def __init__(self,titl,callback,ui_parent):
        super().__init__()
        self.setWindowTitle(titl)
        self.ui_parent = ui_parent
        self.callback = callback
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Apply) # | QDialogButtonBox.Cancel)
        self.buttonBox.clicked.connect(self.handleClick) #lambda: self.mycall(callback) )
        #self.buttonBox.setWindowModality(Qt.ApplicationModal) # By default is modeless... better?

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.createFormGroupBox(titl))
        mainLayout.addWidget(self.buttonBox)
        self.setLayout(mainLayout)

        # ZOOM box
class BoxInfoDialog(QDialog):
    def __init__(self,titl,ui_parent):
        super().__init__(ui_parent)
        #self.setWindowFlag(Qt.FramelessWindowHint) 
        #self.setWindowTitle(titl)
        self.ui_parent = ui_parent

        layout=QVBoxLayout()
        self.text_num = QLabel()
        layout.addWidget(self.text_num)
        self.text_box = QLabel()
        layout.addWidget(self.text_box)
        self.text_centroid = QLabel()
        layout.addWidget(self.text_centroid)
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        self.image_label=image_label
        layout.addWidget(self.image_label)
        self.setLayout(layout)

    def set_box(self,n,box_pix, cent_x, cent_y, centroid_x_abs,centroid_y_abs, box_x,box_y):
        line1="n=%d"%n
        try:
            line1 += "(%f)"%self.ui_parent.engine.box_metrics[n]
        except:
            pass
        self.text_num.setText(line1)
        self.text_box.setText("box center=(%0.3f,%0.3f)"%(box_x,box_y))
        self.text_centroid.setText("centroid=(%0.3f,%0.3f)"%(centroid_x_abs,centroid_y_abs))
        self.box_pix=box_pix
        self.cent_x=cent_x
        self.cent_y=cent_y

    def update_ui(self):
        self.image_label.resize(200,200)
        bits=self.box_pix
        totalBytes = bits.nbytes
        width=bits.shape[1]
        height=bits.shape[0]
        bytesPerLine = int(totalBytes/height)
        qimage = QImage(bits,width,height,bytesPerLine,QImage.Format_Indexed8)

        pixmap = QPixmap(qimage)

        painter = QPainter()
        painter.begin(pixmap)

        pen = QPen(Qt.red, 1.0)
        painter.setPen(pen)
        points=[QPointF(self.cent_x,self.cent_y)]
        painter.drawPoints(points)

        if self.ui_parent.draw_predicted: 
            pen2 = QPen(Qt.green, 1.0)
            painter.setPen(pen2)
            points_centroids=[QPointF(
                self.ui_parent.engine.offline.est_x[n],
                self.ui_parent.engine.offline.est_y[n])
                              for n in np.arange(self.ui_parent.engine.num_boxes)]
            painter.drawPoints(points_centroids)

        painter.end()
        pixmap = pixmap.scaled(200,200,Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.ui_parent.box_info = -1

COLOR_INDICATOR_VAL=0.05

def actuator_color(nval):
        if 0<nval<128*COLOR_INDICATOR_VAL:
            colr=QtGui.qRgb(255-nval,0,0)
        elif nval>(256-128*COLOR_INDICATOR_VAL ): # TODO check this
            colr=QtGui.qRgb(0,nval,0)
        else:
            colr=QtGui.qRgb(nval,nval,nval) # Middle values are gray
        return colr

class ActuatorPlot(QLabel):
    def __init__(self, ui_parent, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #self.setGeometry(QtCore.QRect(0,0,spectroWidth,spectroHeight))
        self.ui_parent = ui_parent
        #self.pixmap = QPixmap(11,11)
        #self.bits=np.random.normal(size=(11,11))*32+128
        self.bits=np.zeros( (11,11), dtype='uint8')
        #self.qi = QImage(self.bits*0,self.bits.shape[0],self.bits.shape[1],QImage.Format_Indexed8)

        self.map_rows=(list(range(3,8))+list(range(2,9))+list(range(1,10))+list(range(0,11))*5+
                       list(range(1,10)) + list(range(2,9)) + list(range(3,8)) )

        self.map_cols=[0]*5+[1]*7+[2]*9+[3]*11+[4]*11+[5]*11+[6]*11+[7]*11+[8]*9+[9]*7+[10]*5

        self.act_colors = [actuator_color(nwhich) for nwhich in np.arange(256)]

    def set_colors(self,widg):
        #https://het.as.utexas.edu/HET/Software/html/qimage.html#image-transformations
        #https://stackoverflow.com/questions/35382088/qimage-custom-indexed-colors-using-setcolortable
        [widg.setColor(n, self.act_colors[n]) for n in np.arange(256)]

    ''' Didn't work: not sure why
        def button_clicked(self, event):
            print("clicked:", event.pos() )
            # Get the geometry of the spot window
            geometry = self.pixmap1.geometry()

            # Access position and size attributes
            x = geometry.x()
            y = geometry.y()
            width = geometry.width()
            height = geometry.height()
         
            x_scaled = event.pos().x() / width #* self.image_pixels.shape[1]
            y_scaled = event.pos().y() / height #* self.image_pixels.shape[0]
            print("scaled: x,y ", x_scaled, y_scaled)
'''        
        
    def paintEvent_manual(self): #, p, *args):
        #mirror_vals=np.array(np.random.normal(size=(97)) )
        #mirror_vals=np.linspace(-0.99,0.99,97)
        mirror_vals = self.ui_parent.engine.comm.mirror_voltages 
        self.bits[ self.map_cols,self.map_rows] = mirror_vals * 128 + 128
        #for y in np.arange(11):
            #for x in np.arange(11):
                #self.bits[y,x]=int(x*(256/11) )
        #self.bits = np.random.normal( size=(11,11))*32+128
        #np.savetxt("/tmp/bits.txt", self.bits)
        # calculate the total number of bytes in the frame 
        width=self.bits.shape[0]
        height=self.bits.shape[1]
        totalBytes = self.bits.nbytes
        bytesPerLine = int(totalBytes/height)

        # Needed to fix skew problem.
        #https://stackoverflow.com/questions/41596940/qimage-skews-some-images-but-not-others

        #image = QImage(bits, width, height, bytesPerLine, QImage.Format_Grayscale8)
        #self.imageLabel.setPixmap(QPixmap.fromImage(image))
        #self.set_colors(self.qi)

        self.qi = QImage(self.bits,width,height,bytesPerLine,QImage.Format_Indexed8)
        self.qi.setColorTable(self.act_colors)
        self.pixmap1 = QPixmap.fromImage(self.qi).scaled(self.height(),self.width(),Qt.KeepAspectRatio)
        self.setPixmap( self.pixmap1 )
        #self.pixmap1.mousePressEvent = self.button_clicked # Didn't work

        #qp = QPainter(self.qi)
        #qp.setBrush(br)
        #qp.setPen(QtGui.QColor(200,0,0)) 
        #qp.setBrush(QtGui.QColor(200,0,0)) 
        #qp.drawRect(10, 10, 30,30)
        #qp.end()
        #pixmap = QPixmap(self.qi)
        #pixmap = pixmap.scaled(self.height(),self.width(),Qt.KeepAspectRatio)
        #self.setPixmap(pixmap)

class MyBarWidget(pg.PlotWidget):

    sigMouseClicked = pyqtSignal(object) # add our custom signal

    def __init__(self, *args, **kwargs):
        super(MyBarWidget, self).__init__(*args, **kwargs)
        self.terms_expanded=False
        self.ylim_manual = None
        #self.setToolTip('This is a tooltip YO.')
        self.installEventFilter(self)

    def mousePressEvent(self, ev):
        super().mousePressEvent(ev)
        self.sigMouseClicked.emit(ev)
        print(ev, ev.pos() )
        print( self.getViewBox().viewRange() )
        if ev.button()==Qt.LeftButton:
            if (self.getViewBox().boundingRect().right() - self.getViewBox().mapFromScene(ev.pos()).x())<20:
                self.terms_expanded = not( self.terms_expanded )

    def eventFilter(self, obj, event):
        if event.type() == QEvent.ToolTip:
            pos_plot = self.getViewBox().mapSceneToView(event.pos())
            bar_which = round(pos_plot.x())
            zernike_val = self.app.engine.zernikes[bar_which-1]
            self.setToolTip("Z%d=%+0.3f"%(bar_which,zernike_val) )

        return super(MyBarWidget, self).eventFilter(obj, event)

    def clamp_current_ylim(self):
        if self.ylim_manual is None:
            ranges = self.viewRange()
            #ranges = self.getViewBox().viewRange() # The first one works okay, keeping this here just in case needed
            self.ylim_manual = ranges[1][1] # ymax
        else:
            self.ylim_manual = None # Toggle
        return

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class OfflineDialog(QDialog):
    def __init__(self,parent):
        super().__init__(parent)

        self.setWindowTitle("Low order aberrations across frames")

        self.sc = MplCanvas(self, width=5, height=4, dpi=100)
        #self.sc.axes.plot([0,1,2,3,4], [10,1,20,3,40])
        #self.setCentralWidget(sc)

        # Create a layout for the dialog
        layout = QVBoxLayout()
        layout.addWidget(self.sc)

        # Set the layout for the dialog
        self.setLayout(layout)


