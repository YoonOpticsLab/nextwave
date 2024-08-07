from PyQt5.QtWidgets import (QMainWindow, QLabel, QSizePolicy, QApplication, QPushButton,
                             QHBoxLayout, QVBoxLayout, QGridLayout, QScrollArea,
                             QWidget, QGroupBox, QTabWidget, QTextEdit, QSpinBox, QDoubleSpinBox, QSlider,
                             QFileDialog, QCheckBox, QDialog, QFormLayout, QDialogButtonBox, QLineEdit)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QFont
from PyQt5.QtCore import Qt, QTimer, QEvent, QLineF, QPointF, pyqtSignal 
import PyQt5.QtGui as QtGui
import PyQt5.QtCore as QtCore

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

WINDOWS=(os.name == 'nt')

# TODO: Configurable?
QIMAGE_HEIGHT=1000
QIMAGE_WIDTH=1000

MAIN_HEIGHT_WIN=1024
MAIN_WIDTH_WIN=1800

# These not used anymore:
SPOTS_HEIGHT_WIN=768
SPOTS_WIDTH_WIN=768

# This is used, but should be based on image, not hard-coded:
SPOTS_WIDTH_WIN_MINIMUM=1024

NUM_ZERN_DIALOG=14 # TODO

# TODO
CAM_EXPO_MIN = 32./1000.0 # TODO
CAM_EXPO_MAX = 100000 # TODO
CAM_GAIN_MIN = 0
CAM_GAIN_MAX = 9.83

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

class BoxInfoDialog(QDialog):
    def __init__(self,titl,ui_parent):
        super().__init__()
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
        self.text_num.setText("n=%d"%n)
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

        painter.end()

        pixmap = pixmap.scaled(200,200,Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.ui_parent.box_info = -1


def actuator_color(nval):
        if 0<nval<64:
            colr=QtGui.qRgb(255-nval,0,0)
        elif nval>(256-64): # TODO check this
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

    def paintEvent_manual(self): #, p, *args):
        #mirror_vals=np.array(np.random.normal(size=(97)) )
        #mirror_vals=np.linspace(-0.99,0.99,97)
        mirror_vals = self.ui_parent.engine.mirror_voltages 
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
        self.setPixmap(QPixmap.fromImage(self.qi).scaled(self.height(),self.width(),Qt.KeepAspectRatio) )

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

class NextWaveMainWindow(QMainWindow):
 def __init__(self):
    super().__init__(parent=None)

    # self.worker=Worker();
    #self.worker_thread=QThread()
    #self.worker.moveToThread(self.worker_thread);
    #self.worker_thread.start();

    self.updater = QTimer(self);
    self.updater.timeout.connect(self.update_ui)

    self.updater_dm = QTimer(self);
    self.updater_dm.timeout.connect(self.update_ui_dm)

    self.draw_refs = False
    self.draw_boxes = True
    self.draw_centroids = True
    self.draw_arrows = False
    self.draw_crosshair = True
    self.iterative_first=True
    self.box_info = -1
    self.box_info_dlg = BoxInfoDialog("HiINFO",self)


    f=open("./config.json")
    self.json_data = json.load(f)
    f.close()

    self.cx=self.json_data["params"]["cx"]
    self.cy=self.json_data["params"]["cy"]

    self.params = [
        {'name': 'UI', 'type': 'group', 'title':'User interface', 'children': [
            {'name': 'update_rate', 'type': 'int', 'value': 500, 'title':'Display update rate (ms)', 'limits':[50,2000]},
            {'name': 'update_rate_dm', 'type': 'int', 'value': 100, 'title':'DM Display update rate (ms)', 'limits':[50,2000]},
            {'name': 'show_boxes', 'type': 'bool', 'value': True, 'title':'Show search boxes'}
        ]},
        {'name': 'WF Camera', 'type': 'group', 'children': [
            {'name': 'Offset (%)', 'type': 'int', 'value': 0, 'limits':[0,100]},
            {'name': 'Binning Mode', 'type': 'list', 'value': 'None', 'limits':['None','2x2','4x4']} ]},
        {'name': 'Pupil Camera', 'type': 'group', 'children': [
            {'name': 'offset', 'title': 'Offset (%)', 'type': 'int', 'value': 0, 'limits':[0,100]},
            {'name': 'gain', 'title':'Gain (%)', 'type': 'int', 'value': 0, 'limits':[0,100]},
            {'name': 'exposure', 'title':'Exposure time (ms)', 'type': 'int', 'value': 0, 'limits':[1,1000]} ]}
        ]

    self.params_offline= [
        {'name': 'system', 'type': 'group', 'title':'System Params', 'children': [
            {'name': 'wavelength', 'type': 'int', 'value': 830, 'title':'Wavelength (nm)', 'limits':[50,2000]},
            {'name': 'lenslet_pitch', 'type': 'float', 'value': 324.0, 'title':'Lenslet pitch (um)'},
            {'name': 'pixel_pitch', 'type': 'float', 'value': 5.5*2, 'title':'Pixel pitch (um)'},
            {'name': 'pupil_diam', 'type': 'float', 'value': 10.0, 'title':'Pupil diameter (mm)'},
            {'name': 'focal_length', 'type': 'float', 'value': 24, 'title':'Focal length'},
        ]}
        ]

    self.params_offline_matlab = [
        {'name': 'system', 'type': 'group', 'title':'System Params', 'children': [
            {'name': 'wavelength', 'type': 'int', 'value': 830, 'title':'Wavelength (nm)', 'limits':[50,2000]},
            {'name': 'lenslet_pitch', 'type': 'float', 'value': 256*2, 'title':'Lenslet pitch (um)'},
            {'name': 'pixel_pitch', 'type': 'float', 'value': 6.4, 'title':'Pixel pitch (um)'},
            {'name': 'pupil_diam', 'type': 'float', 'value': 6.4*2, 'title':'Pupil diameter (mm)'},
            {'name': 'focal_length', 'type': 'float', 'value': 5.904, 'title':'Focal length'},
        ]}
        ]

    self.p = Parameter.create(name='params', type='group', children=self.params)
    self.params = self.p.saveState()
    self.apply_params()

    self.p_offline = Parameter.create(name='params_offline', type='group', children=self.params_offline)
    self.params_offline = self.p_offline.saveState()

    self.engine = NextwaveEngineComm(self)
    self.engine.init()
    self.sockets = NextwaveSocketComm(self)

    box_x,box_y,box_norm_x,box_norm_y=self.engine.make_searchboxes(self.cx,self.cy)

 def apply_params(self):
    self.updater.start(self.get_param("UI","update_rate"))
    self.updater_dm.start(self.get_param("UI","update_rate_dm"))

 def offline_load_image(self):
    ffilt='Binary files (*.bin);; BMP Images (*.bmp);; All files (*.*)'
    thedir = QFileDialog.getOpenFileNames(self, "Choose file in directory",
                ".", ffilt );

    if len(thedir)==0:
        return
    else:
        print( thedir )

    return

 def offline_config(self):
    ffilt='XML config files (*.xml);; JSON config files (*.json);; All files (*.*)'
    thedir = QFileDialog.getOpenFileNames(self, "Choose file in directory",
                ".", ffilt );

    if len(thedir)==0:
        return
    else:
        print( thedir )

    return

 def update_ui(self):

    # if self.engine.status ==  // TODO: see if engine is running before proceed

    image_pixels = self.engine.receive_image()
    self.engine.receive_centroids()
    self.engine.compute_zernikes()

    qimage = QImage(image_pixels, image_pixels.shape[1], image_pixels.shape[0],
                 QImage.Format_Grayscale8)
    pixmap = QPixmap(qimage)

    painter = QPainter()
    painter.begin(pixmap)

    if self.draw_arrows:
        #conicalGradient gradient;
		#gradient.setCenter(rect().center());
		#gradient.setAngle(90);
		#gradient.setColorAt(1.0, Qt::black);
		#gradient.setColorAt(0.0, palette().background().color());
        pen = QPen(Qt.green, 2)
        painter.setPen(pen)
        arrows=[QLineF(self.engine.ref_x[n],
                       self.engine.ref_y[n],
                       self.engine.centroids_x[n],
                       self.engine.centroids_y[n]) for n in np.arange(0,self.engine.num_boxes)]
        painter.drawLines(arrows)

    if self.draw_refs and self.engine.mode>1:
        pen = QPen(Qt.green, 2.0)
        painter.setPen(pen)
        points_ref=[QPointF(self.engine.ref_x[n],self.engine.ref_y[n]) for n in np.arange(self.engine.ref_x.shape[0])]
        painter.drawPoints(points_ref)

    if self.draw_boxes and self.engine.mode>1:
    #if self.get_param("UI","show_boxes"):

        dark_blue = QtGui.QColor.fromRgb(0,0,200)

        pen = QPen(dark_blue, 1.00, Qt.DotLine)
        painter.setPen(pen)
        BOX_BORDER=2
        box_size_pixel = self.engine.box_size_pixel
        boxes1=[QLineF(self.engine.box_x[n]-box_size_pixel//2+BOX_BORDER, # top
                       self.engine.box_y[n]-box_size_pixel//2+BOX_BORDER,
                       self.engine.box_x[n]+box_size_pixel//2-BOX_BORDER,
                       self.engine.box_y[n]-box_size_pixel//2+BOX_BORDER) for n in np.arange(self.engine.box_x.shape[0])]

        painter.drawLines(boxes1)
        boxes1=[QLineF(self.engine.box_x[n]-box_size_pixel//2+BOX_BORDER, # bottom
                       self.engine.box_y[n]+box_size_pixel//2-BOX_BORDER,
                       self.engine.box_x[n]+box_size_pixel//2-BOX_BORDER,
                       self.engine.box_y[n]+box_size_pixel//2-BOX_BORDER) for n in np.arange(self.engine.box_x.shape[0])]
        painter.drawLines(boxes1)
        boxes1=[QLineF(self.engine.box_x[n]-box_size_pixel//2+BOX_BORDER, # left
                       self.engine.box_y[n]-box_size_pixel//2+BOX_BORDER,
                       self.engine.box_x[n]-box_size_pixel//2+BOX_BORDER,
                       self.engine.box_y[n]+box_size_pixel//2-BOX_BORDER) for n in np.arange(self.engine.box_x.shape[0])]
        painter.drawLines(boxes1)
        boxes1=[QLineF(self.engine.box_x[n]+box_size_pixel//2-BOX_BORDER, # right
                       self.engine.box_y[n]-box_size_pixel//2+BOX_BORDER,
                       self.engine.box_x[n]+box_size_pixel//2-BOX_BORDER,
                       self.engine.box_y[n]+box_size_pixel//2-BOX_BORDER) for n in np.arange(self.engine.box_x.shape[0])]
        painter.drawLines(boxes1)

        idx_bads=np.where(np.isnan(self.engine.centroids_x))[0]
        bad_boxes=[QLineF(self.engine.box_x[n]-box_size_pixel//2-BOX_BORDER,
                       self.engine.box_y[n]-box_size_pixel//2+BOX_BORDER,
                       self.engine.box_x[n]+box_size_pixel//2-BOX_BORDER,
                       self.engine.box_y[n]+box_size_pixel//2-BOX_BORDER) for n in idx_bads]
        painter.drawLines(bad_boxes)
        bad_boxes=[QLineF(self.engine.box_x[n]-box_size_pixel//2-BOX_BORDER,
                       self.engine.box_y[n]+box_size_pixel//2+BOX_BORDER,
                       self.engine.box_x[n]+box_size_pixel//2-BOX_BORDER,
                       self.engine.box_y[n]-box_size_pixel//2-BOX_BORDER) for n in idx_bads]
        painter.drawLines(bad_boxes)

    if self.box_info>=0:
        pen = QPen(Qt.yellow, 1.00, Qt.SolidLine)
        painter.setPen(pen)
        BOX_BORDER=2
        box_size_pixel = self.engine.box_size_pixel
        # Do as a list comprehension just for consistency with above
        boxes1=[QLineF(self.engine.box_x[n]-box_size_pixel//2+BOX_BORDER, # top
                       self.engine.box_y[n]-box_size_pixel//2+BOX_BORDER,
                       self.engine.box_x[n]+box_size_pixel//2-BOX_BORDER,
                       self.engine.box_y[n]-box_size_pixel//2+BOX_BORDER) for n in [self.box_info] ]
        painter.drawLines(boxes1)
        boxes1=[QLineF(self.engine.box_x[n]-box_size_pixel//2+BOX_BORDER,
                       self.engine.box_y[n]+box_size_pixel//2-BOX_BORDER,
                       self.engine.box_x[n]+box_size_pixel//2-BOX_BORDER,
                       self.engine.box_y[n]+box_size_pixel//2-BOX_BORDER) for n in [self.box_info] ]
        painter.drawLines(boxes1)
        boxes1=[QLineF(self.engine.box_x[n]-box_size_pixel//2+BOX_BORDER,
                       self.engine.box_y[n]-box_size_pixel//2+BOX_BORDER,
                       self.engine.box_x[n]-box_size_pixel//2+BOX_BORDER,
                       self.engine.box_y[n]+box_size_pixel//2-BOX_BORDER) for n in [self.box_info] ]
        painter.drawLines(boxes1)
        boxes1=[QLineF(self.engine.box_x[n]+box_size_pixel//2-BOX_BORDER,
                       self.engine.box_y[n]-box_size_pixel//2+BOX_BORDER,
                       self.engine.box_x[n]+box_size_pixel//2-BOX_BORDER,
                       self.engine.box_y[n]+box_size_pixel//2-BOX_BORDER) for n in [self.box_info] ]
        painter.drawLines(boxes1)

    # Centroids:
    if self.draw_centroids and self.engine.mode>1:
        #for ncen,cen in enumerate(self.centroids_x):
            #if np.isnan(cen):
                #print(ncen,end=' ')

        pen = QPen(Qt.red, 1.0)
        painter.setPen(pen)
        points_centroids=[QPointF(self.engine.centroids_x[n],self.engine.centroids_y[n]) for n in np.arange(self.engine.num_boxes)]
        painter.drawPoints(points_centroids)

    if self.draw_crosshair:
        pen = QPen(Qt.red, 1.5)
        painter.setPen(pen)
        CROSSHAIR_SIZE=30
        xlines=[QLineF(self.cx+0, # right
                       self.cy-CROSSHAIR_SIZE,
                       self.cx-0, # right
                       self.cy+CROSSHAIR_SIZE),
                QLineF(self.cx+CROSSHAIR_SIZE, # right
                       self.cy+0,
                       self.cx-CROSSHAIR_SIZE, # right
                       self.cy-0)
                ]

        painter.drawLines(xlines)

        #im_buf=self.shmem_data.read(width*height)
    #bytez =np.frombuffer(im_buf, dtype='uint8', count=width*height )
    #ql1=[QLineF(100,100,150,150)]
    #painter.drawLines(ql1)
    painter.end()

    if self.box_info>=0:
        if not self.box_info_dlg.isVisible():
            self.box_info_dlg.setGeometry(self.box_info_loc[0]+40,self.box_info_loc[1],100,100)
            self.box_info_dlg.show()

        xUL=int( self.engine.box_x[self.box_info]-box_size_pixel//2 )
        yUL=int( self.engine.box_y[self.box_info]-box_size_pixel//2 )
        #box_pix=self.engine.image.copy()
        box_pix=self.engine.image[ yUL:yUL+int(box_size_pixel), xUL:xUL+int(box_size_pixel) ].copy()
        # Centroid locations in this zoom box are relative to box upper left
        self.box_info_dlg.set_box(self.box_info, box_pix,
                                  self.engine.centroids_x[self.box_info]-xUL, self.engine.centroids_y[self.box_info]-yUL,
                                  self.engine.centroids_x[self.box_info], self.engine.centroids_y[self.box_info],
                                  self.engine.box_x[self.box_info], self.engine.box_y[self.box_info]
                                  )
        self.box_info_dlg.update_ui()
    else:
        self.box_info_dlg.hide()

    #pixmap = pixmap.scaled(SPOTS_WIDTH_WIN,SPOTS_HEIGHT_WIN, Qt.KeepAspectRatio)
    self.widget_centrals.setMinimumWidth(SPOTS_WIDTH_WIN_MINIMUM)
    self.pixmap_label.setPixmap(pixmap)
    #print ('%0.2f'%bytez.mean(),end=' ', flush=True);

    s=""
    for n in np.arange(13):
        s += 'Z%2d=%+0.4f\n'%(n+1,self.engine.zernikes[n])
    self.text_status.setText(s)

    rms,rms5p,cylinder,sphere,axis=self.engine.calc_diopters()
    left_chars=15
    str_stats=f"{'RMS':<15}= {rms:3.2f}\n"
    str_stats+=f"{'HORMS':<15}= {rms5p:3.2f}\n"
    str_stats+=f"{'Sphere(+cyl)':<15}= {sphere:3.2f}\n"
    #str_stats+=f"{'Sphere(-cyl)':<15}= {sphere:3.2f}\n"
    str_stats+=f"{'Cylinder':<15}= {cylinder:3.2f}\n"
    str_stats+=f"{'Axis(-cyl)':<15}= {axis:3.2f}\n"
    self.text_stats.setText(str_stats)
    #self.text_stats.setHtml(str_stats) # TODO: To get other colors, can embed <font color="red">TEXT</font><br>, etc.

    self.line_centerx.setText(str(self.cx) )
    self.line_centery.setText(str(self.cy) )

    if self.engine.fps0!=0:
        # TODO: get state
        s=""
        if self.engine.mode==3:
            s+="Running "
        if self.chkLoop.isChecked():
            s += "(AO) "
        s+="%3.2f FPS (%04.2f ms: %04.1f+%04.2f ms)"%(1000/self.engine.fps0,self.engine.fps0, float(self.engine.fps1), float(self.engine.fps2)  )
        s+="\nFrames: %d"%self.engine.total_frames
    else:
        s="0 fps"

    self.label_status0.setText(s)
    self.label_status0.setStyleSheet("color: rgb(0, 255, 0); background-color: rgb(0,0,0);")

    # TODO: Perhaps move all this code into the bar_plot object itself??
    self.bar_plot.clear()

    if self.bar_plot.terms_expanded:
        orders_list=np.arange(2,11)
    else:
        orders_list=np.arange(2,5)

    first_term=np.sum(orders_list[0]+1) # First time is sum of orders - 1
    nterms=orders_list[0]+1 # Nterms in first order
    last_term= first_term+nterms+1

    first_term -= 1 # To match the order in the Z array (zero-based issue maybe?)

    num_orders=len(orders_list)
    MAX_BAR_ORDERS=11 # TODO: Or could be based on displayed
    order_colors=[np.array(cmap.tab10(norder))*255 for norder in np.linspace(0.0,1,MAX_BAR_ORDERS)]

    for norder,order in enumerate(orders_list):
        colrx=order_colors[norder]
        zterms1=np.arange(first_term,first_term+nterms)
        xr=zterms1
        bgx = pg.BarGraphItem(x=zterms1+1, height=self.engine.zernikes[zterms1], width=1.0, brush=colrx)
        first_term += nterms #=first_termzterms1[-1]+1
        nterms += 1
        self.bar_plot.addItem(bgx)

    #print( self.bar_plot.getViewBox().state['limits'] )
    # First_term will now be the first of the next order
    maxval=np.max( self.engine.zernikes[5:first_term-1] )
    minval=np.min( self.engine.zernikes[5:first_term-1] )

    if False: # TODO: indicate 3-5 out of range
    #for ntrunc in np.arange(3,6):
        if self.engine.zernikes[ntrunc-1]>maxval:
            itm=pg.ScatterPlotItem([ntrunc],[maxval],symbol="arrow_up",size=40)
            self.bar_plot.addItem(itm)
        elif self.engine.zernikes[ntrunc-1]<minval:
            itm=pg.ScatterPlotItem([ntrunc],[minval],symbol="arrow_down",size=40)
            self.bar_plot.addItem(itm)

    if self.bar_plot.ylim_manual is None:
        if not np.isnan(maxval):
            lim = np.max( (np.abs(minval), np.abs(maxval)) )
            self.bar_plot.setYRange(-lim, lim)
    else:
        self.bar_plot.setYRange(-self.bar_plot.ylim_manual, self.bar_plot.ylim_manual)

    #self.bar_plot.getAxis('left').setTickSpacing(1, 0.1)
    #self.bar_plot.getAxis('left').setTickDensity(2)
    #colr2=np.array(cmap.Spectral(0.8))*255
    #bg2 = pg.BarGraphItem(x=np.arange(4)+3, height=self.engine.zernikes[5:9], width=1.0, brush=colr2)

    #self.bar_plot.addItem(bg3)
    self.bar_plot.showGrid(x=False,y=True)

 def update_ui_dm(self):
    if self.chkLoop.isChecked():
        self.actuator_plot.paintEvent_manual()

    # New method is to send the command somewhere else
    if False:
        if self.chkLoop.isChecked():
            self.sockets.alpao.send(b"L")
            self.actuator_plot.paintEvent_manual()
        else:
            #self.sockets.alpao.send(b"l")
            pass

 def set_follow(self,state):
    buf = ByteStream()
    buf.append(int(state)*2)
    self.shmem_boxes.seek(0)
    self.shmem_boxes.write(buf)
    self.shmem_boxes.flush()

 def show_zernike_dialog(self,which,callback):
     dlg = ZernikeDialog(which, callback,self)
     dlg.exec()

 def get_paramX(self,name_parent,name,level=None):
    if level==None:
        level=self.params['children'] # start at top
    print( level )
    for node in level:
        if node['name']==name_parent:
            return( self.get_param("",name,node["children"]) )
        else:
            if node['name']==name:
                return(node["value"])

 def get_param_new(self,name):
     print( self.params_new)
     print( self.params_params)
     return self.params_new["children"][name]["value"]

 def get_param(self,name_parent,name,offline=False):
     if name=="pupil_diam":
         try:
             val=float( self.line_pupil_diam.text() )
             print("From UI: %s"%val)
             if val>0:
                 return val
         except:
             pass # If UI not ready, get from the parameters

     if offline:
         return self.params_offline["children"][name_parent]["children"][name]["value"]
     else:
         return self.params["children"][name_parent]["children"][name]["value"]

 def set_param(self,name_parent,name,newval,offline=False):
    if offline:
        self.params["children"][name_parent]["children"][name]["value"] = newval
    else:
        self.params["children"][name_parent]["children"][name]["value"] = newval

 def pupil_changed(self):
     #val=float( self.line_pupil_diam.text() )
     #print("From UI: %s"%val)
     self.engine.init_params() # will read from UI
     self.engine.make_searchboxes(self.cx,self.cy)

 def reload_config(self):
     filename = self.edit_xml.text()
     tree = ET.parse(filename)
     root = tree.getroot()
     all_params={}
     processed=[]
     params_params=[]

     for child in root:
        # Make groupname unique by adding ones if necessary
        groupname=child.tag
        if groupname in processed:
            groupname += "1"
        processed += [groupname]

        for attrib1 in child.attrib:
            item_name="%s_%s"%(groupname,attrib1)
            value1=child.attrib[attrib1]
            all_params[item_name]=value1

            params1={'name':item_name, 'type':'str', 'value': str(value1), 'title':item_name}
            params_params += [params1]

     self.xml_params = all_params
     self.params_params = params_params
     self.xml_p = Parameter.create(name='xml_params', type='group', children=self.params_params)
     self.params_new = self.xml_p.saveState()

     self.xml_param_tree = ParameterTree()
     self.xml_param_tree.setParameters(self.xml_p, showTop=False)
     self.layout_config.addWidget(self.xml_param_tree,1,0,-1,4)

     #print (self.xml_params['OPTICS_PupilDiameter'])
     self.box_um = self.get_param_new("LENSLETS_LensletPitch")

     self.setWindowTitle('NextWave: %s'%(self.xml_params["SessionName_name"]))

 def params_apply_clicked(self):
     self.par= self.p.saveState()
     #print(self.par)
     self.params=self.par
     self.apply_params()

     # Auto-populate from parameters when the tab is shown:
    #self.tabWidget.tabBarClicked.connect(self.userSettings)
#def userSettings(self, tabIndex):
    #if tabIndex != 5:
     #self.param_tree.setParameters(self.p, showTop=False)


 # PANELS/layouts, etc.
 def initUI(self):

     self.key_control = False 

     self.setWindowIcon(QtGui.QIcon("./resources/wave_icon.png"))
     self.setWindowTitle('NextWave')
     #self.setWindowTitle("Icon")

     self.widget_centrals = QWidget()
     self.scroll_central = QScrollArea()
     layout=QVBoxLayout()
     pixmap_label = QLabel()
     #pixmap_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
     #pixmap_label.resize(SPOTS_WIDTH_WIN,SPOTS_HEIGHT_WIN)
     pixmap_label.setAlignment(Qt.AlignCenter)
     self.pixmap_label=pixmap_label

     im_np = np.ones((QIMAGE_HEIGHT,QIMAGE_WIDTH),dtype='uint8')
     #im_np = np.transpose(im_np, (1,0,2))
     qimage = QImage(im_np, im_np.shape[1], im_np.shape[0],
                     QImage.Format_Mono)
     pixmap = QPixmap(qimage)
     #pixmap = pixmap.scaled(SPOTS_WIDTH_WIN,SPOTS_HEIGHT_WIN, Qt.KeepAspectRatio)
     pixmap_label.setPixmap(pixmap)
     pixmap_label.mousePressEvent = self.butt

     #Scroll Area Properties
     self.scroll_central.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
     self.scroll_central.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
     self.scroll_central.setWidgetResizable(True)
     self.scroll_central.setWidget(self.pixmap_label)
     layout.addWidget(self.scroll_central,15)

     self.bar_plot = MyBarWidget()
     self.bar_plot.app = self
     layout.addWidget(self.bar_plot,5)
     self.widget_centrals.setLayout(layout)

     self.widget_displays = QWidget()
     layout=QVBoxLayout(self.widget_displays)
     layout.addWidget(QGroupBox('Pupil'))
     #layout.addWidget(QGroupBox('DM'))
     self.actuator_plot = ActuatorPlot(self)
     self.actuator_plot.resize(200,200)
     #layout.addWidget(QGroupBox('Wavefront'))
     #layout.addWidget(QGroupBox('PSF'))
     #self.ap = pg.plot()
     #self.ap.addItem( self.actuator_plot )
     layout.addWidget(self.actuator_plot)

     self.widget_controls = QWidget()
     layout=QVBoxLayout()
     tabs = QTabWidget()
     tabs.setTabPosition(QTabWidget.North)
     tabs.setMovable(True)
     self.tabs = tabs

     l1 = QHBoxLayout()

     self.chkFollow = QCheckBox("Boxes follow centroids")
     self.chkFollow.stateChanged.connect(lambda:self.set_follow(self.chkFollow.isChecked()))
     l1.addWidget(self.chkFollow)

     btn = QPushButton("Search box shift")
     btn.clicked.connect(lambda: self.show_zernike_dialog("Shift search boxes", self.engine.shift_search_boxes ) )
     l1.addWidget(btn)

     btn = QPushButton("Reference shift")
     btn.clicked.connect(lambda: self.show_zernike_dialog("Shift references", self.shift_references ) )
     l1.addWidget(btn)

     self.widget_op = QWidget()
     layout_op = QVBoxLayout()
     self.ops_pupil = QGroupBox('Pupil')
     self.ops_pupil.setStyleSheet(":title {font-weight: bold}") # Doesn't work
     layout_op.addWidget(self.ops_pupil)
     self.ops_source = QGroupBox('Camera/Source')
     layout_op.addWidget(self.ops_source)
     self.ops_dm = QGroupBox('DM')
     layout_op.addWidget(self.ops_dm)
     self.widget_op.setLayout(layout_op)

     panel_names = ["Operation", "Settings", "Config", "Offline"]
     pages = [QWidget(tabs) for nam in panel_names]
     pages[0].setLayout(layout_op)
     for n, tabnames in enumerate(panel_names):
         tabs.addTab(pages[n], tabnames)

     ### Pupil ops
     layout1 = QGridLayout(self.ops_pupil)

     ### Arrows pad 
     self.m = 1
     btnL = QPushButton("\u2190") # l
     layout1.addWidget(btnL,1,4)
     btnL.clicked.connect(lambda: self.move_center(-1,0) )
     btnU = QPushButton("\u2191") # u
     layout1.addWidget(btnU,0,5)
     btnU.clicked.connect(lambda: self.move_center(0,-1) )
     btnR = QPushButton("\u2192") # r
     layout1.addWidget(btnR,1,6)
     btnR.clicked.connect(lambda: self.move_center(1,0) )
     btnD = QPushButton("\u2193") # d
     layout1.addWidget(btnD,2,5)
     btnM = QPushButton() # d
     self.chkMove = QCheckBox("")
     layout1.addWidget(self.chkMove,1,5,alignment=Qt.AlignCenter)
     btnD.clicked.connect(lambda: self.move_center(0,1) )

     self.chkMove.clicked.connect(lambda: self.set_m(self.chkMove.isChecked()) )

     lbl = QLabel("Center X:")
     layout1.addWidget(lbl,0,0)
     lbl = QLabel("Center Y:")
     layout1.addWidget(lbl,1,0)
     lbl = QLabel("Diameter (mm):")
     layout1.addWidget(lbl,2,0)
     self.line_centerx = QLineEdit()
     self.line_centerx.setMaxLength(6)
     layout1.addWidget(self.line_centerx,0,1)
     self.line_centery = QLineEdit()
     self.line_centery.setMaxLength(6)
     layout1.addWidget(self.line_centery,1,1)
     self.line_pupil_diam = QLineEdit()
     self.line_pupil_diam.setMaxLength(6)
     layout1.addWidget(self.line_pupil_diam,2,1)

     self.line_pupil_diam.textChanged.connect(self.pupil_changed)
     #btnFind = QPushButton("Find center")
     #btnFind.setStyleSheet("color : orange")
     #layout1.addWidget(btnFind,2,1)

     self.it_start = QLineEdit("3")
     layout1.addWidget(self.it_start,4,0)
     self.it_step = QLineEdit("0.5")
     layout1.addWidget(self.it_step,4,1)
     self.it_stop = QLineEdit("6.4")
     layout1.addWidget(self.it_stop,4,2)

     btn = QPushButton("Run")
     layout1.addWidget(btn,4,3)
     btn.clicked.connect(lambda: self.iterative_run() )

     btn = QPushButton("Step")
     layout1.addWidget(btn,4,4)
     btn.clicked.connect(lambda: self.iterative_step() )

     #btnIt1 = QPushButton("Step It+=0.5")
     #layout1.addWidget(btnIt1,3,1)
     #btnIt1.clicked.connect(self.run_iterative)

     self.lblIt = QLabel("3.2")
     layout1.addWidget(self.lblIt,4,5)

     ### Camera Ops
     layout1 = QGridLayout(self.ops_source)

     self.chkBackSubtract = QCheckBox("Subtract background")
     self.chkBackSubtract.stateChanged.connect(self.sub_background)
     layout1.addWidget(self.chkBackSubtract,0,0)
     btnBackSet = QPushButton("Set background")
     layout1.addWidget(btnBackSet,0,1)
     btnBackSet.clicked.connect(self.set_background) 

     self.chkReplaceSubtract = QCheckBox("Replace subtracted")
     self.chkReplaceSubtract.stateChanged.connect(self.replace_background)
     #layout1.addWidget(self.chkReplaceSubtract,0,2)

     self.slider_threshold = QSlider(orientation=Qt.Horizontal)
     self.slider_threshold.setMinimum(0) # TODO: Get from camera
     self.slider_threshold.setMaximum(100) # TODO: Get from camera
     layout1.addWidget(self.slider_threshold,1,1)
     self.slider_threshold.valueChanged.connect(self.slider_threshold_changed) # TODO

     self.chkApplyThreshold = QCheckBox("Apply Thresholding")
     self.chkApplyThreshold.stateChanged.connect(self.click_apply_threshold)
     layout1.addWidget(self.chkApplyThreshold,1,0)

     self.threshold_val = QDoubleSpinBox()
     layout1.addWidget(self.threshold_val,1,2)
     self.threshold_val.setDecimals(2)

     lbl = QLabel("Exposure time (ms)")
     layout1.addWidget(lbl,2,0)

     self.slider_exposure = QSlider(orientation=Qt.Horizontal)
     self.slider_exposure.setMinimum(0) # TODO: Get from camera
     self.slider_exposure.setMaximum(100) # TODO: Get from camera
     layout1.addWidget(self.slider_exposure,2,1)
     self.slider_exposure.valueChanged.connect(self.slider_exposure_changed)

     self.exposure = QDoubleSpinBox()
     layout1.addWidget(self.exposure,2,2)
     self.exposure.setDecimals(4)
     self.exposure.setMinimum(CAM_EXPO_MIN)
     self.exposure.setMaximum(CAM_EXPO_MAX)

     lbl = QLabel("Gain (dB)")
     layout1.addWidget(lbl,3,0)

     self.slider_gain = QSlider(orientation=Qt.Horizontal)
     self.slider_gain.setMinimum(0) # TODO: Get from camera
     self.slider_gain.setMaximum(100) # TODO: Get from camera
     layout1.addWidget(self.slider_gain,3,1)
     self.slider_gain.valueChanged.connect(self.slider_gain_changed)

     self.gain = QDoubleSpinBox()
     layout1.addWidget(self.gain,3,2)
     self.gain.setMinimum(CAM_GAIN_MIN)
     self.gain.setMaximum(CAM_GAIN_MAX)

     ### DM Ops
     layout1 = QGridLayout(self.ops_dm)
     self.chkLoop = QCheckBox("Close AO Loop")
     layout1.addWidget(self.chkLoop,0,0)

     btn = QPushButton("Save flat")
     btn.clicked.connect(self.flat_save)
     layout1.addWidget(btn, 0,3 )

     btn = QPushButton("Do flat")
     btn.clicked.connect(self.flat_do)
     layout1.addWidget(btn, 0,1 )

     btn = QPushButton("Do zero")
     btn.clicked.connect(self.zero_do)
     layout1.addWidget(btn, 0,2 )

     btn = QPushButton("Search box shift")
     btn.clicked.connect(lambda: self.show_zernike_dialog("Shift search boxes", self.engine.shift_search_boxes ) )
     layout1.addWidget(btn, 1,0 )

     btn = QPushButton("Reference shift")
     btn.clicked.connect(lambda: self.show_zernike_dialog("Shift references", self.engine.reset_references ) )
     layout1.addWidget(btn, 1,1 )

     btn = QPushButton("Search box RESET")
     btn.clicked.connect(self.engine.reset_search_boxes )
     layout1.addWidget(btn, 2,0 )

     btn = QPushButton("Reference RESET")
     btn.clicked.connect(self.engine.reset_references )
     layout1.addWidget(btn, 2,1 )

     self.widget_mode_buttons = QWidget()
     layoutStatusButtons = QHBoxLayout(self.widget_mode_buttons)

     self.mode_btn1 = QPushButton("Init")
     layoutStatusButtons.addWidget(self.mode_btn1)
     self.mode_btn1.clicked.connect(self.mode_init)

     self.mode_btn2 = QPushButton("Snap")
     layoutStatusButtons.addWidget(self.mode_btn2)
     self.mode_btn2.clicked.connect(self.mode_snap)

     self.mode_btn3 = QPushButton("Run")
     layoutStatusButtons.addWidget(self.mode_btn3)
     self.mode_btn3.clicked.connect(self.mode_run)

     self.mode_btn4 = QPushButton("Stop")
     layoutStatusButtons.addWidget(self.mode_btn4)
     self.mode_btn4.clicked.connect(self.mode_stop)

     self.edit_num_runs = QLineEdit("-1")
     self.edit_num_runs.setMaxLength(6)
     layoutStatusButtons.addWidget(self.edit_num_runs)

     self.mode_btn2.setEnabled( True )
     self.mode_btn3.setEnabled( True )
     #self.mode_btn4.setEnabled( False )

     # Config
     layout1 = QGridLayout(pages[2])
     lbl = QLabel("XML Config: ")
     layout1.addWidget(lbl, 0,0)
     self.edit_xml = QLineEdit(self.json_data["params"]["xml_file"])
     layout1.addWidget(self.edit_xml, 0,1)
     btn = QPushButton("...")
     layout1.addWidget(btn, 0,2)
     btn = QPushButton("Edit+Reload")
     layout1.addWidget(btn, 0,3)
     btn.clicked.connect(self.reload_config)
     #os.system('c:/tmp/sample.txt') # <- on windows this will launch the defalut editor
     self.layout_config = layout1

     # Settings
     layout1 = QGridLayout(pages[1])
     self.param_tree = ParameterTree()
     self.param_tree.setParameters(self.p, showTop=False)
     layout1.addWidget(self.param_tree,0,0)
     btn = QPushButton("Apply")
     btn.clicked.connect(self.params_apply_clicked)
     layout1.addWidget(btn,1,0)

     #self.widget_status_buttons.setLayout(layoutStatusButtons)
     layout.addWidget(self.widget_mode_buttons,1)

     self.label_status0 = QLabel("Status: ")
     layout.addWidget(self.label_status0, 1)

     self.text_status = QTextEdit()
     self.text_status.setReadOnly(True)

     #layout.addWidget(self.text_status, 1)
     layout.addWidget(tabs, 20)

     #self.widget_controls = QGroupBox('Controls')
     self.widget_controls.setLayout(layout)

     font=QFont("Courier",18,QFont.Bold);
     #font.setStyleHint(QFont::TypeWriter);

     #layout.addWidget(QGroupBox('Statistics'), 20)
     self.text_stats = QTextEdit()
     self.text_stats.setCurrentFont(font)
     self.text_stats.setReadOnly(True)
     layout.addWidget(self.text_stats)

     # OFFLINE
     layout1 = QGridLayout(pages[3])
     #btn1 = QPushButton("Load spot image")
     #btn1.setStyleSheet("color : orange")
     #layout1.addWidget(btn1,3,0,0,-1)
     #btn1.clicked.connect(self.offline_load_image)

     lbl = QLabel("Spot image: ")
     layout1.addWidget(lbl, 3,0)
     self.offline_image = QLineEdit("spots.bin")
     layout1.addWidget(self.offline_image, 3,1)
     btn = QPushButton("...")
     btn.clicked.connect(self.offline_load_image)
     layout1.addWidget(btn, 3,2)
     #btn = QPushButton("Edit+Reload")
     #layout1.addWidget(btn, 3,3)

     lbl = QLabel("XML Config: ")
     layout1.addWidget(lbl, 0,0)
     self.offline_edit_xml = QLineEdit("c:\\file\\ao\\offline_config.xml")
     layout1.addWidget(self.offline_edit_xml, 0,1)
     btn = QPushButton("...")
     btn.clicked.connect(self.offline_config)
     layout1.addWidget(btn, 0,2)
     btn = QPushButton("Edit+Reload")
     layout1.addWidget(btn, 0,3)

     #os.system('c:/tmp/sample.txt') # <- on windows this will launch the defalut editor
     self.param_tree_offline = ParameterTree()
     self.param_tree_offline.setParameters(self.p_offline, showTop=False)
     layout1.addWidget(self.param_tree_offline,1,0,2,-1)

     # Main Widget
     self.widget_main = QWidget()
     layoutCentral = QHBoxLayout()
     layoutCentral.addWidget(self.widget_centrals, stretch=3)
     layoutCentral.addWidget(self.widget_displays, stretch=2)
     layoutCentral.addWidget(self.widget_controls, stretch=1)
     self.widget_main.setLayout(layoutCentral)

     self.setCentralWidget(self.widget_main)

     # Initial Setup, startup behavior
     self.reload_config() # Load last selected XML
     #self.tabs.setCurrentIndex(2) # Doesn't allocate enough space: first tab is better
     self.engine.mode_init()
     self.sockets.init()

     menu=self.menuBar().addMenu('&File')
     menu.addAction('&Export Centroids & Zernikes', self.export)
     menu.addAction('e&Xit', self.close)

     pixmap_label.setFocus()

     self.setGeometry(2,2,MAIN_WIDTH_WIN,MAIN_HEIGHT_WIN)
     self.show()

 def slider_threshold_changed(self):
     scaled = self.slider_threshold.value()/100.0
     self.threshold_val.setValue(scaled)

 def slider_exposure_changed(self):
     scaled = 10**( float( self.slider_exposure.value())/100.0*np.log10(CAM_EXPO_MAX/CAM_EXPO_MIN)+np.log10(CAM_EXPO_MIN))
     self.exposure.setValue(scaled)
     self.sockets.camera.send(b"E=%f\x00"%(scaled*1000) ) # Convert to usec

 def slider_gain_changed(self):
     scaled = self.slider_gain.value()/100.0*CAM_GAIN_MAX
     self.gain.setValue(scaled)

 def flat_save(self):
     self.engine.flat_save()
     return
 def flat_do(self):
     self.engine.flat_do()
     return
 def zero_do(self):
     self.engine.zero_do()
     return

 def iterative_step(self):
     try:
         self.engine.iterative_size
     except:
         self.engine.iterative_size = float(self.it_start.text())

     self.engine.iterative_step(self.cx, self.cy,
                                float(self.it_step.text()), float(self.it_start.text()), float(self.it_stop.text()) )

     self.lblIt.setText('%2.2f'%self.engine.iterative_size)

 def iterative_run(self):
    self.engine.iterative_size = float(self.it_start.text())

    while (self.engine.iterative_size != float(self.it_stop.text())):
        self.iterative_step()
        self.update_ui()
        self.repaint()

 def autoshift_search_boxes(self):
     self.engine.autoshift_search_boxes()

 def set_m(self, doit):
     if doit:
         self.m = round( self.engine.box_size_pixel )
     else:
         self.m=1

 def sub_background(self):
    # TODO: Don't allow subtract if it hasn't been set once
    if (self.chkBackSubtract.isChecked()):
        self.sockets.centroiding.send(b"B\x00")
    else:
        self.sockets.centroiding.send(b"b\x00")
 def set_background(self):
    self.sockets.centroiding.send(b"S\x00")
 def replace_background(self):
    if (self.chkBackSubtract.isChecked()):
        self.sockets.centroiding.send(b"R\x00")
    else:
        self.sockets.centroiding.send(b"r\x00")

 def click_apply_threshold(self):
    if (self.chkApplyThreshold.isChecked()):
        self.sockets.centroiding.send(b"T%f\x00"%(self.slider_threshold.value()/100.0) )
    else:
        self.sockets.centroiding.send(b"t\x00")

 def move_center(self, dx, dy, m=1, do_update=True):
    m = self.m
    self.cx += (dx * m)
    self.cy += (dy * m)
    if do_update:
        self.engine.move_searchboxes(dx*m, dy*m)

 def butt(self, event):
    print("clicked:", event.pos() )
    x_scaled =event.pos().x() #/ SPOTS_WIDTH_WIN*992 # TODO: use image size
    y_scaled =event.pos().y() #/ SPOTS_WIDTH_WIN*992 # TODO: use image size
    print("scaled: x,y ", x_scaled, y_scaled)

    which_box = np.where(np.all(
        ( (self.engine.box_x-self.engine.box_size_pixel/2)<x_scaled,
          (self.engine.box_x+self.engine.box_size_pixel/2)>x_scaled,
          (self.engine.box_y-self.engine.box_size_pixel/2)<y_scaled,
          (self.engine.box_y+self.engine.box_size_pixel/2)>y_scaled,
         ), axis=0
    ))[0]

    if which_box.size==1:
        selected=which_box[0]
        if self.box_info==selected: # Already selected one. Toggle
            self.box_info=-1
        else:
            self.box_info=which_box[0]
            self.box_info_loc = (event.pos().x(), event.pos().y() )
    elif which_box.size>1:
        print( "Too many matches" )
    else :
        print( "No matches")
        self.box_info = -1

    return 

 def keyReleaseEvent(self, event):
    if event.key()==Qt.Key_Control:
        self.key_control = False 

 def keyPressEvent(self, event):
    update_search_boxes=False
    if event.key()==ord('A'):
        self.draw_arrows = not( self.draw_arrows )
    elif event.key()==ord('R'):
        self.draw_refs = not( self.draw_refs )
    elif event.key()==ord('B'):
        self.draw_boxes = not( self.draw_boxes )
        self.set_param("UI","show_boxes", not( self.get_param("UI","show_boxes") ) )
    elif event.key()==ord('C'):
        self.draw_centroids = not( self.draw_centroids )
    elif event.key()==ord('X'):
        self.draw_crosshair = not( self.draw_crosshair )
    elif event.key()==ord('H'):
        self.bar_plot.clamp_current_ylim()
    elif event.key()==ord('Q'):
        self.close()
    elif event.key()==Qt.Key_Control:
        self.key_control = True
    elif event.key()==Qt.Key_Left:
        self.move_center(-1,0,self.key_control)
    elif event.key()==Qt.Key_Right:
        self.move_center( 1,0,self.key_control)
    elif event.key()==Qt.Key_Up:
        self.cy -= 1 + 10 * self.key_control
        update_search_boxes=True
    elif event.key()==Qt.Key_Down:
        self.cy += 1 + 10 * self.key_control
        update_search_boxes=True
    else:
        print( "Uknown Key:", event.key() )

    if update_search_boxes:
        self.engine.make_searchboxes(self.cx,self.cy)

        #if event.key() == QtCore.Qt.Key_Q:
        #elif event.key() == QtCore.Qt.Key_Enter:

 def mode_init(self):
    #self.engine.mode_init()
    self.sockets.init()
 def mode_snap(self):
    self.engine.mode_snap()
 def mode_run(self):
    self.engine.mode_run()
 def mode_stop(self):
    self.engine.mode_stop()
    #self.sockets.camera.send(b"E=3.14")
    #self.engine.update_searchboxes()

 def export(self):
    default_filename="centroids.dat"
    filename, _ = QFileDialog.getSaveFileName(
        self, "Save audio file", default_filename, "Audio Files (*.mp3)"
    )
    if filename:
        fil = open(filename,'wt')

        fil.writelines( '[image size = %dx%d]\n'%(self.engine.width,self.engine.height))
        fil.writelines( '[pupil = %f,%d,%d]\n'%(self.engine.pupil_diam,self.cx,self.cy))
        for nbox in np.arange( len(self.engine.ref_x)):
            # TODO: Valid or invalid
            fil.writelines('%d\t%.12f\t%.12f\t%.12f\t%.12f\t\n'%(1,self.engine.ref_x[nbox], self.engine.ref_y[nbox],
                                                     self.engine.centroids_x[nbox], self.engine.centroids_y[nbox] ) )
        fil.close()

    filename="zernikes.txt"
    fil = open(filename,'wt')
    for val in self.engine.zernikes:
        fil.writelines('%f\n'%val)
    fil.close()

    np.save('dx',self.engine.spot_displace_x)
    np.save('dy',self.engine.spot_displace_y)
    np.save('slope',self.engine.slope)
    np.save('zterms',self.engine.zterms)
    np.save('zterms_inv',self.engine.zterms_inv)

    np.savetxt('mirror_voltages.txt',self.engine.mirror_voltages)

 def close(self):
    self.engine.send_quit() # Send stop command to engine
    self.app.exit()

 def closeEvent(self, event):
    self.close()

# rpyc servic definition
# Doesn't let you access member variables, so seems kind of pointless
import rpyc
class MyService(rpyc.Service):
    def exposed_get_nextwave(self):
        return self.win
def start_backdoor(win):
    # start the rpyc server
    from rpyc.utils.server import ThreadedServer
    from threading import Thread
    server = ThreadedServer(MyService, port = 12345)
    MyService.win = win
    t = Thread(target = server.start)
    t.daemon = True
    t.start()

def main():
  app = QApplication(sys.argv)
  win = NextWaveMainWindow()
  win.app = app
  win.initUI()

  start_backdoor(win)

  sys.exit(app.exec_())


if __name__=="__main__":
  main()
