from nextwave_log import log
from PyQt5.QtWidgets import (QMainWindow, QLabel, QSizePolicy, QApplication, QPushButton,
                             QHBoxLayout, QVBoxLayout, QGridLayout, QScrollArea,
                             QWidget, QGroupBox, QTabWidget, QTextEdit, QSpinBox, QDoubleSpinBox, QSlider,
                             QFileDialog, QCheckBox, QDialog, QFormLayout, QDialogButtonBox, QLineEdit,
                             QToolTip)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QFont, QColor, QBrush, QPolygonF, QPalette
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
from collections import OrderedDict

import matplotlib.cm as cmap

from nextwave_code import NextwaveEngineComm
from nextwave_sockets import NextwaveSocketComm

from threading import Thread

import xml.etree.ElementTree as ET

NUM_ZERN_DIALOG=20 # TODO

class FrameGalleryModel(QtCore.QAbstractListModel):
    """ The frames of an offline movie, as a gallery of small thumbnails, each captioned with its frame number (1..N, as
        shown on screen). A movie can have thousands of frames, so this is a virtual list: the view only asks for the ones it
        is showing, and a thumbnail is made when it is first shown (the most recent ones are kept).
        Thumbnails keep the brightest pixel of each block of the frame, not an average or a sample: the frames are mostly black
        with small bright spots, which a plain shrink would lose. """
    THUMB_SIZE = 100 # Longest side of a thumbnail, pixels
    MAX_CACHED = 1500

    def __init__(self, parent=None):
        super().__init__(parent)
        self.movie = None
        self.thumbs = OrderedDict() # row -> QPixmap, least recently used first
        self.shrink = 1 # Each thumbnail pixel is the brightest of shrink x shrink frame pixels
        self.thumb_size = (self.THUMB_SIZE, self.THUMB_SIZE) # (width, height) of a thumbnail

    def set_movie(self, movie):
        """ movie: (frames, height, width) uint8 array. Not copied, so don't change it under us. """
        self.beginResetModel()
        self.movie = movie
        self.thumbs.clear()
        height, width = movie.shape[1], movie.shape[2]
        self.shrink = max(1, -(-max(height, width) // self.THUMB_SIZE)) # (Rounded up, so it fits in THUMB_SIZE)
        self.thumb_size = (width // self.shrink, height // self.shrink)
        self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()):
        return 0 if (self.movie is None or parent.isValid()) else self.movie.shape[0]

    def flags(self, index):
        return Qt.ItemIsEnabled

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DecorationRole:
            return self.thumbnail(index.row())
        if role == Qt.DisplayRole:
            return str(index.row() + 1)
        if role == Qt.TextAlignmentRole:
            return int(Qt.AlignHCenter | Qt.AlignVCenter)
        return None

    def thumbnail(self, row):
        pixmap = self.thumbs.get(row)
        if pixmap is None:
            frame, k = np.asarray(self.movie[row]), self.shrink
            h, w = frame.shape[0] // k * k, frame.shape[1] // k * k
            small = np.ascontiguousarray(frame[:h, :w].reshape(h // k, k, w // k, k).max(axis=(1, 3)))
            qimage = QImage(small.data, small.shape[1], small.shape[0], small.shape[1], QImage.Format_Grayscale8)
            pixmap = QPixmap(qimage) # (Copies the data, so `small` can go)
            self.thumbs[row] = pixmap
            if len(self.thumbs) > self.MAX_CACHED:
                self.thumbs.popitem(last=False)
        else:
            self.thumbs.move_to_end(row)
        return pixmap

class FrameSlider(QSlider):
    """ Horizontal slider for choosing a frame. Its values are the frame numbers as shown on screen (1..N).
        It draws everything itself (track, handle, ticks, markers), at fixed positions, and handles the mouse itself, so that
        it looks and works the same in every Qt style. (The styles disagree about where a slider's groove is: in the Windows
        ones it fills the whole height of the widget, which left no room for markers or ticks.)
        Top to bottom: the flash markers (a triangle pointing down at each flash, set_flash_frames, with a tooltip saying
        which frames), the track with its handle, then the ticks.
        Ticks are drawn at the finest spacing that stays legible: every frame if there's room for them (at least MIN_TICK_PX
        apart), otherwise every 2, 5, 10, 20, 50, ... frames, at multiples of that number, with a taller tick at every 10th of
        them and at the first and last frame. Clicking anywhere on the track jumps straight there, and it can then be dragged
        (the arrow keys step 1 frame, Page Up/Down step 10). """
    MIN_TICK_PX = 4
    TICK_MINOR, TICK_MAJOR = 2, 5 # Pixels
    MARK_HEIGHT, MARK_HALF_WIDTH = 8, 4 # Pixels: the triangle that marks a flash
    MARK_COLOR = QColor(230, 90, 20)
    MARGIN = 4 # Pixels, either side of the handle at its extremes
    HANDLE_W, HANDLE_H = 11, 18
    Y_TIP = MARK_HEIGHT + 1 # The markers' tips
    Y_HANDLE = Y_TIP + 2 # Top of the handle
    Y_TICKS = Y_HANDLE + HANDLE_H + 3 # Top of the ticks
    HEIGHT = Y_TICKS + TICK_MAJOR + 3

    def __init__(self, parent=None):
        super().__init__(Qt.Horizontal, parent)
        self.setTickPosition(QSlider.NoTicks)
        self.setMinimumHeight(self.HEIGHT)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setToolTip("Frame number. Click or drag; arrow keys step 1 frame, Page Up/Down 10")
        self._flashes = [] # (first, last) slider values of each flash
        self.set_frames(0)

    def sizeHint(self):
        return QtCore.QSize(200, self.HEIGHT)

    def minimumSizeHint(self):
        return QtCore.QSize(60, self.HEIGHT)

    def set_frames(self, n_frames):
        """ Set up for a movie of n_frames frames (0: no movie) """
        self.blockSignals(True)
        self.setRange(1, max(1, n_frames))
        self.setValue(1)
        self.setSingleStep(1)
        self.setPageStep(10)
        self.blockSignals(False)
        self.setEnabled(n_frames > 1)
        self.set_flash_frames([]) # (Also updates)

    def set_flash_frames(self, frames):
        """ Mark where the flashes are. frames: 0-based numbers of the flash frames; those within 2 frames of each other are one
            flash. (Slider values, and the numbers in the tooltip, are 1-based like the frame numbers on screen.) """
        events = []
        for f in sorted(int(n) + 1 for n in frames):
            if events and f - events[-1][1] <= 2:
                events[-1][1] = f
            else:
                events.append([f, f])
        self._flashes = [(a, b) for a, b in events]
        self.update()

    def flash_events(self):
        """ [(first, last)] slider values of each flash """
        return list(self._flashes)

    # --- where things are on screen
    def _x_range(self):
        """ (x of the handle's center at the minimum, at the maximum) """
        return self.MARGIN + self.HANDLE_W / 2.0, self.width() - self.MARGIN - self.HANDLE_W / 2.0

    def x_of(self, value):
        """ x (pixels) of the center of the handle when the slider has this value """
        left, right = self._x_range()
        n = self.maximum() - self.minimum()
        return left if n <= 0 else left + (right - left) * (value - self.minimum()) / float(n)

    def value_at(self, x):
        """ The value whose handle is centered nearest x """
        left, right = self._x_range()
        n = self.maximum() - self.minimum()
        if right <= left or n <= 0:
            return self.minimum()
        return int(min(self.maximum(), max(self.minimum(), self.minimum() + round((x - left) / (right - left) * n))))

    def tick_step(self):
        """ Frames between ticks: the smallest of 1, 2, 5, 10, 20, 50, ... that keeps them MIN_TICK_PX apart """
        n = self.maximum() - self.minimum()
        if n <= 0:
            return 1
        left, right = self._x_range()
        px_per_frame = max(1.0, right - left) / float(n)
        step = 1
        while step * px_per_frame < self.MIN_TICK_PX and step < n:
            step *= 2 if str(step)[0] in "15" else 2.5   # 1 -> 2 -> 5 -> 10 -> 20 -> 50 ...
            step = int(round(step))
        return step

    def tick_values(self):
        """ (frame number, is it a major tick) for every tick """
        lo, hi, step = self.minimum(), self.maximum(), self.tick_step()
        ticks = {lo: True, hi: True} # The first and last frame always
        x_lo, x_hi = self.x_of(lo), self.x_of(hi)
        first = ((lo + step - 1) // step) * step # Multiples of the step
        for v in range(first, hi + 1, step):
            x = self.x_of(v)
            if v in ticks or abs(x - x_lo) < self.MIN_TICK_PX or abs(x - x_hi) < self.MIN_TICK_PX:
                continue # (Not one squeezed up against an end tick)
            ticks[v] = (v % (10 * step) == 0)
        return sorted(ticks.items())

    # --- drawing
    def paintEvent(self, event):
        painter = QPainter(self)
        group = QPalette.Active if self.isEnabled() else QPalette.Disabled
        color = lambda role: self.palette().color(group, role)
        y_c = self.Y_HANDLE + self.HANDLE_H / 2.0
        x_handle = self.x_of(self.value())

        # The track, filled up to the handle
        left, right = self._x_range()
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(color(QPalette.Mid)))
        painter.drawRoundedRect(QtCore.QRectF(left, y_c - 2, right - left, 4), 2, 2)
        if self.maximum() > self.minimum():
            painter.setBrush(QBrush(color(QPalette.Highlight)))
            painter.drawRoundedRect(QtCore.QRectF(left, y_c - 2, x_handle - left, 4), 2, 2)

        # The handle
        border = color(QPalette.Highlight) if self.hasFocus() else color(QPalette.Dark)
        painter.setPen(QPen(border, 2 if self.hasFocus() else 1))
        painter.setBrush(QBrush(color(QPalette.Midlight) if self.isSliderDown() else color(QPalette.Button)))
        painter.drawRoundedRect(QtCore.QRectF(x_handle - self.HANDLE_W / 2.0 + 0.5, self.Y_HANDLE + 0.5, self.HANDLE_W - 1, self.HANDLE_H - 1), 3, 3)

        # The ticks
        if self.maximum() > self.minimum():
            tick_color = color(QPalette.WindowText)
            tick_color.setAlpha(170)
            painter.setPen(QPen(tick_color, 1))
            for value, major in self.tick_values():
                x = int(round(self.x_of(value)))
                painter.drawLine(x, self.Y_TICKS, x, self.Y_TICKS + (self.TICK_MAJOR if major else self.TICK_MINOR))

        # The flashes: a triangle pointing down at each
        mark = QColor(self.MARK_COLOR)
        mark.setAlpha(255 if self.isEnabled() else 110)
        painter.setPen(QPen(mark, 1))
        painter.setBrush(QBrush(mark))
        painter.setRenderHint(QPainter.Antialiasing, True)
        for first, last in self._flashes:
            x = (self.x_of(first) + self.x_of(last)) / 2.0
            half = max(self.MARK_HALF_WIDTH, (self.x_of(last) - self.x_of(first)) / 2.0 + 1)
            painter.drawPolygon(QPolygonF([QPointF(x - half, self.Y_TIP - self.MARK_HEIGHT), QPointF(x + half, self.Y_TIP - self.MARK_HEIGHT), QPointF(x, self.Y_TIP)]))
        painter.end()

    def event(self, e):
        if e.type() == QEvent.ToolTip and self._flashes: # Hovering over a flash marker says which frames it is
            for k, (first, last) in enumerate(self._flashes, 1):
                x0, x1 = self.x_of(first), self.x_of(last)
                if x0 - self.MARK_HALF_WIDTH <= e.pos().x() <= x1 + self.MARK_HALF_WIDTH and e.pos().y() <= self.Y_HANDLE:
                    QToolTip.showText(e.globalPos(), "Flash %d: %s" % (k, "frame %d" % first if first == last else "frames %d-%d" % (first, last)), self)
                    return True
        return super().event(e)

    # --- the mouse: click anywhere to jump there, and drag
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.isEnabled():
            self.setSliderDown(True)
            self.setValue(self.value_at(event.pos().x()))
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.isSliderDown():
            self.setValue(self.value_at(event.pos().x()))
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.isSliderDown():
            self.setSliderDown(False)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

class ZernikeDialog(QDialog):
    def createFormGroupBox(self,titl):
        formGroupBox = QGroupBox(titl)
        #layout = QFormLayout()
        layout = QGridLayout()
        num_zs = np.min( (NUM_ZERN_DIALOG,len(self.ui_parent.engine.zernikes) ) )
        self.lines = [QLineEdit() for n in np.arange(num_zs)]
        self.chks = [QCheckBox() for n in np.arange(num_zs)]
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
        log.debug(self.ui_parent.engine.zernikes[4] )

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
            line1 += "(%f)"%self.ui_parent.engine.offline.box_metrics[n]
        except:
            line1 += "(NONE)"
        line1 += "(%dx%d)" % (self.ui_parent.engine.box_size_pixel,
            self.ui_parent.engine.box_size_pixel)

        self.text_num.setText(line1)
        self.text_box.setText("box center=(%0.3f,%0.3f)"%(box_x,box_y))
        self.text_centroid.setText("centroid=(%0.3f,%0.3f)"%(centroid_x_abs,centroid_y_abs))
        self.box_pix=box_pix
        self.cent_x=cent_x
        self.cent_y=cent_y
        self.nbox = n

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
        
    def event(self, event): 
        if event.type() == QtCore.QEvent.EnterWhatsThisMode:
            self.ui_parent.engine.omits[self.nbox] = not ( self.ui_parent.engine.omits[self.nbox] )
            log.debug( "Omit: %d",self.nbox);
            return True
        return QDialog.event(self, event)
    
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
        
    def actuator_window_clicked(self, event):
        # Get the geometry of the spot window
        geometry = self.pixmap1.geometry()

        # Access position and size attributes
        x = geometry.x()
        y = geometry.y()
        width = geometry.width()
        height = geometry.height()
     
        # print("clicked:", event.pos() )
        x_scaled = event.pos().x() / width #* self.image_pixels.shape[1]
        y_scaled = event.pos().y() / height #* self.image_pixels.shape[0]
        log.debug("scaled: x,y ", x_scaled, y_scaled)
        
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
        self.pixmap1 = QPixmap.fromImage(self.qi).scaled(self.height(),self.width(),Qt.KeepAspectRatio)
        self.setPixmap( self.pixmap1 )

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
        log.debug(ev, ev.pos() )
        log.debug( self.getViewBox().viewRange() )
        if ev.button()==Qt.LeftButton:
            if (self.getViewBox().boundingRect().right() - self.getViewBox().mapFromScene(ev.pos()).x())<20:
                self.terms_expanded = not( self.terms_expanded )

    def eventFilter(self, obj, event):
        if event.type() == QEvent.ToolTip:
            pos_plot = self.getViewBox().mapSceneToView(event.pos())
            bar_which = round(pos_plot.x())
            try:
                zernike_val = self.app.engine.zernikes[bar_which-1]
                self.setToolTip("Z%d=%+0.3f"%(bar_which,zernike_val) )
            except IndexError:
                pass # Ok, bad position

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
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Low order aberrations across frames")

        self.sc = MplCanvas(self, width=5, height=4, dpi=100)
        #self.sc.axes.plot([0,1,2,3,4], [10,1,20,3,40])
        #self.setCentralWidget(sc)

        # Create a layout for the dialog
        layout = QVBoxLayout()
        layout.addWidget(self.sc)

        # Set the layout for the dialog
        self.setLayout(layout)
