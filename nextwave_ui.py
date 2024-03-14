from PyQt5.QtWidgets import (QMainWindow, QLabel, QSizePolicy, QApplication, QPushButton,
                             QHBoxLayout, QVBoxLayout, QGridLayout,
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

import matplotlib.cm as cmap

from nextwave_code import NextwaveEngineComm
from nextwave_sockets import NextwaveSocketComm

from threading import Thread

WINDOWS=(os.name == 'nt')

# TODO: Configurable?
QIMAGE_HEIGHT=1000
QIMAGE_WIDTH=1000

MAIN_HEIGHT_WIN=768
MAIN_WIDTH_WIN=1800
SPOTS_HEIGHT_WIN=768
SPOTS_WIDTH_WIN=768

NUM_ZERN_DIALOG=14 # TODO

# TODO
CAM_EXPO_MIN = 32./1000.0 # TODO
CAM_EXPO_MAX = 100000 # TODO
CAM_GAIN_MIN = 0
CAM_GAIN_MAX = 9.83

class ZernikeDialog(QDialog):
    def createFormGroupBox(self,titl):
        formGroupBox = QGroupBox(titl)
        layout = QFormLayout()
        self.lines = [QLineEdit() for n in np.arange(NUM_ZERN_DIALOG)]
        for nZernike,le in enumerate(self.lines):
            layout.addRow(QLabel("Z%2d"%(nZernike)) , le)
        formGroupBox.setLayout(layout)
        return formGroupBox

    def mycall(self,callback):
        zs = [str( l1.text()) for l1 in self.lines]
        zs = [0 if z1=='' else float(z1) for z1 in zs]
        callback(zs)

    def __init__(self,titl,callback):
        super().__init__()
        self.setWindowTitle(titl)
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok) # | QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(lambda: self.mycall(callback) )

        buttonBox.setWindowModality(Qt.ApplicationModal)

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.createFormGroupBox(titl))
        mainLayout.addWidget(buttonBox)
        self.setLayout(mainLayout)

class ActuatorPlot(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.generatePicture()
        #self.init_ui()

    #def init_ui(self):
        #self.show()
        #boxes1=[QLineF(self.engine.box_x[n]-box_size_pixel//2+BOX_BORDER, # top

        self.pixmap = QPixmap(10,10)

        bits=np.random.normal(size=(10,10))*32+128
        self.qi = QImage(bits*0,10,10,QImage.Format_Indexed8)

    def set_colors(self):
        #https://het.as.utexas.edu/HET/Software/html/qimage.html#image-transformations
        #https://stackoverflow.com/questions/35382088/qimage-custom-indexed-colors-using-setcolortable
        for n in np.arange(256):
            if 0<n<64:
                val=QtGui.qRgb(255-n,0,0)
            elif n>=(256-64): # TODO check this
                val=QtGui.qRgb(0,n,0)
            else:
                val=QtGui.qRgb(n,n,n) # Middle values are gray
            self.qi.setColor(n, val)

    def generatePicture(self):
        ## pre-computing a QPicture object allows paint() to run much more quickly, 
        ## rather than re-drawing the shapes every time.
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        p.setPen(pg.mkPen('w'))
        data = [  ## fields are (time, open, close, min, max).
    (1., 10, 13, 5, 15),
    (2., 13, 17, 9, 20),
    (3., 17, 14, 11, 23),
    (4., 14, 15, 5, 19),
    (5., 15, 9, 8, 22),
    (6., 9, 15, 8, 16),
]
        w = (data[1][0] - data[0][0]) / 3.
        for (t, open, close, min, max) in data:
            p.drawLine(QtCore.QPointF(t, min), QtCore.QPointF(t, max))
            if open > close:
                p.setBrush(pg.mkBrush('r'))
            else:
                p.setBrush(pg.mkBrush('g'))
            p.drawRect(QtCore.QRectF(t-w, open, w*2, close-open))
        p.drawPicture(0,0,self.picture)

    def paintEvent_manual(self): #, p, *args):
        #bits[0,0]=128
        #bits[9,9]=128
        #yybits=np.random.normal(0,255,size=(10,10), dtype='uint8' )
        #qi.setColorTable(self.ct)
        #self.generatePicture()
        #self.setStyleSheet("background-color:black;");
        #self.pixmap.fill(Qt.black)
        #for n in np.arange(256):
            #val = QtGui.qRgb(*self.ct[n])
            #self.qi.setColor(n, val)
        #self.qi.loadFromData(bits)

        bits=np.array(np.random.normal(size=(10,10))*32+128, dtype='uint8' )
        bits[0,0:3]=0
        bits[1,0:2]=0
        bits[2,0:1]=0
        bits[0,-3:]=0
        bits[1,-2:]=0
        bits[2,-1:]=0

        bits[-1,0:3]=0
        bits[-2,0:2]=0
        bits[-3,0:1]=0
        bits[-1,-3:]=0
        bits[-2,-2:]=0
        bits[-3,-1:]=0
        # calculate the total number of bytes in the frame 
        width=10
        height = 10
        totalBytes = bits.nbytes
        bytesPerLine = int(totalBytes/height)

        # Needed to fix skew problem.
        #https://stackoverflow.com/questions/41596940/qimage-skews-some-images-but-not-others

        #image = QImage(bits, width, height, bytesPerLine, QImage.Format_Grayscale8)

        #self.imageLabel.setPixmap(QPixmap.fromImage(image))

        self.qi = QImage(bits,width,height,bytesPerLine,QImage.Format_Indexed8)
        self.set_colors()

        #qp = QPainter(self.qi)
        #qp.setBrush(br)
        #qp.setPen(QtGui.QColor(200,0,0)) 
        #qp.setBrush(QtGui.QColor(200,0,0)) 
        #qp.drawRect(10, 10, 30,30)
        #qp.end()
        pixmap = QPixmap(self.qi)
        pixmap = pixmap.scaled(self.height(),self.width(),Qt.KeepAspectRatio)
        self.setPixmap(pixmap)

class MyBarWidget(pg.PlotWidget):

    sigMouseClicked = pyqtSignal(object) # add our custom signal

    def __init__(self, *args, **kwargs):
        super(MyBarWidget, self).__init__(*args, **kwargs)
        self.terms_expanded=False
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

    self.draw_refs = True
    self.draw_boxes = True
    self.draw_centroids = True
    self.draw_arrows = False
    self.draw_crosshair = True
    self.iterative_first=True

    #self.cx=518 # TODO
    #self.cy=488 # TODO
    self.cx=501 # TODO
    self.cy=499.5 # TODO

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

    self.params_offline_testbed = [
        {'name': 'system', 'type': 'group', 'title':'System Params', 'children': [
            {'name': 'wavelength', 'type': 'int', 'value': 830, 'title':'Wavelength (nm)', 'limits':[50,2000]},
            {'name': 'lenslet_f', 'type': 'float', 'value': 24, 'title':'Lenslet f', 'limits':[1,20]},
            {'name': 'lenslet_pitch', 'type': 'float', 'value': 328.0, 'title':'Lenslet pitch'},
            {'name': 'pixel_pitch', 'type': 'float', 'value': 5.5*2, 'title':'Pixel pitch (um)'},
            {'name': 'pupil_diam', 'type': 'float', 'value': 7.168, 'title':'Pupil diameter (mm)'},
            {'name': 'focal_length', 'type': 'float', 'value': 24, 'title':'Focal length'},
        ]}
        ]

    self.params_offline = [
        {'name': 'system', 'type': 'group', 'title':'System Params', 'children': [
            {'name': 'wavelength', 'type': 'int', 'value': 830, 'title':'Wavelength (nm)', 'limits':[50,2000]},
            {'name': 'lenslet_f', 'type': 'float', 'value': 5.12, 'title':'Lenslet f', 'limits':[1,20]},
            {'name': 'lenslet_pitch', 'type': 'float', 'value': 256, 'title':'Lenslet pitch'},
            {'name': 'pixel_pitch', 'type': 'float', 'value': 6.4, 'title':'Pixel pitch (um)'},
            {'name': 'pupil_diam', 'type': 'float', 'value': 6.4, 'title':'Pupil diameter (mm)'},
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
        pen = QPen(Qt.red, 2)
        painter.setPen(pen)
        points_ref=[QPointF(self.engine.ref_x[n],self.engine.ref_y[n]) for n in np.arange(self.engine.ref_x.shape[0])]
        painter.drawPoints(points_ref)

    if self.draw_boxes and self.engine.mode>1:
    #if self.get_param("UI","show_boxes"):
        pen = QPen(Qt.blue, 2.00, Qt.SolidLine)
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

    # Centroids:
    if self.draw_centroids and self.engine.mode>1:
        #for ncen,cen in enumerate(self.centroids_x):
            #if np.isnan(cen):
                #print(ncen,end=' ')

        pen = QPen(Qt.blue, 2)
        painter.setPen(pen)
        points_centroids=[QPointF(self.engine.centroids_x[n],self.engine.centroids_y[n]) for n in np.arange(self.engine.num_boxes)]
        painter.drawPoints(points_centroids)

    if self.draw_crosshair:
        pen = QPen(Qt.red, 2)
        painter.setPen(pen)
        CROSSHAIR_SIZE=20
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

    pixmap = pixmap.scaled(SPOTS_WIDTH_WIN,SPOTS_HEIGHT_WIN, Qt.KeepAspectRatio)
    self.pixmap_label.setPixmap(pixmap)
    #print ('%0.2f'%bytez.mean(),end=' ', flush=True);

    s=""
    for n in np.arange(13):
        s += 'Z%2d=%+0.4f\n'%(n+1,self.engine.zernikes[n])
    self.text_status.setText(s)

    rms,rms5p,cylinder,sphere,axes=self.engine.calc_diopters()
    left_chars=15
    str_stats=f"{'RMS':<15}= {rms:3.4f}\n"
    str_stats+=f"{'RMS(Z5+)':<15}= {rms5p:3.4f}\n"
    str_stats+=f"{'Sphere(+cyl)':<15}= {sphere:3.4f}\n"
    str_stats+=f"{'Sphere(-cyl)':<15}= {sphere:3.4f}\n"
    str_stats+=f"{'Cylinder':<15}= {cylinder:3.4f}\n"
    str_stats+=f"{'Axes(-cyl)':<15}= {axes:3.4f}\n"
    self.text_stats.setText(str_stats)
    #self.text_stats.setHtml(str_stats) # TODO: To get other colors, can embed <font color="red">TEXT</font><br>, etc.

    self.line_centerx.setText(str(self.cx) )
    self.line_centery.setText(str(self.cy) )

    if self.engine.fps0!=0:
        s="Running. %3.2f FPS (%04.1f ms: %04.1f+%04.1f ms)"%(1000/self.engine.fps0,self.engine.fps0, self.engine.fps1, self.engine.fps2)
    else:
        s="0 fps"
        
    self.label_status0.setText(s)
    self.label_status0.setStyleSheet("color: rgb(0, 255, 0); background-color: rgb(0,0,0);")

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

    if not np.isnan(maxval):
        lim = np.max( (np.abs(minval), np.abs(maxval)) )
        self.bar_plot.setYRange(-lim, lim)
    #self.bar_plot.getAxis('left').setTickSpacing(1, 0.1)
    #self.bar_plot.getAxis('left').setTickDensity(2)
    #colr2=np.array(cmap.Spectral(0.8))*255
    #bg2 = pg.BarGraphItem(x=np.arange(4)+3, height=self.engine.zernikes[5:9], width=1.0, brush=colr2)

    #self.bar_plot.addItem(bg3)
 def update_ui_dm(self):
    if self.chkLoop.isChecked():
        self.actuator_plot.paintEvent_manual()

 def set_follow(self,state):
    buf = ByteStream()
    buf.append(int(state)*2)
    self.shmem_boxes.seek(0)
    self.shmem_boxes.write(buf)
    self.shmem_boxes.flush()

 def showdialog(self,which,callback):
     dlg = ZernikeDialog(which, callback)
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

 def get_param(self,name_parent,name,offline=False):
    if offline:
        return self.params_offline["children"][name_parent]["children"][name]["value"] 
    else:
        return self.params["children"][name_parent]["children"][name]["value"] 

 def set_param(self,name_parent,name,newval,offline=False):
    if offline:
        self.params["children"][name_parent]["children"][name]["value"] = newval
    else:
        self.params["children"][name_parent]["children"][name]["value"] = newval

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
     layout=QVBoxLayout()
     pixmap_label = QLabel()
     #pixmap_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
     pixmap_label.resize(SPOTS_WIDTH_WIN,SPOTS_HEIGHT_WIN)
     pixmap_label.setAlignment(Qt.AlignCenter)
     self.pixmap_label=pixmap_label

     im_np = np.ones((QIMAGE_HEIGHT,QIMAGE_WIDTH),dtype='uint8')
     #im_np = np.transpose(im_np, (1,0,2))
     qimage = QImage(im_np, im_np.shape[1], im_np.shape[0],
                     QImage.Format_Mono)
     pixmap = QPixmap(qimage)
     pixmap = pixmap.scaled(SPOTS_WIDTH_WIN,SPOTS_HEIGHT_WIN, Qt.KeepAspectRatio)
     pixmap_label.setPixmap(pixmap)
     pixmap_label.mousePressEvent = self.butt
     layout.addWidget(pixmap_label,15)
     self.bar_plot = MyBarWidget()
     self.bar_plot.app = self
     layout.addWidget(self.bar_plot,5)
     self.widget_centrals.setLayout(layout)

     self.widget_displays = QWidget()
     layout=QVBoxLayout(self.widget_displays)
     layout.addWidget(QGroupBox('Pupil'))
     #layout.addWidget(QGroupBox('DM'))
     self.actuator_plot = ActuatorPlot()
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

     l1 = QHBoxLayout()

     self.chkFollow = QCheckBox("Boxes follow centroids")
     self.chkFollow.stateChanged.connect(lambda:self.set_follow(self.chkFollow.isChecked()))
     l1.addWidget(self.chkFollow)

     btn = QPushButton("Search box shift")
     btn.clicked.connect(lambda: self.showdialog("Shift search boxes", self.engine.shift_search_boxes ) )
     l1.addWidget(btn)

     btn = QPushButton("Reference shift")
     btn.clicked.connect(lambda: self.showdialog("Shift references", self.shift_references ) )
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
     self.line_centerx = QLineEdit()
     self.line_centerx.setMaxLength(5)
     layout1.addWidget(self.line_centerx,0,1)
     self.line_centery = QLineEdit()
     self.line_centery.setMaxLength(5)
     layout1.addWidget(self.line_centery,1,1)

     btnFind = QPushButton("Find center")
     btnFind.setStyleSheet("color : orange")
     layout1.addWidget(btnFind,2,1)

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
     layout1.addWidget(self.chkBackSubtract,0,0)
     btnBackSet = QPushButton("Set background")
     layout1.addWidget(btnBackSet,0,1)
     btnBackReset = QPushButton("Reset Background") # d
     #btnBackReset.setStyleSheet("color : orange")
        # adding action to a button 
        #button.clicked.connect(self.clickme) 
     lbl = QLabel("Exposure time (ms)")
     layout1.addWidget(lbl,1,0)

     self.slider_exposure = QSlider(orientation=Qt.Horizontal)
     self.slider_exposure.setMinimum(0) # TODO: Get from camera
     self.slider_exposure.setMaximum(100) # TODO: Get from camera
     layout1.addWidget(self.slider_exposure,1,1)
     self.slider_exposure.valueChanged.connect(self.slider_exposure_changed)

     self.exposure = QDoubleSpinBox()
     layout1.addWidget(self.exposure,1,2)
     self.exposure.setDecimals(4)
     self.exposure.setMinimum(CAM_EXPO_MIN)
     self.exposure.setMaximum(CAM_EXPO_MAX)

     lbl = QLabel("Gain (dB)")
     layout1.addWidget(lbl,2,0)

     self.slider_gain = QSlider(orientation=Qt.Horizontal)
     self.slider_gain.setMinimum(0) # TODO: Get from camera
     self.slider_gain.setMaximum(100) # TODO: Get from camera
     layout1.addWidget(self.slider_gain,2,1)
     self.slider_gain.valueChanged.connect(self.slider_gain_changed)

     self.gain = QDoubleSpinBox()
     layout1.addWidget(self.gain,2,2)
     self.gain.setMinimum(CAM_GAIN_MIN)
     self.gain.setMaximum(CAM_GAIN_MAX)

     ### DM Ops
     layout1 = QGridLayout(self.ops_dm)
     self.chkLoop = QCheckBox("Close AO Loop")
     layout1.addWidget(self.chkLoop,0,0)

     btn = QPushButton("Search box shift")
     btn.clicked.connect(lambda: self.showdialog("Shift search boxes", self.engine.shift_search_boxes ) )
     layout1.addWidget(btn, 1,0 )

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

     self.mode_btn2.setEnabled( True )
     self.mode_btn3.setEnabled( True )
     #self.mode_btn4.setEnabled( False )

     # Config
     layout1 = QGridLayout(pages[2])
     lbl = QLabel("XML Config: ")
     layout1.addWidget(lbl, 0,0)
     self.edit_xml = QLineEdit("c:\\file\\ao\\test.xml")
     layout1.addWidget(self.edit_xml, 0,1)
     btn = QPushButton("...")
     layout1.addWidget(btn, 0,2)
     btn = QPushButton("Edit+Reload")
     layout1.addWidget(btn, 0,3)
     #os.system('c:/tmp/sample.txt') # <- on windows this will launch the defalut editor

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

     menu=self.menuBar().addMenu('&File')
     menu.addAction('&Export Centroids & Zernikes', self.export)
     menu.addAction('e&Xit', self.close)

     pixmap_label.setFocus()

     self.setGeometry(2,2,MAIN_WIDTH_WIN,MAIN_HEIGHT_WIN)
     self.show()

 def slider_exposure_changed(self):
     scaled = 10**( float( self.slider_exposure.value())/100.0*np.log10(CAM_EXPO_MAX/CAM_EXPO_MIN)+np.log10(CAM_EXPO_MIN))
     self.exposure.setValue(scaled)

 def slider_gain_changed(self):
     scaled = self.slider_gain.value()/100.0*CAM_GAIN_MAX
     self.gain.setValue(scaled)


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
         self.m=10
     else:
         self.m=1

 def move_center(self, dx, dy, m=1, do_update=True):
    m = self.m
    self.cx += dx * m
    self.cy += dy * m
    if do_update:
        self.engine.move_searchboxes(dx*m, dy*m)

 def butt(self, event):
    print("clicked:", event.pos() )
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
    #self.engine.mode_stop()
    self.sockets.camera.send(b"HI")

 def export(self):
    default_filename="centroids.dat"
    filename, _ = QFileDialog.getSaveFileName(
        self, "Save audio file", default_filename, "Audio Files (*.mp3)"
    )
    if filename:
        fil = open(filename,'wt')

        fil.writelines( '[image size = %dx%d]\n'%(QIMAGE_WIDTH,QIMAGE_HEIGHT)) # TODO
        fil.writelines( '[pupil = %f,%d,%d]\n'%(PUPIL,self.cx,self.cy)) # TODO
        for nbox in np.arange( len(self.engine.ref_x)):
            # TODO: Valid or invalid
            fil.writelines('%d\t%f\t%f\t%f\t%f\t\n'%(1,self.engine.ref_x[nbox], self.engine.ref_y[nbox], self.centroids_x[nbox], self.centroids_y[nbox] ) )
        fil.close()

    filename="zernikes.txt"
    fil = open(filename,'wt')
    for val in self.zernikes:
        fil.writelines('%f\n'%val)
    fil.close()

    np.save('dx',self.spot_displace_x)
    np.save('dy',self.spot_displace_y)
    np.save('slope',self.slope)
    np.save('zterms',self.zterms)
    np.save('zterms_inv',self.zterms_inv)

 def close(self):
    self.engine.send_quit() # Send stop command to engine
    self.app.exit()

def main():
  app = QApplication(sys.argv)
  win = NextWaveMainWindow()
  win.app = app
  win.initUI()
  sys.exit(app.exec_())


if __name__=="__main__":
  main()
