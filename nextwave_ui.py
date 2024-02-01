from PyQt5.QtWidgets import (QMainWindow, QLabel, QSizePolicy, QApplication, QPushButton,
                             QHBoxLayout, QVBoxLayout, QWidget, QGroupBox, QTabWidget, QTextEdit,
                             QFileDialog, QCheckBox, QDialog, QFormLayout, QDialogButtonBox, QLineEdit)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QTimer, QLineF, QPointF
import PyQt5.QtGui as QtGui

import pyqtgraph as pg

import numpy as np
import sys
import os

import matplotlib.cm as cmap
#from numba import jit

import mmap
import struct
import extract_memory

import zernike_integrals
from numpy.linalg import svd,lstsq

WINDOWS=(os.name == 'nt')

NUM_ZERNIKES=67 # TODO
NUM_ZERN_DIALOG=14 # TODO
START_ZC=1
NUM_ZCS=68

#0=horizontal, 1=vertical,3=defocus?

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

# TODO: read from butter
QIMAGE_HEIGHT=1000
QIMAGE_WIDTH=1000

#CCD_PIXEL=5.5 * 2
#BOX_UM=328

CCD_PIXEL=6.4
BOX_UM=CCD_PIXEL * 40

box_size_pixel=BOX_UM/CCD_PIXEL # /2: down-sampling ?
print(box_size_pixel)

MAIN_HEIGHT_WIN=1000
MAIN_WIDTH_WIN=1800
SPOTS_HEIGHT_WIN=1024
SPOTS_WIDTH_WIN=1024

MEM_LEN=512
MEM_LEN_DATA=2048*2048*4

# TODO: Put in config file
UI_UPDATE_RATE_MS=2000

# TODO
PUPIL=6.4/2.0
PUPIL_RADIUS_MM=PUPIL
pupil_radius_pixel=PUPIL_RADIUS_MM*1000/CCD_PIXEL
RI_RATIO=pupil_radius_pixel/box_size_pixel
print(pupil_radius_pixel)
FOCAL=5.9041

ri_ratio = pupil_radius_pixel / box_size_pixel
print(ri_ratio)

x=np.arange( QIMAGE_WIDTH )
y=np.arange( QIMAGE_HEIGHT )
X,Y=np.meshgrid(x,y)

class ByteStream(bytearray):
    def append(self, v, fmt='B'):
        self.extend(struct.pack(fmt, v))

class NextwaveEngineComm():
    """ Class to manage:
          - Structures needed for realtime engine (boxes/refs, computed centroids, etc.)
          - Communication with the realtime engine (comm. over shared memory)
          - Computation of Zernikes, matrices for SVD, etc.
    """

    def make_searchboxes(self,cx,cy,pupil_radius_pixel,box_size_pixel,img_max=1000,aperture=1.0):
        """
        Like the MATLAB code to make equally spaced boxes, but doesn't use loops.
        Instead builds and filters arrays

        #"Loopy" code looks like the following:
        for y in np.arange(limit, -limit-1, -1):
            for x in np.arange(-limit, limit+1):
                yy = y + 0.5 * np.sign(y)
                xx = x + 0.5 * np.sign(x)
                rad = np.sqrt(xx*xx + yy*yy)
                if rad <= ri_ratio:
                    count=count+1
                    b_x += [x / ri_ratio]
                    b_y += [y / ri_ratio]

        """

        aperture = ri_ratio * aperture

        print( pupil_radius_pixel, box_size_pixel, ri_ratio)

        # The max number of boxes possible + or -
        max_boxes = np.ceil( pupil_radius_pixel/ box_size_pixel )

        # All possible bilinear box
        boxes_x = np.arange(-max_boxes,max_boxes+1) # +1 to include +max_boxes number
        boxes_y = np.arange(-max_boxes,max_boxes+1)

        # Determine outer edge of each box using corners away from the center:
        # 0.5*sign: positive adds 0.5, negative substracts 0.5
        XX,YY = np.meshgrid(boxes_x, boxes_y )

        RR = np.sqrt( (XX+0.5*np.sign(XX))**2 + (YY+0.5*np.sign(YY))**2 )
        valid_boxes = np.where(RR<aperture)
        max_dist_boxwidths = np.max(RR[RR<aperture])

        # Normalize to range -1 .. 1 (vs. pupil size)
        valid_x_norm=XX[valid_boxes]/ri_ratio
        valid_y_norm=YY[valid_boxes]/ri_ratio

        num_boxes=valid_x_norm.shape[0]
        box_zero = np.where(valid_x_norm**2+valid_y_norm**2==0)[0] # Index of zeroth (middle) element

        # TODO: check this
        #img_max = 992  # TODO
        MULT = img_max / 2.0 # 1000 / 2.0
        valid_x = valid_x_norm * MULT + cx
        valid_y = valid_y_norm * MULT + cy

        self.valid_x = valid_x
        self.valid_y = valid_y
        self.valid_x_norm = valid_x_norm
        self.valid_y_norm = valid_y_norm

        self.box_x = valid_x
        self.box_y = valid_y
        self.ref_x = self.box_x
        self.ref_y = self.box_y
        self.norm_x = valid_x_norm
        self.norm_y = valid_y_norm
        self.initial_x = valid_x
        self.initial_y = valid_y
        self.update_zernike_svd() # Precompute

        self.num_boxes= num_boxes

        self.send_searchboxes(self.shmem_boxes, self.box_x, self.box_y, self.layout_boxes)
        self.update_zernike_svd()

        return valid_x,valid_y,valid_x_norm,valid_y_norm

    def rcv_searchboxes(self,shmem_boxes, layout, box_x, box_y, layout_boxes):
        fields=layout[1]

        adr=fields['box_x']['bytenum_current']
        shmem_boxes.seek(adr)
        box_buf=shmem_boxes.read(NUM_BOXES*4)
        box_x = [struct.unpack('f',box_buf[n*4:n*4+4]) for n in np.arange(NUM_BOXES)]
        #box_x =np.frombuffer(box_buf, dtype='uint8', count=NUM_BOXES )
        #print(box_x[0], box_x[1], box_x[2], box_x[3])

        adr=fields['box_y']['bytenum_current']
        shmem_boxes.seek(adr)
        box_buf=shmem_boxes.read(NUM_BOXES*4)
        box_y = [struct.unpack('f',box_buf[n*4:n*4+4]) for n in np.arange(NUM_BOXES)]
        #box_x =np.frombuffer(box_buf, dtype='uint8', count=NUM_BOXES )

        return box_x,box_y

    def send_searchboxes(self,shmem_boxes, box_x, box_y, layout_boxes):
        defs=layout_boxes[2]
        fields=layout_boxes[1]

        num_boxes=box_x.shape[0]
        #box_size_pixel=box_size_pixel

        buf = ByteStream()
        for item in box_x:
            buf.append(item, 'f')
        shmem_boxes.seek(fields['box_x']['bytenum_current'])
        shmem_boxes.write(buf)
        shmem_boxes.flush()

        buf = ByteStream()
        for item in box_y:
            buf.append(item, 'f')
        shmem_boxes.seek(fields['box_y']['bytenum_current'])
        shmem_boxes.write(buf)
        shmem_boxes.flush()

        # Write header last, so the engine knows when we are ready
        buf = ByteStream()
        buf.append(1)
        buf.append(0)
        buf.append(num_boxes, 'H') # unsigned short
        buf.append(CCD_PIXEL,'d')
        buf.append(BOX_UM, 'd')
        buf.append(pupil_radius_pixel*CCD_PIXEL, 'd')
        shmem_boxes.seek(0)
        shmem_boxes.write(buf)
        shmem_boxes.flush()
        #print(num_boxes)

        # REMOVE ME:
    # @jit(nopython=True)
    def find_centroids(boxes,data,weighted_x,weighted_y,nboxes):
        centroids=np.zeros((nboxes,2) )
        for nbox in range(nboxes):
            box1=boxes[nbox]
            left=box1[0]-boxsize//2
            right=box1[0]+boxsize//2
            upper=box1[1]-boxsize//2
            lower=box1[1]+boxsize//2

            pixels=data[upper:lower,left:right]
            pixels_weighted_x=weighted_x[upper:lower,left:right]
            pixels_weighted_y=weighted_y[upper:lower,left:right]

            centroids[nbox,0]=np.sum(pixels_weighted_x)/np.sum(pixels)
            centroids[nbox,1]=np.sum(pixels_weighted_y)/np.sum(pixels)
        return centroids

    def update_zernike_svd(self):
        lefts =  self.norm_x - 0.5/RI_RATIO
        rights = self.norm_x + 0.5/RI_RATIO
        ups =    -(self.norm_y + 0.5/RI_RATIO)
        downs =  -(self.norm_y - 0.5/RI_RATIO)

        # Compute all integrals for all box corners 
        lenslet_dx,lenslet_dy=zernike_integrals.zernike_integral_average_from_corners(
            lefts, rights, ups, downs, PUPIL_RADIUS_MM)
        lenslet_dx = lenslet_dx[START_ZC:NUM_ZCS,:].T
        lenslet_dy = lenslet_dy[START_ZC:NUM_ZCS,:].T

        #  Compute SVD of stacked X and Y
        zpoly = np.hstack( (lenslet_dx.T, lenslet_dy.T ) ).T
        [uu,ss,vv] = svd(zpoly,False)

        # Pre-compute&save the zterms that are multiplied with the slopes in realtime
        ss_full = np.eye(ss.shape[0])*ss
        leftside = lstsq(ss_full, vv, rcond=0)[0].T # Python equiv to MATLAB's vv/ss (solving system of eqns) is lstsq
        # https://stackoverflow.com/questions/1001634/array-division-translating-from-matlab-to-python
        self.zterms = np.matmul( leftside, uu.T)
        self.zterms_inv = np.linalg.pinv(self.zterms)

    def compute_zernikes(self):
        # find slope
        spot_displace_x = self.ref_x - self.centroids_x
        spot_displace_y = -(self.ref_y - self.centroids_y)

        # Not sure whether should do this:
        #spot_displace_x -= spot_displace_x.mean()
        #spot_displace_y -= spot_displace_y.mean()
        #print( spot_displace_y.mean(), spot_displace_x.mean() )

        slope = np.concatenate( (spot_displace_y, spot_displace_x)) * CCD_PIXEL/FOCAL;

        self.spot_displace_x = spot_displace_x
        self.spot_displace_y = spot_displace_y
        self.slope = slope

        coeff=np.matmul(self.zterms,slope)

        # Copied from MATLAB code
        self.CVS_to_OSA_map = np.array([3,2, 5,4,6, 9,7,8,10, 15,13,11,12,14,
                                21,19,17,16,18,20,
                                27,25,23,22,24,26,28, 35,33,31,29,30,32,34,36,
                                45,43,41,39,37,38,40,42,44,
                                55,53,51,49,47,46,48,50,52,54,
                                65,63,61,59,57,56,58,60,62,64,66,67,68,69,70])
        self.OSA_to_CVS_map = np.array( [np.where(self.CVS_to_OSA_map-2==n)[0][0] for n in np.arange(20) ] ) # TODO
        self.OSA_to_CVS_map += 2
        self.zernikes=coeff[self.CVS_to_OSA_map[START_ZC-1:NUM_ZCS-1]-START_ZC-1 ]

    def shift_search_boxes(self,zs):
        zern_new = np.zeros(NUM_ZERNIKES)
        #zern_new[self.OSA_to_CVS_map[0:NUM_ZERN_DIALOG]]=zs 

        #zern_new[0:NUM_ZERN_DIALOG]=zs 
        # TODO: What about term 0?
        zern_new[self.CVS_to_OSA_map[0:NUM_ZERN_DIALOG-1]-START_ZC-1 ] = zs[1:]
        #print(self.OSA_to_CVS_map)
        #print( zern_new[0:9] )

        delta=np.matmul(self.zterms_inv,zern_new) 
        num_boxes = self.engine.box_x.shape[0] 
        self.engine.box_y = self.initial_y + delta[0:num_boxes]
        self.engine.box_x = self.initial_x - delta[num_boxes:]

        send_searchboxes(self.shmem_boxes, self.box_x, self.box_y, self.layout_boxes)

    def shift_references(self,zs):
        zern_new = np.zeros(NUM_ZERNIKES)
        #zern_new[self.OSA_to_CVS_map[0:NUM_ZERN_DIALOG]]=zs 
        #zern_new[0:NUM_ZERN_DIALOG]=zs 
        # TODO: What about term 0?
        zern_new[self.CVS_to_OSA_map[0:NUM_ZERN_DIALOG-1]-START_ZC-1 ] = zs[1:]
        delta=np.matmul(self.zterms_inv,zern_new) 
        num_boxes = self.box_x.shape[0] 
        self.engine.ref_y = (self.initial_y + delta[0:num_boxes])
        self.engine.ref_x =  self.initial_x - delta[num_boxes:]
        self.update_zernike_svd()

    def init(self):
        self.layout=extract_memory.get_header_format('memory_layout.h')
        self.layout_boxes=extract_memory.get_header_format('layout_boxes.h')

        # Could be math in the defines for sizes, use eval
        MEM_LEN=int( eval(self.layout[2]['SHMEM_HEADER_SIZE'] ) )
        MEM_LEN_DATA=int(eval(self.layout[2]['SHMEM_BUFFER_SIZE'] ) )
        if WINDOWS:
            self.shmem_hdr=mmap.mmap(-1,MEM_LEN,"NW_SRC0_HDR")
            self.shmem_data=mmap.mmap(-1,MEM_LEN_DATA,"NW_SRC0_BUFFER")
            self.shmem_boxes=mmap.mmap(-1,self.layout_boxes[0],"NW_BUFFER2")

            #from multiprocessing import shared_memory
            #self.shmem_hdr = shared_memory.SharedMemory(name="NW_SRC0_HDR" ).buf
            #self.shmem_data = shared_memory.SharedMemory(name="NW_SRC0_BUFFER" ).buf
            #self.shmem_boxes = shared_memory.SharedMemory(name="NW_BUFFER2").buf
        else:
            fd1=os.open('/dev/shm/NW_SRC0_HDR', os.O_RDWR)
            self.shmem_hdr=mmap.mmap(fd1, MEM_LEN)
            fd2=os.open('/dev/shm/NW_SRC0_BUFFER', os.O_RDWR)
            self.shmem_data=mmap.mmap(fd2,MEM_LEN_DATA)
            fd3=os.open('/dev/shm/NW_BUFFER2', os.O_RDWR)
            self.shmem_boxes=mmap.mmap(fd3,self.layout_boxes[0])

        bytez =np.array([CCD_PIXEL, PUPIL*2, BOX_UM], dtype='double').tobytes() 
        fields = self.layout_boxes[1]
        self.shmem_boxes.seek(fields['pixel_um']['bytenum_current'])
        self.shmem_boxes.write(bytez)
        self.shmem_boxes.flush()

    def receive_image(self):
        # TODO: Wait until it's safe (unlocked)
        mem_header=self.shmem_hdr.seek(0)
        mem_header=self.shmem_hdr.read(MEM_LEN)
        #fps=extract_memory.get_item(self.layout,mem_header,'fps')
        self.fps=extract_memory.get_array_item(self.layout,mem_header,'fps',0)
        self.height=extract_memory.get_array_item(self.layout,mem_header,'dimensions',0)
        self.width=extract_memory.get_array_item(self.layout,mem_header,'dimensions',1)

        self.shmem_data.seek(0)
        im_buf=self.shmem_data.read(self.width*self.height)
        bytez =np.frombuffer(im_buf, dtype='uint8', count=self.width*self.height )
        bytes2=np.reshape(bytez,( self.height,self.width)).copy()
        #bytes2 = bytes2.T.copy()

        bytesf = bytes2 / np.max(bytes2)

        if False: #self.chkFollow.isChecked():
            box_x,box_y=rcv_searchboxes(self.shmem_boxes, self.layout_boxes, 0, 0, 0 )
            self.box_x = np.array(box_x)
            self.box_y = np.array(box_y)

        return bytes2

    def receive_centroids(self):
        fields=self.layout_boxes[1]
        self.shmem_boxes.seek(fields['centroid_x']['bytenum_current'])
        buf=self.shmem_boxes.read(self.num_boxes*4)
        self.centroids_x=struct.unpack_from(''.join((['f']*self.num_boxes)), buf)

        self.shmem_boxes.seek(fields['centroid_y']['bytenum_current'])
        buf=self.shmem_boxes.read(self.num_boxes*4)
        self.centroids_y=struct.unpack_from(''.join((['f']*self.num_boxes)), buf)

    def send_quit(self):
        buf = ByteStream()
        buf.append(99) # Status:Quitter
        buf.append(0)  # Lock
        buf.append(1, 'H') # NUM BOXES. Hopefully doesn't matter
        #buf.append(40, 'd')
        #buf.append(500, 'd')
        self.shmem_boxes.seek(0)
        self.shmem_boxes.write(buf)
        self.shmem_boxes.flush()

class NextWaveMainWindow(QMainWindow):
 def __init__(self):
    super().__init__(parent=None)

    # self.worker=Worker();
    #self.worker_thread=QThread()
    #self.worker.moveToThread(self.worker_thread);
    #self.worker_thread.start();

    self.updater = QTimer(self);
    self.updater.timeout.connect(self.update_ui)
    self.updater.start(UI_UPDATE_RATE_MS)

    self.draw_refs = True
    self.draw_boxes = True
    self.draw_centroids = True
    self.draw_arrows = False
    self.draw_crosshair = True
    #self.cx=518 # TODO
    #self.cy=488 # TODO
    self.cx=501 # TODO
    self.cy=499.5 # TODO

    self.engine = NextwaveEngineComm()
    self.engine.init()

    box_x,box_y,box_norm_x,box_norm_y=self.engine.make_searchboxes(self.cx,self.cy,pupil_radius_pixel,box_size_pixel)
    #self.engine.send_searchboxes(self.engine.shmem_boxes, self.engine.box_x, self.engine.box_y, self.engine.layout_boxes)
    #self.engine.box_x = box_x
    #self.engine.box_y = box_y
    #self.engine.ref_x = box_x
    #self.engine.ref_y = box_y
    #self.engine.norm_x = box_norm_x
    #self.engine.norm_y = box_norm_y
    #self.initial_x = box_x
    #self.initial_y = box_y
    #self.update_zernike_svd() # Precompute

    #self.setFixedSize(1024,800)
    #self.move( 100,100 )
    #self.x = 2048/2
    #self.y = 2048/2

 def update_ui(self):
    image_pixels = self.engine.receive_image()
    self.engine.receive_centroids()

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

    if self.draw_refs:
        pen = QPen(Qt.red, 2)
        painter.setPen(pen)
        points_ref=[QPointF(self.engine.ref_x[n],self.engine.ref_y[n]) for n in np.arange(self.engine.ref_x.shape[0])]
        painter.drawPoints(points_ref)

    if self.draw_boxes:
        pen = QPen(Qt.blue, 1, Qt.DotLine)
        painter.setPen(pen)
        BOX_BORDER=2
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

    # Centroids:
    if self.draw_centroids:
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

    self.engine.compute_zernikes()
    s=""
    for n in np.arange(13):
        s += 'Z%2d=%+0.4f\n'%(n+1,self.engine.zernikes[n])
    self.text_status.setText(s)

    self.text_stats.setText("Hi there")

    s="Running. %3.2f FPS (%3.0f ms)"%(1000/self.engine.fps,self.engine.fps)
    self.label_status0.setText(s)
    self.label_status0.setStyleSheet("color: rgb(0, 255, 0); background-color: rgb(0,0,0);")

    colr1=np.array(cmap.Spectral(0.5))*255
    bg1 = pg.BarGraphItem(x=np.arange(3), height=self.engine.zernikes[2:5], width=1.0, brush=colr1)
    colr2=np.array(cmap.Spectral(0.8))*255
    bg2 = pg.BarGraphItem(x=np.arange(4)+3, height=self.engine.zernikes[5:9], width=1.0, brush=colr2)

    self.bar_plot.clear()
    self.bar_plot.addItem(bg1)
    self.bar_plot.addItem(bg2)
    #self.bar_plot.addItem(bg3)


 def set_follow(self,state):
    buf = ByteStream()
    buf.append(int(state)*2)
    self.shmem_boxes.seek(0)
    self.shmem_boxes.write(buf)
    self.shmem_boxes.flush()
 def showdialog(self,which,callback):
     dlg = ZernikeDialog(which, callback)
     dlg.exec()

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
     self.bar_plot = pg.PlotWidget()
     layout.addWidget(self.bar_plot,1)
     self.widget_centrals.setLayout(layout)

     self.widget_displays = QWidget()
     layout=QVBoxLayout()
     self.widget_displays.setLayout(layout)
     layout.addWidget(QGroupBox('Pupil'))
     layout.addWidget(QGroupBox('AO'))
     layout.addWidget(QGroupBox('Wavefront'))
     layout.addWidget(QGroupBox('PSF'))

     self.widget_controls = QWidget()
     layout=QVBoxLayout()
     tabs = QTabWidget()
     tabs.setTabPosition(QTabWidget.North)
     tabs.setMovable(True)

     panel_names = ["Source", "Options", "Coords", "Boxes","Camera"]
     pages = [QWidget(tabs) for nam in panel_names]
     l1 = QHBoxLayout()

     self.chkFollow = QCheckBox("Boxes follow centroids")
     self.chkFollow.stateChanged.connect(lambda:self.set_follow(self.chkFollow.isChecked()))
     l1.addWidget(self.chkFollow)

     btn = QPushButton("Search box shift")
     btn.clicked.connect(lambda: self.showdialog("Shift search boxes", self.shift_search_boxes ) )
     l1.addWidget(btn)

     btn = QPushButton("Reference shift")
     btn.clicked.connect(lambda: self.showdialog("Shift references", self.shift_references ) )
     l1.addWidget(btn)

     pages[3].setLayout(l1)
     for n, tabnames in enumerate(panel_names):
         tabs.addTab(pages[n], tabnames)
     #self.setCentralWidget(tabs)

     self.widget_status_buttons = QWidget()
     layoutStatusButtons = QHBoxLayout()
     self.status_btn1 = QPushButton("Init")
     layoutStatusButtons.addWidget(self.status_btn1)
     self.status_btn2 = QPushButton("Run")
     layoutStatusButtons.addWidget(self.status_btn2)
     self.status_btn3 = QPushButton("AO")
     layoutStatusButtons.addWidget(self.status_btn3)
     self.widget_status_buttons.setLayout(layoutStatusButtons)
     layout.addWidget(self.widget_status_buttons,1)

     self.label_status0 = QLabel("Status: ")
     layout.addWidget(self.label_status0, 1)

     self.text_status = QTextEdit()
     self.text_status.setReadOnly(True)

     layout.addWidget(self.text_status, 20)
     layout.addWidget(tabs, 20)

     #self.widget_controls = QGroupBox('Controls')
     self.widget_controls.setLayout(layout)

     #layout.addWidget(QGroupBox('Statistics'), 20)
     self.text_stats = QTextEdit()
     layout.addWidget(self.text_stats)

     # Main Widget
     self.widget_main = QWidget()
     layoutCentral = QHBoxLayout()
     layoutCentral.addWidget(self.widget_centrals, stretch=3)
     layoutCentral.addWidget(self.widget_displays)
     layoutCentral.addWidget(self.widget_controls, stretch=1)
     self.widget_main.setLayout(layoutCentral)

     self.setCentralWidget(self.widget_main)

     menu=self.menuBar().addMenu('&File')
     menu.addAction('&Export Centroids & Zernikes', self.export)
     menu.addAction('e&Xit', self.close)

     pixmap_label.setFocus()

     self.setGeometry(2,2,MAIN_WIDTH_WIN,MAIN_HEIGHT_WIN)
     self.show()


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
    elif event.key()==ord('C'):
        self.draw_centroids = not( self.draw_centroids )
    elif event.key()==ord('X'):
        self.draw_crosshair = not( self.draw_crosshair )
    elif event.key()==ord('Q'):
        self.close()
    elif event.key()==Qt.Key_Control:
        self.key_control = True
    elif event.key()==Qt.Key_Left:
        self.cx -= 1 + 10 * self.key_control
        update_search_boxes=True
    elif event.key()==Qt.Key_Right:
        self.cx += 1 + 10 * self.key_control
        update_search_boxes=True
    elif event.key()==Qt.Key_Up:
        self.cy -= 1 + 10 * self.key_control
        update_search_boxes=True
    elif event.key()==Qt.Key_Down:
        self.cy += 1 + 10 * self.key_control
        update_search_boxes=True
    else:
        print( "Uknown Key:", event.key() )

    if update_search_boxes:
        self.engine.make_searchboxes(self.cx,self.cy,pupil_radius_pixel,box_size_pixel)

        #if event.key() == QtCore.Qt.Key_Q:
        #elif event.key() == QtCore.Qt.Key_Enter:

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
