from PyQt5.QtWidgets import (QMainWindow, QLabel, QSizePolicy, QApplication, QPushButton,
                             QHBoxLayout, QVBoxLayout, QWidget, QGroupBox, QTabWidget, QTextEdit,
                             QFileDialog)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QTimer, QLineF, QPointF
import numpy as np
import sys
import os

import mmap
import struct

from numba import jit

import extract_memory

import zernike_integrals
from numpy.linalg import svd,lstsq

WINDOWS=False

def showdialog():
   d = QDialog()
   b1 = QPushButton("ok",d)
   b1.move(50,50)
   d.setWindowTitle("Dialog")
   d.setWindowModality(Qt.ApplicationModal)
   d.exec_()

# TODO: read from butter
QIMAGE_HEIGHT=1000
QIMAGE_WIDTH=1000
box_size=40

MAIN_HEIGHT_WIN=1000
MAIN_WIDTH_WIN=1800
SPOTS_HEIGHT_WIN=1024
SPOTS_WIDTH_WIN=1024

MEM_LEN=512
MEM_LEN_DATA=2048*2048*4

UI_UPDATE_RATE_MS=50

# TODO
PUPIL=6.4
RI_RATIO=12.5
START_ZC=1
NUM_ZCS=68
CCD_PIXEL=6.4
FOCAL=5.9041

x=np.arange( QIMAGE_WIDTH )
y=np.arange( QIMAGE_HEIGHT )
X,Y=np.meshgrid(x,y)

def initial_searchboxes(ri_ratio,limit):
    b_x=[]
    b_y=[]
    count=0
    for y in np.arange(limit, -limit-1, -1):
        for x in np.arange(-limit, limit+1):
            yy = y + 0.5 * np.sign(y)
            xx = x + 0.5 * np.sign(x)
            rad = np.sqrt(xx*xx + yy*yy)
            if rad <= ri_ratio:
                count=count+1
                b_x += [x / ri_ratio]
                b_y += [y / ri_ratio]

    sb = np.vstack( (b_x,b_y)).T
    return sb

def make_initial_searchboxes(cx,cy,pupil_radius_pixel=500,box_size_pixel=40):
    # Like the MATLAB code, but doesn't use loops,
    # instead builds and filters arrays

    # How many boxwidths in the pupil
    ri_ratio = pupil_radius_pixel / box_size_pixel

    # The max number of boxes possible + or -
    max_boxes = np.ceil( pupil_radius_pixel/ box_size_pixel )

    # All possible bilinear box
    boxes_x = np.arange(-max_boxes,max_boxes+1) # +1 to include +max_boxes number
    boxes_y = np.arange(-max_boxes,max_boxes+1)

    # Determine outer edge of each box using corners away from the center:
    # 0.5*sign: positive adds 0.5, negative substracts 0.5
    XX,YY = np.meshgrid(boxes_x, boxes_y )

    RR = np.sqrt( (XX+0.5*np.sign(XX))**2 + (YY+0.5*np.sign(YY))**2 )
    valid_boxes = np.where(RR<ri_ratio)
    max_dist_boxwidths = np.max(RR[RR<ri_ratio])

    # Normalize to range -1 .. 1 (vs. pupil size)
    valid_x_norm=XX[valid_boxes]/ri_ratio
    valid_y_norm=YY[valid_boxes]/ri_ratio

    num_boxes=valid_x_norm.shape[0]
    box_zero = np.where(valid_x_norm**2+valid_y_norm**2==0)[0] # Index of zeroth (middle) element

    valid_x = valid_x_norm * 500 + cx
    valid_y = valid_y_norm * 500 + cy

    return valid_x,valid_y,valid_x_norm,valid_y_norm

class ByteStream(bytearray):
    def append(self, v, fmt='>B'):
        self.extend(struct.pack(fmt, v))

def send_searchboxes(shmem_boxes, valid_x, valid_y, layout_boxes):
    defs=layout_boxes[2]
    fields=layout_boxes[1]
    NUM_BOXES=defs['MAX_BOXES']

    num_boxes=valid_x.shape[0]
    box_size_pixel=box_size
    pupil_radius_pixel=500

    buf = ByteStream()
    for item in valid_x:
        buf.append(item, 'f')
    shmem_boxes.seek(fields['reference_x']['bytenum_current'])
    shmem_boxes.write(buf)
    shmem_boxes.flush()

    buf = ByteStream()
    for item in valid_y:
        buf.append(item, 'f')
    shmem_boxes.seek(fields['reference_y']['bytenum_current'])
    shmem_boxes.write(buf)
    shmem_boxes.flush()

    # Write header last, so the engine knows when we are ready
    buf = ByteStream()
    buf.append(1)
    buf.append(0)
    buf.append(num_boxes, 'H')
    buf.append(box_size_pixel, 'd')
    buf.append(pupil_radius_pixel, 'd')
    shmem_boxes.seek(0)
    shmem_boxes.write(buf)
    shmem_boxes.flush()
    #print(num_boxes)

boxes=[ [512,512] ]
boxsize=40 # Hard-code for now
nboxes=np.shape(boxes)[0]
# im_ratio = ri_ratio * search_box_size_pixels;
# pPupilRadius_um = pupilDiameter_um/2;
# pInterLensletDistance_um = searchBoxSize_um;
# ri_ratio = pPupilRadius_um/pInterLensletDistance_um;
# search_box_size_pixels=searchBoxSize_um/ccd_pixel;
ccd_pixel=6.4;
focal=5.9041;
pupilDiameter_um=7168;
searchBoxSize_um=256;

pInterLensletDistance_um = searchBoxSize_um;
num_pix_lenslet=searchBoxSize_um/ccd_pixel; 

ri_ratio = pupilDiameter_um/2.0/pInterLensletDistance_um;
MAXCOLS = np.ceil(pupilDiameter_um/searchBoxSize_um/2)*2+2;
limit = MAXCOLS/2;
searchBoxes=initial_searchboxes(ri_ratio,limit);
references=searchBoxes; # Initially, searchBoxes and refere

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

class NextWaveMainWindow(QMainWindow):

 def __init__(self):
    super().__init__(parent=None)

    # self.worker=Worker();
    #self.worker_thread=QThread()
    #self.worker.moveToThread(self.worker_thread);
    #self.worker_thread.start();

    self.updater = QTimer(self);
    #self.updater.setInterval(1000)
    self.updater.timeout.connect(self.update_ui)
    self.updater.start(UI_UPDATE_RATE_MS)

    self.draw_refs = True
    self.draw_boxes = True
    self.draw_centroids = True
    self.draw_arrows = False
    self.draw_crosshair = True

    MEM_LEN=512
    MEM_LEN_DATA=2048*2048*4 # TODO: Get from file

    self.layout=extract_memory.get_header_format('memory_layout.h')
    self.layout_boxes=extract_memory.get_header_format('layout_boxes.h')
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

    self.cx=500
    self.cy=500
    box_x,box_y,box_norm_x,box_norm_y=make_initial_searchboxes(self.cx,self.cy)
    send_searchboxes(self.shmem_boxes, box_x, box_y, self.layout_boxes)
    self.ref_x = box_x
    self.ref_y = box_y
    self.norm_x = box_norm_x
    self.norm_y = box_norm_y
    self.update_zernike_svd() # Precompute

    #self.setFixedSize(1024,800)
    #self.move( 100,100 )
    self.x = 2048/2
    self.y = 2048/2


 def make_boxes():
    # Recompute searchbox and reference info each time based on center
    # For y, all pixel numbers are increasing from top to bottom (array order)
    self.searchBoxes_pixel_coord[:,0]=(   searchBoxes[:,0]*im_ratio) + self.cx;
    self.searchBoxes_pixel_coord[:,1]=( -(searchBoxes[:,1]*im_ratio ) ) + self.cy;
    self.upperleftx=round( self.searchBoxes_pixel_coord[:,0]-search_box_size_pixels/2.0 );
    self.upperlefty=round( self.searchBoxes_pixel_coord[:,1]-search_box_size_pixels/2.0 );

    self.center_pixel_coord=[self.cx, self.cy];
    #center_offset=[imsize_x/2,imsize_y/2]-center_pixel_coord;

    # Offset the references by the chosen center
    self.references_pixel_coord[:,0]=  (references[:,0]*im_ratio ) + self.cx;
    self.references_pixel_coord[:,1]= -(references[:,1]*im_ratio ) + self.cy;

 def update_zernike_svd(self):
    lefts =  self.norm_x - 0.5/RI_RATIO
    rights = self.norm_x + 0.5/RI_RATIO
    ups =    -(self.norm_y + 0.5/RI_RATIO)
    downs =  -(self.norm_y - 0.5/RI_RATIO)

    # Compute all integrals for all box corners 
    lenslet_dx,lenslet_dy=zernike_integrals.zernike_integral_average_from_corners(
        lefts, rights, ups, downs, PUPIL)
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

 def compute_zernikes(self):
    # find slope
    spot_displace_x = self.ref_x - self.centroids_x
    spot_displace_y = -(self.ref_y - self.centroids_y)
    slope = np.concatenate( (spot_displace_y, spot_displace_x)) * CCD_PIXEL/FOCAL;
    self.spot_displace_x = spot_displace_x
    self.spot_displace_y = spot_displace_y
    self.slope = slope
    #print( coeff)

    coeff=np.matmul(self.zterms,slope)
    # Copied from MATLAB code
    CVS_to_OSA_map = np.array([3,2, 5,4,6, 9,7,8,10, 15,13,11,12,14,
                            21,19,17,16,18,20,
                            27,25,23,22,24,26,28, 35,33,31,29,30,32,34,36,
                            45,43,41,39,37,38,40,42,44,
                            55,53,51,49,47,46,48,50,52,54,
                            65,63,61,59,57,56,58,60,62,64,66,67,68,69,70])

    self.zernikes=coeff[CVS_to_OSA_map[START_ZC-1:NUM_ZCS-1]-START_ZC-1 ]

 def update_ui(self):
    # TODO: Wait until it's safe (unlocked)
    mem_header=self.shmem_hdr.seek(0)
    mem_header=self.shmem_hdr.read(MEM_LEN)
    height=extract_memory.get_array_item(self.layout,mem_header,'dimensions',0)
    width=extract_memory.get_array_item(self.layout,mem_header,'dimensions',1)
    #print ('%dx%d'%(height,width),end=' ', flush=True);
    #print( type(height) )

    self.shmem_data.seek(0)
    im_buf=self.shmem_data.read(width*height)
    bytez =np.frombuffer(im_buf, dtype='uint8', count=width*height )
    bytes2=np.reshape(bytez,( height,width)).copy()
    #bytes2 = bytes2.T.copy()

    bytesf = bytes2 / np.max(bytes2)

    if False:
        weighted_x = X*np.array(bytesf,dtype='float')
        weighted_y = Y*np.array(bytesf,dtype='float')
        cen=find_centroids(boxes,bytesf,weighted_x,weighted_y,nboxes)
        #print( cen )

        cx=int(cen[0][0])
        cy=int(cen[0][1])
        bytes2[(cx-10):(cx+10),(cx-12):(cx-10)] = 255
        bytes2[(cx-10):(cx+10),(cx+10):(cx+12)] = 255
        bytes2[(cx-12):(cx-10),(cx-10):(cx+10)] = 255
        bytes2[(cx+10):(cx+12),(cx-10):(cx+10)] = 255

    qimage = QImage(bytes2, bytes2.shape[1], bytes2.shape[0],
                 QImage.Format_Grayscale8)
    #qimage = QImage(bytes2, bytes2.shape[1], bytes2.shape[0],
                    #QImage.Format_RGB32)
    pixmap = QPixmap(qimage)

    painter = QPainter()
    painter.begin(pixmap)

    if self.draw_arrows:
        #conicalGradient gradient;
		#gradient.setCenter(rect().center());
		#gradient.setAngle(90);
		#gradient.setColorAt(1.0, Qt::black);
		#gradient.setColorAt(0.0, palette().background().color());

        num_boxes=437 # TODO
        fields=self.layout_boxes[1]
        self.shmem_boxes.seek(fields['centroid_x']['bytenum_current'])
        buf=self.shmem_boxes.read(num_boxes*4)
        centroids_x=struct.unpack_from(''.join((['f']*num_boxes)), buf)

        self.shmem_boxes.seek(fields['centroid_y']['bytenum_current'])
        buf=self.shmem_boxes.read(num_boxes*4)
        centroids_y=struct.unpack_from(''.join((['f']*num_boxes)), buf)

        pen = QPen(Qt.green, 2)
        painter.setPen(pen)
        arrows=[QLineF(self.ref_x[n],
                       self.ref_y[n],
                       centroids_x[n],
                       centroids_y[n]) for n in np.arange(0,num_boxes)]
        painter.drawLines(arrows)

    if self.draw_refs:
        pen = QPen(Qt.red, 2)
        painter.setPen(pen)
        points_ref=[QPointF(self.ref_x[n],self.ref_y[n]) for n in np.arange(self.ref_x.shape[0])]
        painter.drawPoints(points_ref)

    if self.draw_boxes:
        pen = QPen(Qt.blue, 1, Qt.DotLine)
        painter.setPen(pen)
        BOX_BORDER=2
        boxes1=[QLineF(self.ref_x[n]-box_size//2+BOX_BORDER, # top
                       self.ref_y[n]-box_size//2+BOX_BORDER,
                       self.ref_x[n]+box_size//2-BOX_BORDER,
                       self.ref_y[n]-box_size//2+BOX_BORDER) for n in np.arange(self.ref_x.shape[0])]


        painter.drawLines(boxes1)
        boxes1=[QLineF(self.ref_x[n]-box_size//2+BOX_BORDER, # bottom
                       self.ref_y[n]+box_size//2-BOX_BORDER,
                       self.ref_x[n]+box_size//2-BOX_BORDER,
                       self.ref_y[n]+box_size//2-BOX_BORDER) for n in np.arange(self.ref_x.shape[0])]
        painter.drawLines(boxes1)
        boxes1=[QLineF(self.ref_x[n]-box_size//2+BOX_BORDER, # left
                       self.ref_y[n]-box_size//2+BOX_BORDER,
                       self.ref_x[n]-box_size//2+BOX_BORDER,
                       self.ref_y[n]+box_size//2-BOX_BORDER) for n in np.arange(self.ref_x.shape[0])]
        painter.drawLines(boxes1)
        boxes1=[QLineF(self.ref_x[n]+box_size//2-BOX_BORDER, # right
                       self.ref_y[n]-box_size//2+BOX_BORDER,
                       self.ref_x[n]+box_size//2-BOX_BORDER,
                       self.ref_y[n]+box_size//2-BOX_BORDER) for n in np.arange(self.ref_x.shape[0])]
        painter.drawLines(boxes1)

    # Centroids:
    if self.draw_centroids:
        num_boxes=437 # TODO
        fields=self.layout_boxes[1]
        self.shmem_boxes.seek(fields['centroid_x']['bytenum_current'])
        buf=self.shmem_boxes.read(num_boxes*4)
        self.centroids_x=struct.unpack_from(''.join((['f']*num_boxes)), buf)

        self.shmem_boxes.seek(fields['centroid_y']['bytenum_current'])
        buf=self.shmem_boxes.read(num_boxes*4)
        self.centroids_y=struct.unpack_from(''.join((['f']*num_boxes)), buf)

        pen = QPen(Qt.blue, 2)
        painter.setPen(pen)
        points_centroids=[QPointF(self.centroids_x[n],self.centroids_y[n]) for n in np.arange(num_boxes)]
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

    im_buf=self.shmem_data.read(width*height)
    bytez =np.frombuffer(im_buf, dtype='uint8', count=width*height )

    #ql1=[QLineF(100,100,150,150)]
    #painter.drawLines(ql1)
    painter.end()

    pixmap = pixmap.scaled(SPOTS_WIDTH_WIN,SPOTS_HEIGHT_WIN, Qt.KeepAspectRatio)
    self.pixmap_label.setPixmap(pixmap)
    #print ('%0.2f'%bytez.mean(),end=' ', flush=True);

    self.compute_zernikes()
    s=""
    for n in np.arange(13):
        s += 'Z%2d=%+0.4f\n'%(n+1,self.zernikes[n])
    self.text_status.setText(s)

 def initUI(self):

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

     self.widget_displays = QWidget()
     layout=QVBoxLayout()
     layout.addWidget(QGroupBox('Pupil'))
     layout.addWidget(QGroupBox('AO'))
     layout.addWidget(QGroupBox('Wavefront'))
     layout.addWidget(QGroupBox('PSF'))
     self.widget_displays.setLayout(layout)

     self.widget_controls = QWidget()
     layout=QVBoxLayout()
     tabs = QTabWidget()
     tabs.setTabPosition(QTabWidget.North)
     tabs.setMovable(True)
     for n, tabnames in enumerate(["Source", "Options", "Coords", "Boxes","Camera"]):
         tabs.addTab(QTabWidget(), tabnames)
     #self.setCentralWidget(tabs)

     self.text_status = QTextEdit()
     self.text_status.setReadOnly(True)
     layout.addWidget(self.text_status)

     layout.addWidget(tabs)
     #self.widget_controls = QGroupBox('Controls')
     self.widget_controls.setLayout(layout)

     layout.addWidget(QGroupBox('Statistics'))

     # Main Widget
     self.widget_main = QWidget()
     layoutCentral = QHBoxLayout()
     layoutCentral.addWidget(self.pixmap_label, stretch=3)
     layoutCentral.addWidget(self.widget_displays)
     layoutCentral.addWidget(self.widget_controls, stretch=1)
     self.widget_main.setLayout(layoutCentral)

     self.setCentralWidget(self.widget_main)

     self.setWindowTitle('NextWave')
     menu=self.menuBar().addMenu('&File')
     menu.addAction('&Export Centroids & Zernikes', self.export)
     menu.addAction('e&Xit', self.close)
     self.setGeometry(2,2,MAIN_WIDTH_WIN,MAIN_HEIGHT_WIN)
     self.show()

 def butt(self, event):
    #print("clicked:", event.pos() )
    self.x = event.pos().x()
    self.y = event.pos().y()
    top_left = (130,18)
    bottom_right = (895,781)
    self.x = (self.x - top_left[0])/(bottom_right[0]-top_left[0])*1000 # TODO: img_siz
    self.y = (self.y - top_left[1])/(bottom_right[1]-top_left[1])*1000
    #print(self.x, self.y)

    #box_x,box_y=make_initial_searchboxes(500,500)
    box_x,box_y,box_norm_x,box_norm_y=make_initial_searchboxes(500,500)
    send_searchboxes(self.shmem_boxes, box_x, box_y, self.layout_boxes)
    self.ref_x = box_x
    self.ref_y = box_y
    self.norm_x = box_norm_x
    self.norm_y = box_norm_y
    self.update_zernike_svd()

 def keyPressEvent(self, event):
     if event.key()==ord('A'):
         self.draw_arrows = not( self.draw_arrows )
     if event.key()==ord('R'):
         self.draw_refs = not( self.draw_refs )
     if event.key()==ord('B'):
         self.draw_boxes = not( self.draw_boxes )
     if event.key()==ord('C'):
         self.draw_centroids = not( self.draw_centroids )
     if event.key()==ord('X'):
         self.draw_crosshair = not( self.draw_crosshair )
     if event.key()==ord('Q'):
         self.close()

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
        for nbox in np.arange( len(self.ref_x)):
            # TODO: Valid or invalid
            fil.writelines('%d\t%f\t%f\t%f\t%f\t\n'%(1,self.ref_x[nbox], self.ref_y[nbox], self.centroids_x[nbox], self.centroids_y[nbox] ) )
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

 def close(self):
    buf = ByteStream()
    buf.append(2)
    buf.append(2) # Quitter
    buf.append(437, 'H')
    buf.append(40, 'd')
    buf.append(500, 'd')
    self.shmem_boxes.seek(0)
    self.shmem_boxes.write(buf)
    self.shmem_boxes.flush()
    self.app.exit()

def main():
  app = QApplication(sys.argv)
  win = NextWaveMainWindow()
  win.app = app
  win.initUI()
  sys.exit(app.exec_())

if __name__=="__main__":
  main()
