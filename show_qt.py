from PyQt5.QtWidgets import QMainWindow, QLabel, QSizePolicy, QApplication
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QTimer, QLineF, QPointF
import numpy as np
import sys
import os

import mmap
import struct

from numba import jit

import extract_memory

# TODO: read from butter
QIMAGE_HEIGHT=1000
QIMAGE_WIDTH=1000
box_size=40

HEIGHT_WIN=1500
WIDTH_WIN=768

MEM_LEN=512
MEM_LEN_DATA=2048*2048*4

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
    valid_x=XX[valid_boxes]/ri_ratio
    valid_y=YY[valid_boxes]/ri_ratio

    num_boxes=valid_x.shape[0]
    print( num_boxes )
    box_zero = np.where(valid_x**2+valid_y**2==0)[0]

    valid_x = valid_x * 500 + cx
    valid_y = valid_y * 500 + cy

    return valid_x,valid_y

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

class Test(QMainWindow):

 def __init__(self):
    super().__init__()

    # self.worker=Worker();
    #self.worker_thread=QThread()
    #self.worker.moveToThread(self.worker_thread);
    #self.worker_thread.start();

    self.updater = QTimer(self);
    #self.updater.setInterval(1000)
    self.updater.timeout.connect(self.doit)
    self.updater.start(50)
    #self.updater.start(1000);

    self.draw_refs = False
    self.draw_boxes = False
    self.draw_centroids = False


    MEM_LEN=512
    MEM_LEN_DATA=2048*2048*4 # TODO: Get from file
    WINDOWS=False
    self.layout=extract_memory.get_header_format('memory_layout.h')
    self.layout_boxes=extract_memory.get_header_format('layout_boxes.h')
    if WINDOWS:
        self.shmem_hdr=mmap.mmap(-1,MEM_LEN,"NW_SRC0_HDR")
        self.shmem_data=mmap.mmap(-1,MEM_LEN_DATA,"NW_SRC0_BUFFER")
    else:
        fd1=os.open('/dev/shm/NW_SRC0_HDR', os.O_RDWR)
        self.shmem_hdr=mmap.mmap(fd1, MEM_LEN)
        fd2=os.open('/dev/shm/NW_SRC0_BUFFER', os.O_RDWR)
        self.shmem_data=mmap.mmap(fd2,MEM_LEN_DATA)
        fd3=os.open('/dev/shm/NW_BUFFER2', os.O_RDWR)
        self.shmem_boxes=mmap.mmap(fd3,self.layout_boxes[0])

    box_x,box_y=make_initial_searchboxes(500,500)
    send_searchboxes(self.shmem_boxes, box_x, box_y, self.layout_boxes)

    self.setFixedSize(1024,800)
    self.move( 100,100 )
    self.x = 2048/2
    self.y = 2048/2

    self.initUI()

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

 def doit(self):
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

    if self.draw_refs:
        pen = QPen(Qt.red, 2)
        painter.setPen(pen)
        points_ref=[QPointF(self.ref_x[n],self.ref_y[n]) for n in np.arange(self.ref_x.shape[0])]
        painter.drawPoints(points_ref)

    if self.draw_boxes:
        pen = QPen(Qt.red, 2)
        painter.setPen(pen)
        boxes1=[QLineF(self.ref_x[n]-box_size//2, # top
                       self.ref_y[n]-box_size//2,
                       self.ref_x[n]+box_size//2,
                       self.ref_y[n]-box_size//2) for n in np.arange(self.ref_x.shape[0])]
        painter.drawLines(boxes1)
        boxes1=[QLineF(self.ref_x[n]-box_size//2, # bottom
                       self.ref_y[n]+box_size//2,
                       self.ref_x[n]+box_size//2,
                       self.ref_y[n]+box_size//2) for n in np.arange(self.ref_x.shape[0])]
        painter.drawLines(boxes1)
        boxes1=[QLineF(self.ref_x[n]-box_size//2, # left
                       self.ref_y[n]-box_size//2,
                       self.ref_x[n]-box_size//2,
                       self.ref_y[n]+box_size//2) for n in np.arange(self.ref_x.shape[0])]
        painter.drawLines(boxes1)
        boxes1=[QLineF(self.ref_x[n]+box_size//2, # right
                       self.ref_y[n]-box_size//2,
                       self.ref_x[n]+box_size//2,
                       self.ref_y[n]+box_size//2) for n in np.arange(self.ref_x.shape[0])]
        painter.drawLines(boxes1)

    # Centroids:
    if self.draw_centroids:
        num_boxes=437 # TODO
        fields=self.layout_boxes[1]
        self.shmem_boxes.seek(fields['centroid_x']['bytenum_current'])
        buf=self.shmem_boxes.read(num_boxes*4)
        centroids_x=struct.unpack_from(''.join((['f']*num_boxes)), buf)

        self.shmem_boxes.seek(fields['centroid_y']['bytenum_current'])
        buf=self.shmem_boxes.read(num_boxes*4)
        centroids_y=struct.unpack_from(''.join((['f']*num_boxes)), buf)

        pen = QPen(Qt.blue, 2)
        painter.setPen(pen)
        points_centroids=[QPointF(centroids_x[n],centroids_y[n]) for n in np.arange(num_boxes)]
        painter.drawPoints(points_centroids)

    im_buf=self.shmem_data.read(width*height)
    bytez =np.frombuffer(im_buf, dtype='uint8', count=width*height )


    #ql1=[QLineF(100,100,150,150)]
    #painter.drawLines(ql1)
    painter.end()

    pixmap = pixmap.scaled(WIDTH_WIN,HEIGHT_WIN, Qt.KeepAspectRatio)
    self.pixmap_label.setPixmap(pixmap)
    #print ('%0.2f'%bytez.mean(),end=' ', flush=True);

 def initUI(self):
     self.setGeometry(10,10,WIDTH_WIN,HEIGHT_WIN)

     pixmap_label = QLabel()
     pixmap_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
     pixmap_label.resize(WIDTH_WIN,HEIGHT_WIN)
     pixmap_label.setAlignment(Qt.AlignCenter)
     self.pixmap_label=pixmap_label

     im_np = np.ones((QIMAGE_HEIGHT,QIMAGE_WIDTH),dtype='uint8')
     #im_np = np.transpose(im_np, (1,0,2))
     qimage = QImage(im_np, im_np.shape[1], im_np.shape[0],
                     QImage.Format_Mono)
     pixmap = QPixmap(qimage)
     pixmap = pixmap.scaled(WIDTH_WIN,HEIGHT_WIN, Qt.KeepAspectRatio)
     pixmap_label.setPixmap(pixmap)

     pixmap_label.mousePressEvent = self.butt

     self.setCentralWidget(self.pixmap_label)
     self.show()

 def butt(self, event):
    print("clicked:", event.pos() )
    self.x = event.pos().x()
    self.y = event.pos().y()
    top_left = (130,18)
    bottom_right = (895,781)
    self.x = (self.x - top_left[0])/(bottom_right[0]-top_left[0])*1000 # TODO: img_siz
    self.y = (self.y - top_left[1])/(bottom_right[1]-top_left[1])*1000
    print(self.x, self.y)

    box_x,box_y=make_initial_searchboxes(500,500)
    send_searchboxes(self.shmem_boxes, box_x, box_y, self.layout_boxes)
    self.ref_x = box_x
    self.ref_y = box_y

 def keyPressEvent(self, event):
     if event.key()==ord('R'):
         self.draw_refs = not( self.draw_refs )
     if event.key()==ord('B'):
         self.draw_boxes = not( self.draw_boxes )
     if event.key()==ord('C'):
         self.draw_centroids = not( self.draw_centroids )
     if event.key()==ord('Q'):
            # Write header last, so the engine knows when we are ready
            buf = ByteStream()
            buf.append(1)
            buf.append(2) # Quitter
            buf.append(437, 'H')
            buf.append(40, 'd')
            buf.append(500, 'd')
            self.shmem_boxes.seek(0)
            self.shmem_boxes.write(buf)
            self.shmem_boxes.flush()



        #if event.key() == QtCore.Qt.Key_Q:
        #elif event.key() == QtCore.Qt.Key_Enter:

def main():
  app = QApplication(sys.argv)
  win = Test()
  sys.exit(app.exec_())

if __name__=="__main__":
  main()
