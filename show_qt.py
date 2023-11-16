from PyQt5.QtWidgets import QMainWindow, QLabel, QSizePolicy, QApplication
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
import numpy as np
import sys
import mmap

from numba import jit

import extract_memory

QIMAGE_HEIGHT=2048
QIMAGE_WIDTH=2048

HEIGHT_WIN=1500
WIDTH_WIN=768

MEM_LEN=512
MEM_LEN_DATA=2048*2048*32

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

    self.initUI() 

    MEM_LEN=512
    self.shmem_hdr=mmap.mmap(-1,MEM_LEN,"NW_SRC0_HDR")
    MEM_LEN_DATA=2048*2048*32
    self.shmem_data=mmap.mmap(-1,MEM_LEN_DATA,"NW_SRC0_BUFFER")

    self.setFixedSize(1024,800)
    self.move( 100,100 )
    
    self.layout=extract_memory.get_header_format('memory_layout.h')    
    
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

    if True:
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
    pixmap = QPixmap(qimage)
    pixmap = pixmap.scaled(WIDTH_WIN,HEIGHT_WIN, Qt.KeepAspectRatio)
    self.pixmap_label.setPixmap(pixmap)
    #print ('%0.2f'%bytez.mean(),end=' ', flush=True);

 def initUI(self):
     print("OK")
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

     #pixmap_label.setMouseTracking(True)


     self.setCentralWidget(self.pixmap_label)
     print('HI')
     self.show()
     print('there')
     

 def butt(self, event):
    print("clicked")
    print(event.pos().x() )   
    self.x = event.pos.x()
    self.y = event.pos.y()

def main():
  app = QApplication(sys.argv)
  win = Test()
  sys.exit(app.exec_())

if __name__=="__main__":
  main()
