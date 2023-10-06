from PyQt5.QtWidgets import QMainWindow, QLabel, QSizePolicy, QApplication
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
import numpy as np
import sys
import mmap

from numba import jit

HEIGHT=2048
WIDTH=2048

HEIGHT_WIN=1500
WIDTH_WIN=768

MEM_LEN=512
MEM_LEN_DATA=2048*2048*32

x=np.arange( WIDTH )
y=np.arange( HEIGHT )
X,Y=np.meshgrid(x,y)


boxes=[ [512,512] ]
boxsize=40 # Hard-code for now
nboxes=np.shape(boxes)[0]

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
    self.shmem_hdr=mmap.mmap(-1,MEM_LEN,"DC_SRC0_HDR")
    MEM_LEN_DATA=2048*2048*32
    self.shmem_data=mmap.mmap(-1,MEM_LEN_DATA,"DC_SRC0_DATA")

    self.setFixedSize(1024,800)
    self.move( 100,100 )

 def doit(self):
    buf=self.shmem_hdr.seek(0)
    buf=self.shmem_hdr.read(MEM_LEN)
#    print ('%02x'%buf[88],end=' ', flush=True);

    self.shmem_data.seek(0)
    im_buf=self.shmem_data.read(2048*2048)
    bytez =np.frombuffer(im_buf, dtype='uint8', count=2048*2048 )
    bytes2=np.reshape(bytez,( 2048,2048)).copy()

    bytesf = bytes2 / np.max(bytes2)

    weighted_x = X*np.array(bytesf,dtype='float')
    weighted_y = Y*np.array(bytesf,dtype='float')
    cen=find_centroids(boxes,bytesf,weighted_x,weighted_y,nboxes)
    print( cen )

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
    print ('%0.2f'%bytez.mean(),end=' ', flush=True);

 def initUI(self):
     print("OK")
     self.setGeometry(10,10,WIDTH_WIN,HEIGHT_WIN)

     pixmap_label = QLabel()
     pixmap_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
     pixmap_label.resize(WIDTH_WIN,HEIGHT_WIN)
     pixmap_label.setAlignment(Qt.AlignCenter)
     self.pixmap_label=pixmap_label

     im_np = np.ones((HEIGHT,WIDTH),dtype='uint8')
     #im_np = np.transpose(im_np, (1,0,2))
     qimage = QImage(im_np, im_np.shape[1], im_np.shape[0],
                     QImage.Format_Mono)
     pixmap = QPixmap(qimage)
     pixmap = pixmap.scaled(WIDTH_WIN,HEIGHT_WIN, Qt.KeepAspectRatio)
     pixmap_label.setPixmap(pixmap)

     self.setCentralWidget(self.pixmap_label)
     print('HI')
     self.show()
     print('there')

def main():
  app = QApplication(sys.argv)
  win = Test()
  sys.exit(app.exec_())

if __name__=="__main__":
  main()
