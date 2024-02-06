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

'''
Zernike order, first mode in that order:
0 0
1 1
2 3
3 6
4 10
5 15
6 21
7 28
8 36
9 45
10 55
11 66
'''

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

# TODO
PUPIL=6.4/2.0
PUPIL_RADIUS_MM=PUPIL
pupil_radius_pixel=PUPIL_RADIUS_MM*1000/CCD_PIXEL
RI_RATIO=pupil_radius_pixel/box_size_pixel
print(pupil_radius_pixel)
FOCAL=5.9041

ri_ratio = pupil_radius_pixel / box_size_pixel
print(ri_ratio)

class ByteStream(bytearray):
    def append(self, v, fmt='B'):
        self.extend(struct.pack(fmt, v))

class NextwaveEngineComm():
    """ Class to manage:
          - Structures needed for realtime engine (boxes/refs, computed centroids, etc.)
          - Communication with the realtime engine (comm. over shared memory)
          - Computation of Zernikes, matrices for SVD, etc.
    """
    def __init__(self):
        # TODO:
        self.pupil_radius_pixel=pupil_radius_pixel
        self.box_size_pixel=box_size_pixel

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
