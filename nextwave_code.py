import numpy as np
import sys
import os
import time

import matplotlib.cm as cmap
#from numba import jit
from numpy.linalg import svd,lstsq

import mmap
import struct
import extract_memory

import zernike_integrals
import iterative

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

MEM_LEN=512
MEM_LEN_DATA=2048*2048*4

class ByteStream(bytearray):
    def append(self, v, fmt='B'):
        self.extend(struct.pack(fmt, v))

class NextwaveEngineComm():
    """ Class to manage:
          - Structures needed for realtime engine (boxes/refs, computed centroids, etc.)
          - Communication with the realtime engine (comm. over shared memory)
          - Computation of Zernikes, matrices for SVD, etc.
    """
    def __init__(self,ui):
        # TODO:
        self.ui = ui
        self.mode = 0

    def init(self):
        self.layout=extract_memory.get_header_format('memory_layout.h')
        self.layout_boxes=extract_memory.get_header_format('layout_boxes.h')

        # Could be math in the defines for sizes, use eval
        MEM_LEN=int( eval(self.layout[2]['SHMEM_HEADER_SIZE'] ) )
        MEM_LEN_DATA=int(eval(self.layout[2]['SHMEM_BUFFER_SIZE'] ) )
        MEM_LEN_BOXES=self.layout_boxes[0]
        if WINDOWS:
            # TODO: Get these all from the .h defines
            self.shmem_hdr=mmap.mmap(-1,MEM_LEN,"NW_SRC0_HDR")
            self.shmem_data=mmap.mmap(-1,MEM_LEN_DATA,"NW_SRC0_BUFFER")
            self.shmem_boxes=mmap.mmap(-1,MEM_LEN_BOXES,"NW_BUFFER_BOXES")

            #from multiprocessing import shared_memory
            #self.shmem_hdr = shared_memory.SharedMemory(name="NW_SRC0_HDR" ).buf
            #self.shmem_data = shared_memory.SharedMemory(name="NW_SRC0_BUFFER" ).buf
            #self.shmem_boxes = shared_memory.SharedMemory(name="NW_BUFFER2").buf
        else:
            fd1=os.open('/dev/shm/NW_SRC0_HDR', os.O_RDWR)
            self.shmem_hdr=mmap.mmap(fd1, MEM_LEN)
            fd2=os.open('/dev/shm/NW_SRC0_BUFFER', os.O_RDWR)
            self.shmem_data=mmap.mmap(fd2,MEM_LEN_DATA)
            fd3=os.open('/dev/shm/NW_BUFFER_BOXES', os.O_RDWR)
            self.shmem_boxes=mmap.mmap(fd3,MEM_LEN_BOXES)

        self.init_params() # MAYBE do this for defaults

    def init_params(self, overrides=None):
        self.ccd_pixel = self.ui.get_param("system","pixel_pitch",True)
        self.pupil_diam = self.ui.get_param("system","pupil_diam",True)
        self.box_um = self.ui.get_param("system","lenslet_pitch",True)
        self.focal = self.ui.get_param("system","focal_length",True)

        if overrides:
            self.pupil_diam=overrides['pupil_diam']

        self.pupil_radius_mm=self.pupil_diam / 2.0
        self.pupil_radius_pixel=self.pupil_radius_mm * 1000 / self.ccd_pixel
        self.box_size_pixel=self.box_um / self.ccd_pixel
        self.ri_ratio = self.pupil_radius_pixel / self.box_size_pixel
        print( "Init:", self.ri_ratio, self.box_size_pixel, self.pupil_radius_pixel )

        if overrides:
            bytez =np.array([self.num_boxes], dtype="uint16").tobytes() 
            fields = self.layout_boxes[1]
            self.shmem_boxes.seek(fields['num_boxes']['bytenum_current'])
            self.shmem_boxes.write(bytez)
            self.shmem_boxes.flush()
            print( self.num_boxes)
        #except:
            #pass

        bytez =np.array([self.ccd_pixel, self.box_um, self.pupil_radius_mm], dtype='double').tobytes() 
        fields = self.layout_boxes[1]
        self.shmem_boxes.seek(fields['pixel_um']['bytenum_current'])
        self.shmem_boxes.write(bytez)
        self.shmem_boxes.flush()

    def iterative_run(self, cx, cy, step):
        return

    def read_mode(self):
        self.shmem_hdr.seek(2) #TODO: get address
        buf=self.shmem_hdr.read(1)
        mode= struct.unpack('B',buf)[0]
        return mode

    def iterative_step(self, cx, cy, step, start, stop):
        if self.iterative_size>=stop:
            self.iterative_size = start
        elif self.iterative_size+step > stop:
            self.iterative_size = stop
        else:
            self.iterative_size += step

        self.make_searchboxes(cx,cy,pupil_radius_pixel=self.iterative_size/2.0*1000/self.ccd_pixel)
        #self.update_zernike_svd() # TODO: maybe integrate into make_sb
        #self.send_searchboxes(self.shmem_boxes, self.box_x, self.box_y, self.layout_boxes)

        mode=self.read_mode()
        print ( "MODE: %d"%mode, end='')
        self.init_params( {'pupil_diam': self.iterative_size})
        self.mode_snap(False)

        mode=self.read_mode()
        # TODO: don't wait forever; lokcup
        while( mode>1 ):
            mode=self.read_mode()
            #print ( "MODE: %d"%mode, end='')
            time.sleep(0.01)

        self.receive_centroids()
        self.compute_zernikes()
        zs = self.zernikes

        factor = self.iterative_size / (self.iterative_size+step)
        z_new =  zs #iterative.extrapolate_zernikes(zs, factor)
        #print( zs[5], zs[0:5] )
        #print( z_new[0:5] )
        self.shift_search_boxes(z_new,from_dialog=False)

    def set_iterative_size(self,value):
        self.iterative_size = value

    def move_searchboxes(self,dx,dy):
        self.box_x += dx
        self.box_y += dy

        self.update_zernike_svd() # Precompute
        #self.num_boxes= num_boxes

        self.send_searchboxes(self.shmem_boxes, self.box_x, self.box_y, self.layout_boxes)
        self.update_zernike_svd()


    def make_searchboxes(self,cx,cy,img_max=1000,aperture=1.0,pupil_radius_pixel=None):
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
        if pupil_radius_pixel is None:
            pupil_radius_pixel=self.pupil_radius_pixel
        box_size_pixel=self.box_size_pixel

        ri_ratio = pupil_radius_pixel / box_size_pixel

        print( pupil_radius_pixel, box_size_pixel, ri_ratio )

        aperture = ri_ratio * aperture
        print( "Make:", pupil_radius_pixel, box_size_pixel, ri_ratio)

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

        print("NUM BOX:", num_boxes)
        box_zero = np.where(valid_x_norm**2+valid_y_norm**2==0)[0] # Index of zeroth (middle) element

        MULT = ri_ratio * box_size_pixel
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
        buf.append(self.ccd_pixel,'d')
        buf.append(self.box_um, 'd')
        buf.append(self.pupil_radius_pixel*self.ccd_pixel, 'd')
        shmem_boxes.seek(0)
        shmem_boxes.write(buf)
        shmem_boxes.flush()

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
        lefts =  self.norm_x - 0.5/self.ri_ratio
        rights = self.norm_x + 0.5/self.ri_ratio
        ups =    -(self.norm_y + 0.5/self.ri_ratio)
        downs =  -(self.norm_y - 0.5/self.ri_ratio)

        # Compute all integrals for all box corners 
        lenslet_dx,lenslet_dy=zernike_integrals.zernike_integral_average_from_corners(
            lefts, rights, ups, downs, self.pupil_radius_mm)
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

        slope = np.concatenate( (spot_displace_y, spot_displace_x)) * self.ccd_pixel/self.focal;

        self.spot_displace_x = spot_displace_x
        self.spot_displace_y = spot_displace_y
        self.slope = slope

        coeff=np.matmul(self.zterms,slope)

        # TODO: only do this once
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

    def autoshift_searchboxes(self):
        #shift_search_boxes(self,zs,from_dialog=True):
        pass
        #return

    def calc_diopters(self):
        radius = self.pupil_radius_mm
        EPS=1e-10
        sqrt3=np.sqrt(3.0)
        sqrt6=np.sqrt(6.0)
        z3=self.zernikes[3-1]
        z4=self.zernikes[4-1]
        z5=self.zernikes[5-1]
        cylinder = (4.0 * sqrt6 / (radius * radius)) * np.sqrt((z3 * z3) + (z5 * z5))
        sphere = (-4.0 * sqrt3 * z4 / (radius * radius)) - 0.5 * cylinder

        if (np.abs(z5) <= EPS):
            thetaRad = 1.0 if (np.abs(z3) > EPS) else -1.0
            thetaRad *= float(np.pi) / 4.0
        else:
            thetaRad = 0.5 * np.arctan(z3 / z5)

        axis = thetaRad * 180.0 / float(np.pi)
        if (axis < 0.0):
            axis += 180.0

        rms=np.sqrt( np.nansum(self.zernikes[(3-1):]**2 ) )
        rms5p=np.sqrt( np.nansum(self.zernikes[(5-1):]**2 ) )

        return rms,rms5p,cylinder,sphere,axis

    def shift_search_boxes(self,zs,from_dialog=True):
        zern_new = np.zeros(NUM_ZERNIKES)
        #zern_new[self.OSA_to_CVS_map[0:NUM_ZERN_DIALOG]]=zs 

        #zern_new[0:NUM_ZERN_DIALOG]=zs 
        # TODO: What about term 0?
        if from_dialog:
            zern_new[self.CVS_to_OSA_map[0:NUM_ZERN_DIALOG-1]-START_ZC-1 ] = zs[1:]
        else:
            zern_new[self.CVS_to_OSA_map[0:zs.shape[0]]-START_ZC-1 ] = zs

        print( zern_new[0:9])
        #print(self.OSA_to_CVS_map)
        #print( zern_new[0:9] )

        delta=np.matmul(self.zterms_inv,zern_new) 
        num_boxes = self.box_x.shape[0] 
        self.box_y = self.initial_y + delta[0:num_boxes]
        self.box_x = self.initial_x - delta[num_boxes:]

        self.send_searchboxes(self.shmem_boxes, self.box_x, self.box_y, self.layout_boxes)
        self.update_zernike_svd()

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

    def receive_image(self):
        # TODO: Wait until it's safe (unlocked)

        self.fps0=extract_memory.get_array_item2(self.layout,self.shmem_hdr,'fps',0, True)/10.0
        self.fps1=extract_memory.get_array_item2(self.layout,self.shmem_hdr,'fps',1, True)/10.0
        self.fps2=extract_memory.get_array_item2(self.layout,self.shmem_hdr,'fps',2, True)/10.0

        self.height=extract_memory.get_array_item2(self.layout,self.shmem_hdr,'dimensions',0, False)
        self.width=extract_memory.get_array_item2(self.layout,self.shmem_hdr,'dimensions',1, False)

        self.shmem_hdr.seek(0)
        mem_header=self.shmem_hdr.read(MEM_LEN)

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

        DEBUGGING=False
        if DEBUGGING:
            print (self.num_boxes, np.min(self.centroids_x), np.max(self.centroids_x)  )
            for n in np.arange(self.num_boxes):
                if np.isnan(self.centroids_x[n]):
                    print( n, end=' ')
                    print (self.num_boxes, self.centroids_x[100], self.centroids_y[100]  )

    def send_quit(self):
        buf = ByteStream()
        buf.append(255) # TODO: MODE from file
        #buf.append(0)  # Lock
        #buf.append(1, 'H') # NUM BOXES. Hopefully doesn't matter
        #buf.append(40, 'd')
        #buf.append(500, 'd')
        self.shmem_hdr.seek(2) # TODO: get address
        self.shmem_hdr.write(buf)
        self.shmem_hdr.flush()

    def mode_init(self):
        self.mode=1
        self.mode_snap()

    def mode_snap(self, reinit=True):
        self.mode=2
        if reinit:
            self.init_params()

        buf = ByteStream()
        buf.append(2) # TODO: MODE_CENTROIDING
        self.shmem_hdr.seek(2) #TODO: get address
        self.shmem_hdr.write(buf)
        self.shmem_hdr.flush()

    def mode_run(self, reinit=True):
        self.mode=3
        if reinit:
            self.init_params()
        buf = ByteStream()
        buf.append(9) # TODO: Greater than MODE_CEN_ONE
        self.shmem_hdr.seek(2) #TODO: get address
        self.shmem_hdr.write(buf)
        self.shmem_hdr.flush()

    def mode_stop(self):
        buf = ByteStream()
        buf.append(255) # TODO: Back to ready
        self.shmem_hdr.seek(2) #TODO: get address
        self.shmem_hdr.write(buf)
        self.shmem_hdr.flush()
