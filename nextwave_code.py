import numpy as np
import sys
import os
import time

import matplotlib.cm as cmap
#from numba import jit
from numpy.linalg import svd,lstsq
import scipy
from scipy.optimize import minimize

import numpy.random as random
from scipy.ndimage import gaussian_filter

import mmap
import struct
import extract_memory

import zernike_integrals
import iterative

import ffmpegcv # Read AVI... Better than OpenCV (built-in ffmpeg?)

WINDOWS=(os.name == 'nt')

NUM_ZERNIKES=67 # TODO
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

class OpticsParams():
    def __init__(self,ccd_pixel,pupil_diam,box_um,focal):
        self.ccd_pixel = ccd_pixel
        self.pupil_diam = pupil_diam
        self.box_um = box_um
        self.focal = focal

        # Computed:
        self.pupil_radius_mm=self.pupil_diam / 2.0
        self.pupil_radius_pixel=self.pupil_radius_mm * 1000 / self.ccd_pixel
        self.box_size_pixel=self.box_um / self.ccd_pixel
        self.ri_ratio = self.pupil_radius_pixel / self.box_size_pixel
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

        self.init_params()

    def init_params(self, overrides=None):
        #self.ccd_pixel = self.ui.get_param("system","pixel_pitch",True)
        #self.pupil_diam = self.ui.get_param("system","pupil_diam",True)
        #self.box_um = self.ui.get_param("system","lenslet_pitch",True)
        #self.focal = self.ui.get_param("system","focal_length",True)
        self.focal =     self.ui.get_param_xml("LENSLETS_LensletFocalLength")/1000.0
        self.box_um =    self.ui.get_param_xml("LENSLETS_LensletPitch")
        self.ccd_pixel = self.ui.get_param_xml("CAMERA1_CameraPixelPitch")
        self.pupil_diam =self.ui.get_param_xml("OPTICS_PupilDiameter")

        # New method, not used much yet:
        self.params = OpticsParams(self.ccd_pixel, self.pupil_diam, self.box_um, self.focal)

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

    def circle(self,cx,cy,rad):
        #X,Y=np.meshgrid( np.arange(self.desired.shape[1]), np.arange(self.desired.shape[0]) )
        X=self.box_x
        Y=self.box_y
        r=np.sqrt((X-cx-1.0*np.sign(X))**2+(cy-Y+1.0*np.sign(Y))**2) # 0.5 fudge
        result=(r<rad)*1.0
        return result

        #downs =  -(self.norm_y - 0.5/self.ri_ratio)

    def circle_err(self,p):
        ssq=np.sum( (self.circle(*p)-self.desired) **2 )
        print(p)
        return ssq

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

        #while self.iterative_size<9:
        if True:
            #self.iterative_size += step
            #print(self.iterative_size)

            self.make_searchboxes(cx,cy,pupil_radius_pixel=self.iterative_size/2.0*1000/self.ccd_pixel)
            #self.update_zernike_svd() # TODO: maybe integrate into make_sb
            #self.send_searchboxes(self.shmem_boxes, self.box_x, self.box_y, self.layout_boxes)
            if self.ui.mode_offline:
                self.iterative_offline()
                return # Don't get boxes from engine

            self.init_params( {'pupil_diam': self.iterative_size})
            self.mode_snap(False,False)
            mode=self.read_mode()
            # TODO: don't wait forever; lokcup
            while( mode>1 ):
                mode=self.read_mode()
                time.sleep(0.005)

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
        self.ref_x += dx
        self.ref_y += dy

        #print( self.ref_x[0], self.ui.cx )

        #self.update_searchboxes()

    def make_searchboxes(self,cx=None,cy=None,aperture=1.0,pupil_radius_pixel=None):
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
        if cx is None:
            cx=self.ui.cx
        if cy is None:
            cy=self.ui.cy

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
        self.ref_x = valid_x.copy()
        self.ref_y = valid_y.copy()
        self.norm_x = valid_x_norm
        self.norm_y = valid_y_norm
        self.initial_x = valid_x.copy()
        self.initial_y = valid_y.copy()
        self.update_zernike_svd() # Precompute

        self.num_boxes= num_boxes

        self.centroids_x = np.zeros(num_boxes) + np.nan
        self.centroids_y = np.zeros(num_boxes) + np.nan

        # Determine neighbors (for nan interpolation)
        self.neighbors = np.zeros( (self.num_boxes, 4), dtype='int32')
        for nidx in np.arange(num_boxes):
            distances = (self.box_x - self.box_x[nidx])**2 + (self.box_y - self.box_y[nidx])**2
            self.neighbors[nidx]=np.argsort( distances)[1:5] # Take 4 nearest, excluding self (which will be 0)

        self.update_searchboxes()

        return self.ref_x,self.ref_y,self.norm_x,self.norm_y

    def dump_vars(self):
        fil=open("dump_vars.py","wt")
        fil.writelines("from numpy import array\n")
        fil.writelines("from numpy import uint8,int32,nan\n")
        for var1 in dir(self):
            if var1[0:2]=="__":
                continue
            else:
                #s="%s=%s\n"%(var1,eval("self.%s"%var1) )
                s="%s=%s\n"%(var1,eval("self.%s.__repr__()"%var1) )
                if not ("=<" in s) and not ("..." in s):
                    fil.writelines(s)
        fil.close()

    def update_searchboxes(self):
        self.update_zernike_svd()
        self.update_influence();
        self.send_searchboxes(self.shmem_boxes, self.box_x, self.box_y, self.layout_boxes)
        self.dump_vars()

    def rcv_searchboxes(self,shmem_boxes, layout, box_x, box_y, layout_boxes):
        fields=layout[1]

        adr=fields['box_x']['bytenum_current']
        shmem_boxes.seek(adr)
        box_buf=shmem_boxes.read(NUM_BOXES*4) # TODO: ALL WRONG FOR DOUBLES
        box_x = [struct.unpack('f',box_buf[n*4:n*4+4]) for n in np.arange(NUM_BOXES)]
        #box_x =np.frombuffer(box_buf, dtype='uint8', count=NUM_BOXES )
        #print(box_x[0], box_x[1], box_x[2], box_x[3])

        adr=fields['box_y']['bytenum_current']
        shmem_boxes.seek(adr)
        box_buf=shmem_boxes.read(NUM_BOXES*4)
        box_y = [struct.unpack('f',box_buf[n*4:n*4+4]) for n in np.arange(NUM_BOXES)]
        #box_x =np.frombuffer(box_buf, dtype='uint8', count=NUM_BOXES )

        return box_x,box_y

    def offline_frame(self,nframe):
            fields=self.layout[1] # TODO: fix
            dims=np.zeros(2,dtype='uint16')
            dims[0]=self.offline_movie[nframe].shape[0]
            dims[1]=self.offline_movie[nframe].shape[1]
            self.shmem_hdr.seek(fields['dimensions']['bytenum_current']) #TODO: nicer
            self.shmem_hdr.write(dims)
            self.shmem_hdr.flush()

            for nbuf in np.arange(4):
                self.shmem_data.seek(nbuf*2048*2048)
                self.shmem_data.write(self.offline_movie[nframe])
                self.shmem_data.flush()

    def load_offline_background(self,file_info):
        # file_info: from dialog. Tuple: (list of files, file types)
        if '.bin' in file_info[1]:
            pass # TODO
        elif '.avi' in file_info[1]:
            fname=file_info[0][0]
            print("Offline movie: ",fname)

            vidin = ffmpegcv.VideoCapture(fname)
            buf_movie=None

            with vidin:
                for nf,frame in enumerate(vidin):
                    f1=frame.mean(2)[0:1024,0:1024] # Avg RGB. TODO: crop hard-code
                    if buf_movie is None:
                        buf_movie=np.zeros( (50,f1.shape[0],f1.shape[1]), dtype='uint8') # TODO: grow new chunk if necessary
                    buf_movie[nf]=f1
                    print(nf,end=' ')

            print("Background: read %d frames of %dx%d"%(nf,f1.shape[0],f1.shape[1]) )
            buf_movie=buf_movie[0:nf,:,:] # Trim to correct
            self.offline_background = buf_movie

            if self.offline_movie.shape[0] != self.offline_background.shape[0]:
                print("Sub average ")
                # Different number of frames in background and movie. Subtract mean background from each frame
                offline_mean = np.array(self.offline_background.mean(0),dtype='int32') # Mean across frames
                self.offline_movie = self.offline_movie - offline_mean
                self.offline_movie[ self.offline_movie<0] = 0
                self.offline_movie = np.array( self.offline_movie, dtype='uint8')
                self.ui.add_offline(buf_movie)
            else:
                print("Sub whole movie")
                subbed = np.array(self.offline_movie,dtype='int32') - self.offline_background
                subbed[subbed<0]=0
                subbed=np.array( subbed, dtype='uint8')
                self.ui.add_offline( subbed)


    def load_offline(self,file_info):
        fields=self.layout[1] # TODO: fix
        # file_info: from dialog. Tuple: (list of files, file types)
        if '.bin' in file_info[1]:
            print("Offline: ",file_info[0][0])
            #fil=open(file_info[0][0],'rb')
            bytez=np.fromfile(file_info[0][0],'uint8')
            width =int(np.sqrt(len(bytez)) ) #  Hopefully it's square
            print( width )

            dims=np.zeros(2,dtype='uint16')
            dims[0]=width
            dims[1]=width
            #buf = ByteStream()
            #buf.append(dims) 
            self.shmem_hdr.seek(fields['dimensions']['bytenum_current']) #TODO: nicer
            self.shmem_hdr.write(dims)
            self.shmem_hdr.flush()

            for nbuf in np.arange(4):
                self.shmem_data.seek(nbuf*2048*2048)
                self.shmem_data.write(bytez)
                self.shmem_data.flush()

        elif '.avi' in file_info[1]:
            fname=file_info[0][0]
            print("Offline movie: ",fname)
            vidin = ffmpegcv.VideoCapture(fname)
            buf_movie=None

            with vidin:
                for nf,frame in enumerate(vidin):
                    f1=frame.mean(2)[0:1024,0:1024] # Avg RGB. TODO: crop hard-code
                    if buf_movie is None:
                        buf_movie=np.zeros( (50,f1.shape[0],f1.shape[1]), dtype='uint8') # TODO: grow new chunk if necessary
                    buf_movie[nf]=f1
                    print(nf,end=' ')

            print("Read %d frames of %dx%d"%(nf,f1.shape[0],f1.shape[1]) )
            buf_movie=buf_movie[0:nf,:,:] # Trim to correct
            self.offline_movie = buf_movie
            self.ui.add_offline(buf_movie)

    def metric_patch(self,patch_orig):
        #filtd=gaussian_filter(buf_movie[nframe],3.0)
        po=patch_orig.copy()
        patch=gaussian_filter(po.copy(),3.0)
        permed=gaussian_filter( random.permutation(po.flatten()).reshape(patch.size), 3)
        vals=(np.array(np.sort(patch.flatten()),dtype='int32') - np.sort(permed.flatten()) ) # More bits, to allow negative
        try:
            #vals_norm = vals - vals.min()
            metric1 = np.mean(vals[2800:])
        except ValueError:
            return -1 #vals * 0.0 # Not good

        #self.good_dbg1 = vals[-500:]
        #self.good_dbg2 = vals_norm[-500:]
        #self.good_dbg3 = np.sort(patch.flatten())[-500:]
        #self.good_dbg4 = np.sort(permed.flatten())[-500:]
        return metric1 #_norm

    def box_fit_gauss(self,box_pix,siz):
        sizo=((siz-1)//2) 
        if np.prod(box_pix.shape) < 1:
            print("Too small")
            return 0,0, 9997
        print (box_pix.shape)
        ind_max = np.unravel_index(np.argmax(box_pix, axis=None), box_pix.shape)
        local_pix=box_pix[ind_max[0]-sizo:ind_max[0]+sizo+1,ind_max[1]-sizo:ind_max[1]+sizo+1]

        if np.any( (ind_max[0]<sizo,ind_max[1]<sizo,ind_max[0]>=box_pix.shape[0]+sizo,ind_max[1] >= box_pix.shape[1]+sizo ) ):
            return ind_max[1], ind_max[0],9999 # give up if too close to edge

        lf=local_pix.flatten()

        try:
            soln=np.matmul( lf, self.mati)
            print("ok")
        except AttributeError:
            # Remake inverse matrix to fit quadratic
            idxs=np.arange(siz)-sizo
            XX,YY=np.meshgrid(idxs,idxs)
            XXf=XX.flatten(); YYf=YY.flatten()
            self.pm=np.vstack( [XXf**2, XXf*YYf,YYf*YYf,XXf,YYf,[1]*len(XXf)] ).T # Matches Mulligan
            self.mati=np.linalg.pinv(self.pm).T

            soln=np.matmul( lf, self.mati)
        except ValueError:
            # On the edge maybe?
            return ind_max[1], ind_max[0],9998 # give up if too close to edge

            # Equivalent loopy code:
            #print(idxs)
            #parm_mat = []
            #for Y in idxs:
                #for X in idxs:
                    #row1=( [X*X, X*Y, Y*Y, X, Y, 1] )
                    #try:
                        #parm_mat = np.vstack( (parm_mat,row1))
            #except ValueError:
                #parm_mat = [row1]

        A=0;B=1;C=2;D=3;E=4;F=5
        det1=(soln[B]**2-4*soln[A]*soln[C])
        goody=(2*soln[A]*soln[E]-soln[D]*soln[B])/det1
        goodx=(2*soln[C]*soln[D]-soln[E]*soln[B])/det1

        recon=np.matmul(soln,self.pm.T)
        gof = np.sum( (lf - recon)**2/recon)

        return ind_max[1]+goodx,ind_max[0]+goody,gof

    def offline_nextbox(self,nframe):
        nbox_clicked=self.ui.box_info

        for loop in [0,1]:
            self.box_metrics = np.zeros( self.num_boxes) 
            cenx=np.zeros( self.num_boxes ) / 0.0
            ceny=np.zeros( self.num_boxes ) / 0.0
            centroids=np.zeros(2)
            GAUSS_SD=3
            BOX_THRESH=-0.5

            for nbox in np.arange(self.num_boxes):
                xUL=int( self.box_x[nbox]-self.box_size_pixel//2 )
                yUL=int( self.box_y[nbox]-self.box_size_pixel//2 )
                pix=self.image[ yUL:yUL+int(self.box_size_pixel), xUL:xUL+int(self.box_size_pixel) ] #.copy

                if True: #len(pix) > 100: #==self.box_size_pixel**2:
                    metric1=self.metric_patch(pix)
                    try:
                        #val = np.mean( (metric1[self.good_idx:]-self.good_template[self.good_idx:])**2)
                        val=self.metric_patch(pix)
                    except ValueError:
                        val = -999.0
                    self.box_metrics[nbox]=val

                if val>BOX_THRESH: # Valid boxes
                    for ndim in []: #[0,1]: # Skip this code (max in seperate dims), use the code below which is 2D at-once
                        sig=np.mean(pix,ndim)
                        filtd=sig #gaussian_filter1d(sig,3) # if unfiltereted
                        xmax=np.argmax(filtd)

                        if (xmax<5) or (xmax>self.box_size_pixel-5):
                            continue

                        xlocal=np.arange(xmax-5,xmax+5)
                        a,b,c=np.polyfit(xlocal, filtd[xlocal], 2 )
                        solution=(-b / (2*a) ) # Deriv=0 is peak of Quadratic
                        #print( ax,xmax,solution)
                        centroids[ndim]=solution

                    pix=gaussian_filter(pix,GAUSS_SD) 
                    centroids=self.box_fit_gauss(pix,17)
                    cenx[nbox] = centroids[0] + xUL
                    ceny[nbox] = centroids[1] + yUL
                    #self.box_metrics[nbox] = centroids[2] # gof
                    #print( centroids, end=' ' )

            self.centroids_x=cenx
            self.centroids_y=ceny

            desired = np.all((self.box_metrics > BOX_THRESH, np.isnan(cenx)==False ), 0) *1.0 # binarize 

            print ( desired.shape, desired )

            #X,Y=np.meshgrid( np.arange(desired.shape[1]), np.arange(desired.shape[0]) )

            guess =[ np.sum( desired*self.box_x / np.sum(desired ) ) ,
                np.sum( desired*self.box_y / np.sum(desired ) ),
                np.sqrt( np.sum(desired) / np. pi ) ]

            self.desired=desired

            opt1=minimize( self.circle_err, guess, method='Nelder-Mead')
            self.opt1=opt1['x']

            print("Nextbox OK", opt1 )
            distances = (self.box_x - self.opt1[0])**2 + (self.box_y - self.opt1[1])**2
            box_min = np.argmin( (self.box_x - self.opt1[0])**2 + (self.box_y - self.opt1[1])**2 )
            self.ui.cx = self.box_x[box_min]
            self.ui.cy = self.box_y[box_min]
            self.cx_best = self.box_x[box_min]
            self.cy_best = self.box_y[box_min]
            print("Nextbox OK", opt1, box_min)

            if loop==0:
                #self.make_searchboxes()
                try: # TODO
                    self.make_searchboxes(pupil_radius_pixel=self.iterative_size/2.0*1000/self.ccd_pixel)
                except:
                    self.make_searchboxes()

        self.compute_zernikes()
        zs = self.zernikes

        dx,dy=self.get_deltas(zs,from_dialog=False)

        spot_displace_x =   self.ref_x - self.centroids_x
        spot_displace_y = -(self.ref_y - self.centroids_y)

        self.est_x =   self.box_x - dx*self.focal/self.ccd_pixel
        self.est_y =   self.box_y + dy*self.focal/self.ccd_pixel

        try: # TODO
            if self.iterative_size>0:
                zs = self.zernikes
                factor = self.iterative_size / (self.iterative_size+step)
                z_new =  zs #iterative.extrapolate_zernikes(zs, factor)
                self.shift_search_boxes(z_new,from_dialog=False)
        except:
            pass

        self.ui.mode_offline=True

    def iterative_offline(self):
        pass

    def offline_goodbox(self,nframe):
        nbox=self.ui.box_info

        GOOD_THRESH=0.25 # TODO
        patch=self.ui.box_pix
        self.good_template=self.metric_patch(self.ui.box_pix)
        #self.good_idx=np.where( self.good_template>GOOD_THRESH)[0][0]
        self.good_idx=int(len(self.good_template)*0.6) # TODO
        print("Goodbox", nbox,nframe,self.good_idx, patch.shape, self.box_size_pixel)

    def send_searchboxes(self,shmem_boxes, box_x, box_y, layout_boxes):
        defs=layout_boxes[2]
        fields=layout_boxes[1]

        num_boxes=box_x.shape[0]
        #box_size_pixel=box_size_pixel

        buf = ByteStream()
        for item in box_x:
            buf.append(item, 'd')
        shmem_boxes.seek(fields['box_x']['bytenum_current'])
        shmem_boxes.write(buf)
        shmem_boxes.flush()

        buf = ByteStream()
        for item in box_y:
            buf.append(item, 'd')
        shmem_boxes.seek(fields['box_y']['bytenum_current'])
        shmem_boxes.write(buf)
        shmem_boxes.flush()

        buf = ByteStream()
        for item in self.ref_x:
            buf.append(item, 'd')
        shmem_boxes.seek(fields['reference_x']['bytenum_current'])
        shmem_boxes.write(buf)
        shmem_boxes.flush()

        buf = ByteStream()
        for item in self.ref_y:
            buf.append(item, 'd')
        shmem_boxes.seek(fields['reference_y']['bytenum_current'])
        shmem_boxes.write(buf)
        shmem_boxes.flush()

        if True:
            buf = ByteStream()
            for item in self.influence.T.flatten():
                buf.append(item, 'd')
            shmem_boxes.seek(fields['influence']['bytenum_current'])
            shmem_boxes.write(buf)
            shmem_boxes.flush()

        buf = ByteStream()
        print(  "INF_INV %f"%np.max( self.influence_inv ) )
        for item in self.influence_inv.T.flatten():
            buf.append(item, 'd')
        shmem_boxes.seek(fields['influence_inv']['bytenum_current'])
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

        buf.append(self.nTerms, 'H')
        buf.append(self.nActuators, 'H')

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
        np.save('zterms.npy',self.zterms)

    def update_influence(self):
        #try:
        influence = np.loadtxt(self.ui.json_data["params"]["influence_file"], skiprows=1)
        #except:
            #influence = np.random.normal ( loc=0, scale=0.01, size=(97, self.num_boxes * 2)  )
        valid_idx=np.sum(influence**2,0)>0 # TODO... base on pupil size or something?
        self.influence = influence[:, valid_idx]
        self.influence_inv = np.linalg.pinv(self.influence) # pseudoinverse
        self.nActuators=self.influence.shape[0]
        self.nTerms=self.influence.shape[1]

    def compute_zernikes(self):
        # find slope
        spot_displace_x =   self.ref_x - self.centroids_x
        spot_displace_y = -(self.ref_y - self.centroids_y)

        # Not sure whether should do this:
        #spot_displace_x -= spot_displace_x.mean()
        #spot_displace_y -= spot_displace_y.mean()
        #print( spot_displace_y.mean(), spot_displace_x.mean() )

        self.spot_displace_interpolated = np.zeros( self.num_boxes )

        for nidx in np.arange(self.num_boxes):
            if np.isnan(spot_displace_x[nidx]) or np.isnan(spot_displace_y[nidx]):
                spot_displace_x[nidx] = np.nanmean( spot_displace_x[self.neighbors[nidx]] )
                spot_displace_y[nidx] = np.nanmean( spot_displace_y[self.neighbors[nidx]] )
                self.spot_displace_interpolated[nidx] = 1

            if np.isnan(spot_displace_x[nidx]) or np.isnan(spot_displace_y[nidx]):
                print( "Still nan", nidx, self.neighbors[nidx])
                spot_displace_x[nidx]=0
                spot_displace_y[nidx]=0
                self.spot_displace_interpolated[nidx] = 2

        slope = np.concatenate( (spot_displace_y, spot_displace_x)) * self.ccd_pixel/self.focal;

        self.spot_displace_x = spot_displace_x # debugging
        self.spot_displace_y = spot_displace_y
        self.slope = slope

        coeff=np.matmul(self.zterms,slope)

        # TODO: only need to do this once
        # Order copied from MATLAB code
        self.CVS_to_OSA_map = np.array([3,2, 5,4,6, 9,7,8,10, 15,13,11,12,14,
                                21,19,17,16,18,20,
                                27,25,23,22,24,26,28, 35,33,31,29,30,32,34,36,
                                45,43,41,39,37,38,40,42,44,
                                55,53,51,49,47,46,48,50,52,54,
                                65,63,61,59,57,56,58,60,62,64,66,67,68,69,70])
        self.OSA_to_CVS_map = np.array( [np.where(self.CVS_to_OSA_map-2==n)[0][0] for n in np.arange(20)] ) # TODO
        self.OSA_to_CVS_map += 2
        self.zernikes=coeff[self.CVS_to_OSA_map[0:NUM_ZCS-1]-1-1] # Return value will is OSA

    def autoshift_searchboxes(self):
        #shift_search_boxes(self,zs,from_dialog=True):
        pass
        #return

    def calc_diopters(self):
        radius = self.pupil_radius_mm
        radius2 = radius*radius
        EPS=1e-10
        sqrt3=np.sqrt(3.0)
        sqrt6=np.sqrt(6.0)
        z3=self.zernikes[3-1]
        z4=self.zernikes[4-1]
        z5=self.zernikes[5-1]

        J45 =  (-2.0 * sqrt6 / radius2) * z3
        J180 = (-2.0 * sqrt6 / radius2) * z5
        cylinder = (4.0 * sqrt6 / (radius * radius)) * np.sqrt((z3 * z3) + (z5 * z5))
        sphere = (-4.0 * sqrt3 * z4 / (radius * radius)) - 0.5 * cylinder

        if (np.abs(z5) <= EPS):
            thetaRad = 1.0 if (np.abs(z3) > EPS) else -1.0
            thetaRad *= float(np.pi) / 4.0
        else:
            thetaRad = 0.5 * np.arctan(J45 / J180)

        axis = thetaRad * 180.0 / float(np.pi)
        if (axis < 0.0):
            axis += 180.0

        rms=np.sqrt( np.nansum(self.zernikes[(3-1):]**2 ) )
        rms5p=np.sqrt( np.nansum(self.zernikes[(6-1):]**2 ) )

        return rms,rms5p,cylinder,sphere,axis

    def get_deltas(self,zs,from_dialog):
        zern_new = np.zeros(NUM_ZERNIKES)
        if from_dialog:
            # TODO: Figure out relation to OSA_to_CVS, etc.
            dlg_fix=np.array([1,0,3,2,4,6,7,5,8,11,12,10,13,9,17,16,18,15,18,14],dtype='int')
            zern_new[0:20]=np.array(zs[0:20])[dlg_fix]
            zern_new[0:2]=0 # Remove tip/tilt
        else:
            dlg_fix=np.array([1,0,3,2,4,6,7,5,8,11,12,10,13,9,17,16,18,15,18,14],dtype='int')
            zern_new[0:20]=np.array(zs[0:20])[dlg_fix]
            zern_new[0:2]=0 # Remove tip/tilt
        delta=np.matmul(self.zterms_inv,zern_new)
        num_boxes = self.box_x.shape[0]
        delta_y = delta[0:num_boxes] #/(self.ccd_pixel/self.focal)
        delta_x = delta[num_boxes:] #/(self.ccd_pixel/self.focal)
        self.delta_x=delta_x
        self.delta_y=delta_y
        self.zern_new = zern_new
        self.zs = zs
        return delta_x,delta_y


    def shift_search_boxes(self,zs,from_dialog=True):
        dx,dy = self.get_deltas(zs,from_dialog)
        self.box_x = self.initial_x - dx
        self.box_y = self.initial_y + dy
        self.update_searchboxes()

    def shift_references(self,zs,from_dialog=True):
        dx,dy = self.get_deltas(zs,from_dialog)
        self.ref_x = self.initial_x - dx
        self.ref_y = self.initial_y + dy
        #self.update_zernike_svd() // This happens as part of the "send" on next snap

    def reset_search_boxes(self):
        self.make_searchboxes() # Reconstruct from cx,cy and theoretical pos
        self.update_searchboxes()    

    def reset_references(self):
        return

    def receive_image(self):
        # TODO: Wait until it's safe (unlocked)

        #TODO. Could use this method to read everything into memory. Probably more efficient:
        #self.shmem_hdr.seek(0)
        #mem_header=self.shmem_hdr.read(MEM_LEN)

        # This divider needs to match that in the engine code
        self.fps0=extract_memory.get_array_item2(self.layout,self.shmem_hdr,'fps',0, False)/100.0
        self.fps1=extract_memory.get_array_item2(self.layout,self.shmem_hdr,'fps',1, False)/100.0
        self.fps2=extract_memory.get_array_item2(self.layout,self.shmem_hdr,'fps',2, False)/100.0

        self.height=extract_memory.get_array_item2(self.layout,self.shmem_hdr,'dimensions',0, False)
        self.width=extract_memory.get_array_item2(self.layout,self.shmem_hdr,'dimensions',1, False)

        self.total_frames=extract_memory.get_array_item2(self.layout,self.shmem_hdr,'total_frames',0, False)

        nwhich_buffer=extract_memory.get_array_item2(self.layout,self.shmem_hdr,'current_frame',0, False)
        
        self.shmem_data.seek(self.width*self.height*nwhich_buffer)
        im_buf=self.shmem_data.read(self.width*self.height)
        bytez =np.frombuffer(im_buf, dtype='uint8', count=self.width*self.height )
        bytes2=np.reshape(bytez,( self.height,self.width)).copy()

        bytesf = bytes2 / np.max(bytes2)

        if False: #self.chkFollow.isChecked():
            box_x,box_y=rcv_searchboxes(self.shmem_boxes, self.layout_boxes, 0, 0, 0 )
            self.box_x = np.array(box_x)
            self.box_y = np.array(box_y)

        self.image = bytes2

        return self.image

    def write_mirrors(self,data):
        bytez =np.array(data, dtype="double").tobytes() 
        fields=self.layout_boxes[1] # TODO: fix
        self.shmem_boxes.seek(fields['mirror_voltages']['bytenum_current'])
        self.shmem_boxes.write(bytez)
        self.shmem_boxes.flush()

    def zero_do(self):
        self.write_mirrors( np.zeros(97) ) # TODO

    def flat_do(self):
        self.write_mirrors( self.mirror_state_flat )

    def flat_save(self):
        self.mirror_state_flat = np.copy(self.mirror_voltages)

    def do_snap(self, mode):
        buf = ByteStream()
        buf.append(mode) # TODO: MODE_CENTROIDING
        self.shmem_hdr.seek(2) #TODO: get address
        self.shmem_hdr.write(buf)
        self.shmem_hdr.flush()    
    
    def do_calibration(self,updater):
        mirrors = np.zeros( 98 ) # TODO
        
        slopes_x=np.zeros( (98, self.num_boxes) )
        slopes_y=np.zeros( (98, self.num_boxes) )
        for n in np.arange(98): #len( mirrors ):
            mirrors *= 0
            
            if n >0:
                mirrors[n] = 0.15
                
            updater( "Calibrating +%d (%f).."%(n, np.sum(mirrors)) )

            self.write_mirrors( mirrors )
            
            # First make sure mirrors are read in and programmed
            self.do_snap(0x40) # TODO: MODE_CALIBRATING
            self.do_snap(0x40) # TODO: MODE_CALIBRATING
            #self.do_snap(0x40) # TODO: MODE_CALIBRATING
            #time.sleep(0.1)
            
            slopes_x[n] *= 0
            slopes_y[n] *= 0
            NUM_ITS=3
            for it in np.arange(NUM_ITS):
                self.do_snap(0x40) # TODO: MODE_CALIBRATING
                # The main loop will keep snapping and calc-ing, so we can just poll that occasionally
                # time.sleep( 0.1 ) # TODO: better to wait for status/handshake
                SIZEOF_DOUBLE=8
                fields=self.layout_boxes[1]            
                self.shmem_boxes.seek(fields['delta_x']['bytenum_current'])
                buf=self.shmem_boxes.read(self.num_boxes*SIZEOF_DOUBLE)
                delta_x=struct.unpack_from(''.join((['d']*self.num_boxes)), buf)

                self.shmem_boxes.seek(fields['delta_y']['bytenum_current'])
                buf=self.shmem_boxes.read(self.num_boxes*SIZEOF_DOUBLE)
                delta_y=struct.unpack_from(''.join((['d']*self.num_boxes)), buf)
                
                print ( n, np.max( delta_x), np.min(delta_x), np.mean(delta_x) )
                
                slopes_x[n] += delta_x
                slopes_y[n] += delta_y
            slopes_x[n] /= NUM_ITS
            slopes_y[n] /= NUM_ITS
            
        np.savez("calib_p.npz",slopes_x, slopes_y);
        
        for n in np.arange(97): #len( mirrors ):
            print( "Calibrating -%d.."%n)
            mirrors *= 0
            mirrors[n] = -0.2
            self.write_mirrors( mirrors )
            
            # First make sure mirrors are read in and programmed
            self.do_snap(0x40) # TODO: MODE_CALIBRATING
            self.do_snap(0x40) # TODO: MODE_CALIBRATING
            #time.sleep(0.25)
            
            slopes_x[n] *= 0
            slopes_y[n] *= 0
            NUM_ITS=3
            for it in np.arange(NUM_ITS):
                self.do_snap(0x40) # TODO: MODE_CALIBRATING
                # The main loop will keep snapping and calc-ing, so we can just poll that occasionally
                #time.sleep(0.25) # TODO: better to wait for status/handshake
                SIZEOF_DOUBLE=8
                fields=self.layout_boxes[1]            
                self.shmem_boxes.seek(fields['delta_x']['bytenum_current'])
                buf=self.shmem_boxes.read(self.num_boxes*SIZEOF_DOUBLE)
                delta_x=struct.unpack_from(''.join((['d']*self.num_boxes)), buf)

                self.shmem_boxes.seek(fields['delta_y']['bytenum_current'])
                buf=self.shmem_boxes.read(self.num_boxes*SIZEOF_DOUBLE)
                delta_y=struct.unpack_from(''.join((['d']*self.num_boxes)), buf)
                
                print ( n, np.max( delta_x), np.min(delta_x), np.mean(delta_x) )
                
                slopes_x[n] += delta_x
                slopes_y[n] += delta_y
            slopes_x[n] /= NUM_ITS
            slopes_y[n] /= NUM_ITS
            
        np.savez("calib_n.npz",slopes_x, slopes_y);        
            

    def receive_centroids(self):
        SIZEOF_DOUBLE=8
        fields=self.layout_boxes[1]
        self.shmem_boxes.seek(fields['centroid_x']['bytenum_current'])
        buf=self.shmem_boxes.read(self.num_boxes*SIZEOF_DOUBLE)
        self.centroids_x=struct.unpack_from(''.join((['d']*self.num_boxes)), buf)

        self.shmem_boxes.seek(fields['centroid_y']['bytenum_current'])
        buf=self.shmem_boxes.read(self.num_boxes*SIZEOF_DOUBLE)
        self.centroids_y=struct.unpack_from(''.join((['d']*self.num_boxes)), buf)

        self.shmem_boxes.seek(fields['delta_x']['bytenum_current'])
        buf=self.shmem_boxes.read(self.num_boxes*SIZEOF_DOUBLE)
        self.delta_x=struct.unpack_from(''.join((['d']*self.num_boxes)), buf)

        self.shmem_boxes.seek(fields['delta_y']['bytenum_current'])
        buf=self.shmem_boxes.read(self.num_boxes*SIZEOF_DOUBLE)
        self.delta_y=struct.unpack_from(''.join((['d']*self.num_boxes)), buf)

        self.shmem_boxes.seek(fields['mirror_voltages']['bytenum_current'])
        buf=self.shmem_boxes.read(self.num_boxes*SIZEOF_DOUBLE)
        #self.mirror_voltages=np.array( struct.unpack_from(''.join((['d']*self.nActuators)), buf) )

        DEBUGGING=False
        if DEBUGGING:
            print (self.num_boxes, np.min(self.centroids_x), np.max(self.centroids_x)  )
            for n in np.arange(self.num_boxes):
                if np.isnan(self.centroids_x[n]):
                    print( n, end=' ')
                    print (self.num_boxes, self.centroids_x[100], self.centroids_y[100]  )

    def send_quit(self):
        buf = ByteStream()
        #buf.append(0)  # Lock
        buf.append(255) # TODO: MODE from() file
        #buf.append(1, 'H') # NUM BOXES. Hopefully doesn't matter
        #buf.append(40, 'd')
        #buf.append(500, 'd')
        self.shmem_hdr.seek(2) # TODO: get address
        self.shmem_hdr.write(buf)
        self.shmem_hdr.flush()

    def mode_init(self):
        self.mode=1
        self.init_params(False)
        self.mode_snap()
        #time.sleep(0.1)

    def mode_snap(self, reinit=True, allow_AO=True):
        if self.ui.mode_offline:
            return

        self.mode=2
        if reinit:
            self.init_params()
            self.update_searchboxes()

        val=3 if (self.ui.chkLoop.isChecked() and allow_AO) else 2
        buf = ByteStream()
        buf.append(val) # TODO: MODE_CENTROIDING
        self.shmem_hdr.seek(2) #TODO: get address
        self.shmem_hdr.write(buf)
        self.shmem_hdr.flush()

    def mode_run(self, reinit=True, numruns=1):
        self.mode=3
        if reinit:
            self.init_params()

        fields=self.layout[1]
        buf = ByteStream()

        val= np.array( numruns, dtype='uint64' )
        #val= np.array( self.ui.edit_num_runs.text(), dtype='uint64' )
        #buf.append(val.tobytes())
        self.shmem_hdr.seek(fields['frames_left']['bytenum_current'])
        self.shmem_hdr.write(val.tobytes())
        self.shmem_hdr.flush()

        val=9 if self.ui.chkLoop.isChecked() else 8

        buf = ByteStream()
        buf.append(val) # TODO: Greater than MODE_CEN_ONE
        self.shmem_hdr.seek(2) #TODO: get address
        self.shmem_hdr.write(buf)
        self.shmem_hdr.flush()

    def mode_stop(self):
        val=0
        
        buf = ByteStream()
        buf.append(val) # TODO: Back to ready
        self.shmem_hdr.seek(2) #TODO: get address
        self.shmem_hdr.write(buf)
        self.shmem_hdr.flush()
