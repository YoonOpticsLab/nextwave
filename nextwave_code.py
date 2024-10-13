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

import zernike_functions
import iterative

from nextwave_comm import NextwaveEngineComm

import ffmpegcv # Read AVI... Better than OpenCV (built-in ffmpeg?)

from PIL import Image, TiffImagePlugin

WINDOWS=(os.name == 'nt')


NACT_PER_NZERN=4

GAUSS_SD=3
BOX_THRESH=2.5

OFFLINE_ITERATIVE_START=3.0
OFFLINE_ITERATIVE_STEP=0.25
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

class NextwaveEngine():
    """ Class to manage:
          - Structures needed for realtime engine (boxes/refs, computed centroids, etc.)
          - Communication with the realtime engine (comm. over shared memory)
          - Computation of Zernikes, matrices for SVD, etc.
    """
    def __init__(self,ui):
        # TODO:
        self.ui = ui
        self.mode = 0
        self.comm = NextwaveEngineComm(self)

    def init(self):
        self.comm.init()
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
            self.pupil_diam=overrides.get('pupil_diam',self.pupil_diam)

        self.pupil_radius_mm=self.pupil_diam / 2.0
        self.pupil_radius_pixel=self.pupil_radius_mm * 1000 / self.ccd_pixel
        self.box_size_pixel=self.box_um / self.ccd_pixel

        if overrides:
            self.box_size_pixel=overrides.get('box_size_pixel',self.box_size_pixel)

        self.ri_ratio = self.pupil_radius_pixel / self.box_size_pixel
        print( "Init:", self.ri_ratio, self.box_size_pixel, self.pupil_radius_pixel )

        bytez =np.array([self.ccd_pixel, self.box_um, self.pupil_radius_mm], dtype='double').tobytes() 

        self.comm.write_params(overrides)

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
        #print(p)
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
            self.init_params( {'pupil_diam': self.iterative_size})

            #self.update_zernike_svd() # TODO: maybe integrate into make_sb
            #self.send_searchboxes(self.shmem_boxes, self.box_x, self.box_y, self.layout_boxes)
            if self.ui.mode_offline:
                self.iterative_offline()
                return # Don't get boxes from engine

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
            z_new =  iterative.extrapolate_zernikes(zs, factor)
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

    def make_searchboxes(self,cx=None,cy=None,aperture=1.0,pupil_radius_pixel=None, box_spacing_pixel=None):
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
        if box_spacing_pixel is None:
            box_spacing_pixel=box_size_pixel

        ri_ratio = pupil_radius_pixel / box_spacing_pixel

        print( "Make SB ",pupil_radius_pixel, box_size_pixel, box_spacing_pixel, ri_ratio )
        aperture = ri_ratio * aperture

        # The max number of boxes possible + or -
        max_boxes = np.ceil( pupil_radius_pixel/ box_spacing_pixel )

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

        MULT = ri_ratio * box_spacing_pixel
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
        self.num_boxes= num_boxes
        self.centroids_x = np.zeros(num_boxes) + np.nan
        self.centroids_y = np.zeros(num_boxes) + np.nan

        self.update_zernike_svd() # Precompute


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
        self.comm.send_searchboxes(self.box_x, self.box_y)
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
        fname = file_info[0][0]
        self.offline_fname = fname
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

        elif '.bmp' in file_info[1]:
            print("Offline: ",file_info[0][0])

            im = Image.open(file_info[0][0])
            bytez = np.array(im) # TODO: assumes Im is already 8bit monochrome

            dims=np.zeros(2,dtype='uint16')
            dims[0]=bytez.shape[0]
            dims[1]=bytez.shape[1]
            #buf = ByteStream()
            #buf.append(dims) 
            self.shmem_hdr.seek(fields['dimensions']['bytenum_current']) #TODO: nicer
            self.shmem_hdr.write(dims)
            self.shmem_hdr.flush()

            for nbuf in np.arange(4):
                self.shmem_data.seek(nbuf*2048*2048)
                self.shmem_data.write(bytez)
                self.shmem_data.flush()

            buf_movie=np.array([im])
            self.offline_movie = buf_movie
            self.ui.add_offline(buf_movie)

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

        out_fname = self.offline_fname + "_zern.csv"
        self.f_out = open(out_fname,'w')
        s="frame_num,num_boxes,pupil,cx,cy,"
        for nz in np.arange(65):
            s += "Z%d,"%(nz+1)
        s += "\n"
        self.f_out.write(s)

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
            return 0,0, -997
        ind_max = np.unravel_index(np.argmax(box_pix, axis=None), box_pix.shape)
        local_pix=box_pix[ind_max[0]-sizo:ind_max[0]+sizo+1,ind_max[1]-sizo:ind_max[1]+sizo+1]

        if np.any( (ind_max[0]<sizo,ind_max[1]<sizo,ind_max[0]>=box_pix.shape[0]+sizo,ind_max[1] >= box_pix.shape[1]+sizo ) ):
            return ind_max[1], ind_max[0],-999 # give up if too close to edge

        lf=local_pix.flatten()

        try:
            soln=np.matmul( lf, self.mati)
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
            return ind_max[1], ind_max[0],-998 # give up if too close to edge

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

        # Peak location inside entire box:
        goodx = goodx + ind_max[1]
        goody = goody + ind_max[0]

        recon=np.matmul(soln,self.pm.T)
        #gof = np.sum( (lf - recon)**2/recon)
        xidx=int(round(goodx))
        yidx=int(round(goody))
        try:
            gof = box_pix[yidx,xidx] - np.mean(box_pix)
        except:
            gof = 0

        return goodx,goody,gof

    def offline_centroids(self,do_apply=True):
            self.box_metrics = np.zeros( self.num_boxes) 
            cenx=np.zeros( self.num_boxes ) / 0.0
            ceny=np.zeros( self.num_boxes ) / 0.0
            centroids=np.zeros(2)

            for nbox in np.arange(self.num_boxes):
                xUL=int( self.box_x[nbox]-self.box_size_pixel//2 )
                yUL=int( self.box_y[nbox]-self.box_size_pixel//2 )
                pix=np.array(self.image[ yUL:yUL+int(self.box_size_pixel), xUL:xUL+int(self.box_size_pixel) ]).copy()

                if True: #len(pix) > 100: #==self.box_size_pixel**2:
                    metric1=self.metric_patch(pix)
                    try:
                        #val = np.mean( (metric1[self.good_idx:]-self.good_template[self.good_idx:])**2)
                        val=self.metric_patch(pix)
                    except ValueError:
                        val = -999.0
                    #self.box_metrics[nbox]=val
                    self.box_metrics[nbox]=BOX_THRESH*2.0

                if True: #val>BOX_THRESH: # Valid boxes
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
                    self.box_metrics[nbox] = centroids[2] # gof
                    #print( centroids, end=' ' )

                    if centroids[2] < BOX_THRESH:
                        cenx[nbox] = np.nan
                        ceny[nbox] = np.nan

            self.cenx = cenx
            self.ceny = ceny

            if do_apply:
                self.centroids_x=self.cenx
                self.centroids_y=self.ceny

    def offline_centroids_update(self):
        self.compute_zernikes()
        zs = self.zernikes
        print(  "in update:", zs[0:3])

        dx,dy=self.get_deltas(zs,from_dialog=False)

        spot_displace_x =   self.ref_x - self.centroids_x
        spot_displace_y = -(self.ref_y - self.centroids_y)

        self.est_x =   self.box_x - dx*self.focal/self.ccd_pixel
        self.est_y =   self.box_y + dy*self.focal/self.ccd_pixel

    def offline_auto2(self):
        #self.make_searchboxes()
        self.box_size_pixel = self.box_size_pixel - 5

        self.offline_centroids() # TODO: DEBUG
        self.offline_centroids_update();
        return

        print( self.offline_movie.shape)
        for nframe in np.arange(self.offline_movie.shape[0]):
            self.ui.offline_curr=nframe
            self.offline_frame(self.ui.offline_curr)
            self.offline_startbox()
            self.offline_auto()

    def offline_auto(self):
        #it1=self.offline_stepbox()
        #while it1>0:
            #it1=self.offline_stepbox()
        self.offline_centroids()
        self.offline_centroids_update()
        zs = self.zernikes
        self.shift_search_boxes(zs,from_dialog=False) # Shift by appropriate number

    def offline_stepbox(self):
        self.offline_centroids()
        self.offline_centroids_update()
        zs = self.zernikes

        it_size_pix=self.iterative_size / 2.0 * 1000.0/self.ccd_pixel
        if it_size_pix < self.box_size_pixel * (self.opt1[2]+1.0):
            factor = self.iterative_size / (self.iterative_size+OFFLINE_ITERATIVE_STEP)
            #z_new =  iterative.extrapolate_zernikes(zs, factor)
            z_new=zs
            self.iterative_size += OFFLINE_ITERATIVE_STEP

            #self.offline_centroids()
            #self.offline_centroids_update()
            #print( z_new[0:3])
            #z_new[0:2]=0 # Clear out tip/tilt. Use as center

            self.ui.cx -= z_new[1] / self.focal * self.ccd_pixel
            self.ui.cy += z_new[0] / self.focal * self.ccd_pixel
            self.init_params( {'pupil_diam': self.iterative_size})
            self.make_searchboxes(pupil_radius_pixel=self.iterative_size/2.0*1000/self.ccd_pixel)

            #print( z_new[0:10] )
            #self.shift_search_boxes(z_new,from_dialog=False) # Shift by appropriate number

            #self.offline_centroids()
            #self.offline_centroids_update()

            #zs = self.zernikes
            #self.shift_search_boxes(zs,from_dialog=False) # Shift by appropriate number
        else:
            print ("Shrink!")

            self.compute_zernikes()
            zs = self.zernikes

            frame_name = self.offline_fname + "_%02d.png"%self.ui.offline_curr
            self.ui.update_ui()
            self.ui.image.save(frame_name)

            s="%d,%d,%f,%d,%d,"%(self.ui.offline_curr,self.num_boxes,self.iterative_size,self.ui.cx,self.ui.cy)

            for zern1 in self.zernikes:
                s += "%0.6f,"%zern1

            s += '\n'
            print(s)

            self.f_out.write(s)
            self.f_out.flush()

            #self.box_size_pixel = int( self.box_size_pixel * 0.8 )
            #self.init_params(overrides={'box_size_pixel': int(self.box_size_pixel*0.9)})
            #self.make_searchboxes()
            #self.offline_centroids()
            #self.offline_centroids_update()

            #double_test = [self.ui.image, self.ui.image]

            #with TiffImagePlugin.AppendingTiffWriter("./test.tiff",True) as tf:
                #for im1 in double_test:
                    #im1.save(tf)
                    #tf.newFrame()

            return -1

        print( self.opt1, self.iterative_size )
        return 1

    def offline_startbox(self):
        #self.ui.cx=518 # TODO
        #self.ui.cy=491

        self.pupil_radius_pixel = np.sqrt(600**2+600**2)
        self.make_searchboxes()

        self.offline_centroids()
        self.offline_centroids_update()

        desired = np.all((self.box_metrics > BOX_THRESH, np.isnan(self.cenx)==False ), 0) *1.0 # binarize 

        print ( desired.shape, desired )

        guess =[ np.sum( desired*self.box_x / np.sum(desired ) ) ,
            np.sum( desired*self.box_y / np.sum(desired ) ),
            np.sqrt( np.sum(desired) / np. pi ) ]

        self.desired=desired

        opt1=minimize( self.circle_err, guess, method='Nelder-Mead')
        self.opt1=opt1['x']

        print("Startbox OK", opt1 )
        distances = (self.box_x - self.opt1[0])**2 + (self.box_y - self.opt1[1])**2
        box_min = np.argmin( (self.box_x - self.opt1[0])**2 + (self.box_y - self.opt1[1])**2 )
        self.ui.cx = self.box_x[box_min]
        self.ui.cy = self.box_y[box_min]
        self.cx_best = self.box_x[box_min]
        self.cy_best = self.box_y[box_min]
        print("Startbox OK", opt1, box_min)

        self.make_searchboxes() # Use new center
        self.offline_centroids()
        self.centroids_x=self.cenx
        self.centroids_y=self.ceny
        self.offline_centroids_update()

        self.iterative_size = OFFLINE_ITERATIVE_START
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
        #print(  "INF_INV %f"%np.max( self.influence_inv ) )
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
        lefts =    self.norm_x - 0.5/self.ri_ratio # (or 0.5 * self.box_size_pixel/self.pup)
        rights =   self.norm_x + 0.5/self.ri_ratio
        ups =    -(self.norm_y + 0.5/self.ri_ratio)
        downs =  -(self.norm_y - 0.5/self.ri_ratio)

        # Compute all integrals for all box corners 
        lenslet_dx,lenslet_dy=zernike_functions.zernike_integral_average_from_corners(
            lefts, rights, ups, downs, self.pupil_radius_mm)

        # Remove piston
        lenslet_dx = lenslet_dx[1:,:]
        lenslet_dy = lenslet_dy[1:,:]

        # TODO: Dumb to make loop, making into function would be better
        for nsubset in [0,1]:
            nmax_from_boxes=int(self.num_boxes/NACT_PER_NZERN)
            nmax_from_boxes= np.min( (nmax_from_boxes,zernike_functions.MAX_ZERNIKES))
            orders_max=np.cumsum(np.arange(zernike_functions.MAX_ORDER+2)) - 1 # Last index in each order
            valid_max = orders_max[nmax_from_boxes<=orders_max][0]
            if nsubset==0:
                nvalid = valid_max
            if nsubset==1: #
                nvalid = np.min( (20, nmax_from_boxes) )

            dx_subset=lenslet_dx[0:nvalid,:]
            dy_subset=lenslet_dy[0:nvalid,:]

            #  Compute SVD of stacked X and Y
            zpoly = np.hstack( (dx_subset, dy_subset ) ).T

            # Pre-compute&save the zterms that are multiplied with the slopes in realtime
            [uu,ss,vv] = svd(zpoly,False)

            # Need to prevent over/underflow. Tiny ss's lead to huge/problematic zterms.
            # https://scicomp.stackexchange.com/questions/26763/how-much-regularization-to-add-to-make-svd-stable
            ss[ ss < 1e-10 ]=0

            ss_full = np.eye(ss.shape[0])*ss
            leftside = lstsq(ss_full, vv, rcond=0)[0].T # Python equiv to MATLAB's vv/ss (solving system of eqns) is lstsq
            # https://stackoverflow.com/questions/1001634/array-division-translating-from-matlab-to-python
            zterms = np.matmul( leftside, uu.T)
            zterms_inv = np.linalg.pinv(zterms)

            if nsubset==0:
                self.zterms_full=zterms
                self.zterms_full_inv=zterms_inv
            elif nsubset==1:
                self.zterms_20=zterms
                self.zterms_20_inv=zterms_inv

        # DBG:
        self.lenslet_dx=lenslet_dx # Debugging
        self.lenslet_dy=lenslet_dy
        np.save('zterms_full.npy',self.zterms_full)
        np.save('zterms_20.npy',self.zterms_20)
        np.save('zpoly.npy',zpoly)

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

        coeff=np.matmul(self.zterms_full,slope)
        self.zernikes=coeff[zernike_functions.CVS_to_OSA_map] # Return value will is OSA

    def autoshift_searchboxes(self):
        #shift_search_boxes(self,zs,from_dialog=True):
        pass
        #return
    def get_deltas(self,zs,from_dialog,full=True):
        #if from_dialog:

        valid_idxs=np.arange(0,self.zterms_full_inv.shape[1] )
        zern_new=np.array(zs[valid_idxs])[zernikes_functions.OSA_to_CVS_map[valid_idxs]]
        zern_new[0:2]=0 # Remove tip/tilt

        delta=np.matmul(self.zterms_full_inv,zern_new)
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
        return self.comm.receive_image()
    def receive_centroids(self):
        return self.comm.receive_centroids()

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

    def send_quit(self):
        self.comm.set_mode(255)

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

        self.comm.set_mode(val)

    def mode_run(self, reinit=True, numruns=1):
        self.mode=3
        if reinit:
            self.init_params()

        fields=self.layout[1]
        buf = ByteStream()

        val= np.array( numruns, dtype='uint64' )
        #val= np.array( self.ui.edit_num_runs.text(), dtype='uint64' )
        #buf.append(val.tobytes())
        self.comm.set_nframes(val)

        val=9 if self.ui.chkLoop.isChecked() else 8
        self.comm.set_mode(val)

    def mode_stop(self):
        self.comm.set_mode(0)
