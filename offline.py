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

GAUSS_SD=3
BOX_THRESH=2.5

OFFLINE_ITERATIVE_START=3.0

class NextwaveOffline():
    """ Class to manage:
          - Structures needed for realtime engine (boxes/refs, computed centroids, etc.)
          - Communication with the realtime engine (comm. over shared memory)
          - Computation of Zernikes, matrices for SVD, etc.
    """
    def __init__(self,parent):
        # TODO:
        self.parent = parent

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
            if self.parent.ui.mode_offline:
                self.iterative_offline()
                return # Don't get boxes from engine

            self.mode_snap(False,False)
            mode=self.read_mode()
            # TODO: don't wait forever; lokcup
            while( mode>1 ):
                mode=self.read_mode()
                time.sleep(0.005)

            #self.receive_centroids() # TODO
            self.compute_zernikes()
            zs = self.zernikes

            factor = self.iterative_size / (self.iterative_size+step)
            z_new =  iterative.extrapolate_zernikes(zs, factor)
            #print( zs[5], zs[0:5] )
            #print( z_new[0:5] )
            self.shift_search_boxes(z_new,from_dialog=False)

    def set_iterative_size(self,value):
        self.iterative_size = value

    def offline_frame(self,nframe):
        dims=np.zeros(2,dtype='uint16')
        dims[0]=self.offline_movie[nframe].shape[0]
        dims[1]=self.offline_movie[nframe].shape[1]
        bytez=self.offline_movie[nframe]
        self.parent.comm.write_image(dims,bytez)

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
                self.parent.ui.add_offline(buf_movie)
            else:
                print("Sub whole movie")
                subbed = np.array(self.offline_movie,dtype='int32') - self.offline_background
                subbed[subbed<0]=0
                subbed=np.array( subbed, dtype='uint8')
                self.parent.ui.add_offline( subbed)

    def load_offline(self,file_info):
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
            self.parent.comm.write_image(dims,bytez)

            buf_movie=np.array([im])
            self.offline_movie = buf_movie
            self.parent.ui.add_offline(buf_movie)

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
            self.parent.ui.add_offline(buf_movie)

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

            try:
                soln=np.matmul( lf, self.mati)
            except ValueError:
                return ind_max[1], ind_max[0],-998 # give up if too close to edge
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
        num_boxes = self.parent.num_boxes

        self.box_metrics = np.zeros( num_boxes) 
        cenx=np.full( num_boxes, np.nan )
        ceny=np.full( num_boxes, np.nan )
        centroids=np.zeros(2)
        box_size_pixel = self.parent.box_size_pixel

        for nbox in np.arange(num_boxes):
            xUL=int( self.parent.box_x[nbox]-box_size_pixel//2 )
            yUL=int( self.parent.box_y[nbox]-box_size_pixel//2 )
            im = self.parent.image_bytes
            pix=np.array( im[ yUL:yUL+int(box_size_pixel), xUL:xUL+int(box_size_pixel) ]).copy()

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
                for ndim in []: #[0,1]: # Skip this code (max in separate dims), use the code below which is 2D at-once
                    sig=np.mean(pix,ndim)
                    filtd=sig #gaussian_filter1d(sig,3) # if unfiltereted
                    xmax=np.argmax(filtd)

                    if (xmax<5) or (xmax>box_size_pixel-5):
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
            self.parent.centroids_x=self.cenx
            self.parent.centroids_y=self.ceny

        self.parent.compute_zernikes()
        self.zernikes = self.parent.zernikes # TODO: 

        dx,dy=self.parent.get_deltas(self.zernikes,from_dialog=False)

        #spot_displace_x =   self.parent.ref_x - self.parent.centroids_x
        #spot_displace_y = -(self.parent.ref_y - self.parent.centroids_y)

        self.est_x =  self.parent.ref_x - dx
        self.est_y =  self.parent.ref_y + dy

    def offline_auto2(self):
        #self.make_searchboxes()
        self.box_size_pixel = self.box_size_pixel - 5

        self.offline_centroids() # TODO: DEBUG
        return

        print( self.offline_movie.shape)
        for nframe in np.arange(self.offline_movie.shape[0]):
            self.parent.ui.offline_curr=nframe
            self.offline_frame(self.parent.ui.offline_curr)
            self.offline_startbox()
            self.offline_auto()

    def offline_auto(self):
        #it1=self.offline_stepbox()
        #while it1>0:
            #it1=self.offline_stepbox()
        self.offline_centroids()

        #zs = self.zernikes
        #self.shift_search_boxes(zs,from_dialog=False) # Shift by appropriate number

    def offline_stepbox(self,step_size,max_size=6.4):
        self.offline_centroids()
        zs = self.zernikes

        #max_size = self.box_size_pixel * (self.opt1[2]+1.0)

        ccd_pixel = self.parent.ccd_pixel
        focal = self.parent.focal

        it_size_pix=self.iterative_size / 2.0 * 1000.0/ccd_pixel
        if it_size_pix < max_size /2.0 * 1000 / ccd_pixel:
            factor = self.iterative_size / (self.iterative_size+step_size)
            z_new =  iterative.extrapolate_zernikes(zs, factor)

            self.iterative_size += step_size 

            self.parent.ui.cx -= int( z_new[1] / focal * ccd_pixel )
            self.parent.ui.cy += int( z_new[0] / focal * ccd_pixel )

            self.parent.init_params( {'pupil_diam': self.iterative_size})
            self.parent.make_searchboxes(pupil_radius_pixel=self.iterative_size/2.0*1000/ccd_pixel)

            #print( z_new[0:10] )
            # Truncate to 20 terms To use the reduced mattrix
            z_bigger = np.zeros( self.parent.zterms_full.shape[0])
            z_bigger[0:len(z_new)] = z_new

            self.parent.shift_search_boxes(z_bigger,from_dialog=False) 
            self.offline_centroids()
            zs = self.zernikes
            self.parent.shift_search_boxes(zs,from_dialog=False) 
        else:
            print ("Shrink!")

            self.compute_zernikes()
            zs = self.zernikes

            frame_name = self.offline_fname + "_%02d.png"%self.parent.ui.offline_curr
            self.parent.ui.update_ui()
            self.parent.ui.image.save(frame_name)

            s="%d,%d,%f,%d,%d,"%(self.parent.ui.offline_curr,self.parent.num_boxes,self.iterative_size,self.parent.ui.cx,self.parent.ui.cy)

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

            #double_test = [self.parent.ui.image, self.parent.ui.image]

            #with TiffImagePlugin.AppendingTiffWriter("./test.tiff",True) as tf:
                #for im1 in double_test:
                    #im1.save(tf)
                    #tf.newFrame()

            return -1

        #print( self.opt1, self.iterative_size )
        return 1

    def offline_startbox(self):
        #self.parent.ui.cx=518 # TODO
        #self.parent.ui.cy=491

        self.pupil_radius_pixel = np.sqrt(600**2+600**2)
        self.parent.make_searchboxes()

        self.offline_centroids()

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
        self.parent.ui.cx = self.box_x[box_min]
        self.parent.ui.cy = self.box_y[box_min]
        self.cx_best = self.box_x[box_min]
        self.cy_best = self.box_y[box_min]
        print("Startbox OK", opt1, box_min)

        self.make_searchboxes() # Use new center
        self.offline_centroids()
        self.centroids_x=self.cenx
        self.centroids_y=self.ceny

        self.iterative_size = OFFLINE_ITERATIVE_START
        self.parent.ui.mode_offline=True

    def iterative_offline(self):
        pass

    def offline_goodbox(self,nframe):
        nbox=self.parent.ui.box_info

        GOOD_THRESH=0.25 # TODO
        patch=self.parent.ui.box_pix
        self.good_template=self.metric_patch(self.parent.ui.box_pix)
        #self.good_idx=np.where( self.good_template>GOOD_THRESH)[0][0]
        self.good_idx=int(len(self.good_template)*0.6) # TODO
        print("Goodbox", nbox,nframe,self.good_idx, patch.shape, self.box_size_pixel)
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
