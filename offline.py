import numpy as np
import sys
import os
import time
import pickle
from pathlib import Path

import matplotlib.cm as cmap
#from numba import jit
from numpy.linalg import svd,lstsq
import scipy
from scipy.optimize import minimize
from scipy import ndimage

import numpy.random as random

# Image processing:
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.optimize import minimize
from skimage import filters

from fit_circle import circle_fitter
import mmap
import struct
import extract_memory

import zernike_functions
import iterative

from nextwave_comm import NextwaveEngineComm
import defaults

# import ffmpegcv # Read AVI... Better than OpenCV (built-in ffmpeg?)
from PIL import Image, TiffImagePlugin # Needed

class info_saver():
    def __init__(self,parent):
        self.parent=parent
        self.offline=parent
        self.engine=parent.parent
        self.ui=self.engine.ui
        self.data = {}

    def save1(self,nframe):
        data_record = {
            'box_x':self.engine.box_x,
            'box_y':self.engine.box_y,
            'ref_x':self.engine.ref_x,
            'ref_y':self.engine.ref_y,
            'centroids_x':self.engine.centroids_x,
            'centroids_y':self.engine.centroids_y,
            'est_x':self.offline.est_x,
            'est_y':self.offline.est_y,
            'cx':self.ui.cx,
            'cy':self.ui.cy,
            'pupil_diam':self.engine.pupil_diam / self.engine.pupil_mag, # In pupil coords, not sensor
            'zernikes':self.engine.zernikes}
        self.data[nframe]=data_record
        #print( 'saved: ', data_record, flush=True)

    def load1(self,nframe):
        try:
            data_record = self.data[nframe]
        except KeyError:
            return None# If this record doesn't exist, just ignore
        self.engine.box_x = data_record['box_x']
        self.engine.box_y = data_record['box_y']
        self.engine.ref_x = data_record['ref_x']
        self.engine.ref_y = data_record['ref_y']
        self.engine.centroids_x = data_record['centroids_x']
        self.engine.centroids_y = data_record['centroids_y']
        self.offline.est_x = data_record['est_x']
        self.offline.est_y = data_record['est_y']
        self.ui.cx = data_record['cx']
        self.ui.cy = data_record['cy']
        self.engine.pupil_diam = data_record['pupil_diam'] * self.engine.pupil_mag  # TODO: Should we rebuild boxes ?
        self.engine.zernikes = data_record['zernikes']

        self.engine.num_boxes = len( self.engine.centroids_x)
        return data_record

    def printable1(self,nframe):
        data_record=self.load1(nframe)
        if not data_record is None:
            try:
                s=("%s,%s,%d,%0.2f,%0.3f,%d,%d,")%(self.offline.sub_id,self.offline.scan_dir,nframe,defaults.scan_frame_to_ecc[self.offline.scan_dir][nframe],data_record['pupil_diam'],data_record['cx'],data_record['cy'])
            except: # without the sub_id params
                s=("%s,%s,%d,%0.2f,%0.3f,%d,%d,")%("","",nframe,0.0,data_record['pupil_diam'],data_record['cx'],data_record['cy'])
            for nz1,z1 in enumerate(data_record['zernikes']):
                s += "%0.6f,"%(z1)
        else:
            s='%d,'%nframe
        return s

    def serialize(self):
        self.fname = self.offline.offline_fname+'.pkl'
        with open(self.fname,'wb') as f:
            pickle.dump(self.data, f)
        f.close()

    def unserialize(self):
        self.fname = self.offline.offline_fname+'.pkl'
        try:
            with open(self.fname,'rb') as f:
                self.data = pickle.load(f)
            f.close() 
        except FileNotFoundError:
            self.data = {}


class NextwaveOffline():
    """ Class to manage:
          - Structures needed for realtime engine (boxes/refs, computed centroids, etc.)
          - Communication with the realtime engine (comm. over shared memory)
          - Computation of Zernikes, matrices for SVD, etc.
    """
    def __init__(self,parent):
        # TODO:
        self.parent = parent
        self.engine = self.parent
        self.ui = self.parent.ui
        self.saver = info_saver(self)
        
    def iterative_run(self, cx, cy, step):
        return

    def circle(self,cx,cy,rad):
        #X,Y=np.meshgrid( np.arange(self.desired.shape[1]), np.arange(self.desired.shape[0]) )
        X=self.parent.box_x
        Y=self.parent.box_y
        r=np.sqrt((X-cx-1.0*np.sign(X))**2+(cy-Y+1.0*np.sign(Y))**2) # 0.5 fudge
        result=(r<rad)*1.0
        return result

        #downs =  -(self.norm_y - 0.5/self.ri_ratio)

    def circle_err(self,p):
        ssq=np.nansum( (self.circle(*p)-self.desired) **2 )
        return ssq

    def offline_frame(self,nframe):
        #dims=np.zeros(2,dtype='uint16')
        #dims[0]=self.offline_movie[nframe].shape[0]
        #dims[1]=self.offline_movie[nframe].shape[1]
        self.dims=np.array( self.offline_movie[nframe].shape, dtype='uint16')  # TODO: np.array( [[shape]], dtype='uint16') seems better
        bytez=self.offline_movie[nframe]
        self.im = bytez
        self.parent.comm.write_image(self.dims,bytez)
        self.ui.image_pixels = bytez

    def load_offline_background(self,file_info):
        # file_info: from dialog. Tuple: (list of files, file types)
        if '.bin' in file_info[1]:
            pass # TODO
        elif '.avi' in file_info[1]:
            fname=file_info[0][0]
            #print("Offline movie: ",fname)

            vidin = ffmpegcv.VideoCapture(fname)
            buf_movie=None

            with vidin:
                for nf,frame in enumerate(vidin):
                    #f1=frame.mean(2)[0:1024,0:1024] # Avg RGB. TODO: crop hard-code
                    f1=frame.mean(2)
                    if buf_movie is None:
                        buf_movie=np.zeros( (50,f1.shape[0],f1.shape[1]), dtype='uint8') # TODO: grow new chunk if necessary
                    buf_movie[nf]=f1
                    #print(nf,end=' ')

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
                self.parent.ui.add_offline(self.offline_movie)
            else:
                print("Sub whole movie")
                subbed = np.array(self.offline_movie,dtype='int32') - self.offline_background
                subbed[subbed<0]=0
                subbed=np.array( subbed, dtype='uint8')
                self.parent.ui.add_offline( subbed)                
        elif '.bmp' in file_info[1]:
            buf_movie=None
            nf=0 # USE nf instead of nf_x to allow skipping (e.g. if directory is in there)
            for nf_x,frame1 in enumerate(file_info[0]):
                if not (".bmp" in frame1):
                    continue
                #print("Offline: ",nf,frame1)
                im = Image.open(frame1)
                f1 = np.array(im) # TODO: assumes Im is already 8bit monochrome
                if buf_movie is None:
                        buf_movie=np.zeros( (50,f1.shape[0],f1.shape[1]), dtype='uint8') # TODO: grow new chunk if necessary
                buf_movie[nf]=f1
                nf += 1

            print("Read %d frames of %dx%d"%(nf,f1.shape[0],f1.shape[1]) )
            buf_movie=buf_movie[0:nf,:,:] # Trim to correct
            self.offline_background = buf_movie

            if self.offline_movie.shape[0] != self.offline_background.shape[0]:
                print("Sub average")
                # Different number of frames in background and movie. Subtract mean background from each frame
                offline_mean = np.array(self.offline_background.mean(0),dtype='int32') # Mean across frames
                self.offline_movie = self.offline_movie - offline_mean
                self.offline_movie[ self.offline_movie<0] = 0
                self.offline_movie = np.array( self.offline_movie, dtype='uint8')
                self.parent.ui.add_offline(self.offline_movie)
            else:
                print("Sub each frame from each frame")
                subbed = np.array(self.offline_movie,dtype='int32') - self.offline_background
                subbed[subbed<0]=0
                subbed=np.array( subbed, dtype='uint8')
                self.offline_movie = subbed                
                self.parent.ui.add_offline( subbed)                

    def load_offline(self,file_info):
        # file_info: from dialog. Tuple: (list of files, file types)
        fname = file_info[0][0]
        self.offline_fname = fname
        
        self.parent.ui.mode_offline=True
        
        if '.bin' in file_info[1]:
            print("Offline: ",file_info[0][0])
            #fil=open(file_info[0][0],'rb')
            bytez=np.fromfile(file_info[0][0],'uint8')
            width =int(np.sqrt(len(bytez)) ) #  Hopefully it's square
            print( width )

            dims=np.zeros(2,dtype='uint16')
            dims[0]=width
            dims[1]=width
            self.dims = dims
            self.parent.comm.write_image(dims,bytez)

        elif '.png' in file_info[1]:
            buf_movie=None
            pathname = file_info[0][0].upper()
            idxSub=pathname.find("SWS") # TODO
            if idxSub==-1:
                self.sub_id="NONAME"
            else:
                self.sub_id = pathname[idxSub+4:idxSub+8]
            
            # Condition might exist as a middle directory, between subId and last directory
            i0=pathname[idxSub:].find('/') + idxSub + 1
            i1=pathname[i0:].find('/') + i0 + 1
            i2=pathname[i1:].find('/') + i1 + 1
            if True or (i1==i2) or -1 in (i0,i1,i2):
                self.condition="COND" # There was no directory between subId and last
            else:
                self.condition=pathname[i0:i1-1]

            idxScanDir=pathname.find("CAM") # TODO
            if idxScanDir==-1:
                self.scan_dir='X'
            else:
                idxScanDir += 5
                if pathname[idxScanDir:idxScanDir+2] == 'D2':
                    self.scan_dir = 'D2'
                else:
                    self.scan_dir = pathname[idxScanDir:idxScanDir+1]  
            
            nf=0 # USE nf instead of nf_x to allow skipping (e.g. if directory is in there)
            for nf_x,frame1 in enumerate(file_info[0]):
                if not (".png" in frame1):
                    continue
                #print("Offline: ",nf,frame1)
                im = Image.open(frame1)
                f1 = np.array(im) # TODO: assumes Im is already 8bit monochrome
                if len(f1.shape)==3: # It's an RGB
                    f1 = f1.mean(2)
                if buf_movie is None:
                        buf_movie=np.zeros( (2048,f1.shape[0],f1.shape[1]), dtype='uint8') # TODO: grow new chunk if necessary
                buf_movie[nf]=f1
                nf += 1

            buf_movie = buf_movie[0:nf]

            print(pathname, self.condition, self.scan_dir, self.sub_id)
            print("Read %d frames of %dx%d"%(nf,f1.shape[0],f1.shape[1]) )
            buf_movie=buf_movie[0:nf,:,:] # Trim to correct
            self.offline_movie = buf_movie
            self.parent.ui.add_offline(buf_movie)
            self.dims=np.array([buf_movie.shape[1],buf_movie.shape[2]])

        elif '.bmp' in file_info[1]:
            buf_movie=None
            pathname = file_info[0][0].upper()
            idxSub=pathname.find("SWS") # TODO
            if idxSub==-1:
                self.sub_id="NONAME"
            else:
                self.sub_id = pathname[idxSub+4:idxSub+8]
            
            # Condition might exist as a middle directory, between subId and last directory
            i0=pathname[idxSub:].find('/') + idxSub + 1
            i1=pathname[i0:].find('/') + i0 + 1
            i2=pathname[i1:].find('/') + i1 + 1
            if (i1==i2) or -1 in (i0,i1,i2):
                self.condition="COND" # There was no directory between subId and last
            else:
                self.condition=pathname[i0:i1-1]

            idxScanDir=pathname.find("CAM") # TODO
            if idxScanDir==-1:
                self.scan_dir='X'
            else:
                idxScanDir += 5
                if pathname[idxScanDir:idxScanDir+2] == 'D2':
                    self.scan_dir = 'D2'
                else:
                    self.scan_dir = pathname[idxScanDir:idxScanDir+1]  
            
            nf=0 # USE nf instead of nf_x to allow skipping (e.g. if directory is in there)
            for nf_x,frame1 in enumerate(file_info[0]):
                if not (".bmp" in frame1):
                    continue
                #print("Offline: ",nf,frame1)
                im = Image.open(frame1)
                f1 = np.array(im) # TODO: assumes Im is already 8bit monochrome
                if buf_movie is None:
                        buf_movie=np.zeros( (50,f1.shape[0],f1.shape[1]), dtype='uint8') # TODO: grow new chunk if necessary
                buf_movie[nf]=f1
                nf += 1

            print(pathname, self.condition, self.scan_dir, self.sub_id)
            print("Read %d frames of %dx%d"%(nf,f1.shape[0],f1.shape[1]) )
            buf_movie=buf_movie[0:nf,:,:] # Trim to correct
            
            buf_movie[buf_movie >= defaults.SATURATION_MINIMUM] = 0
            
            self.offline_movie = buf_movie
            self.parent.ui.add_offline(buf_movie)
            self.dims=np.array([buf_movie.shape[1],buf_movie.shape[2]])

        elif '.avi' in file_info[1]:
            fname=file_info[0][0]
            print("Offline movie: ",fname)
            vidin = ffmpegcv.VideoCapture(fname)
            buf_movie=None

            with vidin:
                for nf,frame in enumerate(vidin):
                    f1=frame.mean(2) #[0:1024,0:1024] # Avg RGB. TODO: crop hard-code
                    if buf_movie is None:
                        buf_movie=np.zeros( (512,f1.shape[0],f1.shape[1]), dtype='uint8') # TODO: grow new chunk if necessary
                    buf_movie[nf]=f1
                    print(nf,end=' ')

            f1 = f1[nf,:,:]

            print("Read %d frames of %dx%d"%(nf,f1.shape[0],f1.shape[1]) )
            buf_movie=buf_movie[0:nf,:,:] # Trim to correct
            self.offline_movie = buf_movie
            self.parent.ui.add_offline(buf_movie)
            self.dims=np.array([buf_movie.shape[1],buf_movie.shape[2]])

        self.max_frame = buf_movie.shape[0]

        self.saver.unserialize() # Load previous if they exist
        self.saver.load1(0) # Restore if possible

    def export_all_zernikes(self,dir1="."):
        idx=0
        #out_fname = self.offline_fname + "_zern_%02d.csv"%idx        
        out_fname = "%s/%s_%s_%s_%02d.csv"%(dir1,self.sub_id,self.condition,self.scan_dir,idx)
        
        while Path(out_fname).exists():
            idx += 1
            #out_fname = self.offline_fname + "_zern_%02d.csv"%idx
            out_fname = "%s/%s_%s_%s_%02d.csv"%(dir1,self.sub_id,self.condition,self.scan_dir,idx)
        
        self.f_out = open(out_fname,'w')
        s="subject_id,scan_dir,frame_num,ecc,pupil_diam_mm,cx,cy,"
        for nz in np.arange(65):
            s += "Z%d,"%(nz+1)
        s += "\n"
        self.f_out.write(s)

        for nframe in np.arange(self.max_frame):
            s=self.saver.printable1(nframe)
            s += "\n"
            self.f_out.write(s)
        self.f_out.close()

    def metric_patch(self,patch_orig):
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

    def box_fit_gauss(self,box_pix,siz,n_which_box=-1,conservative_threshold=True):
        # n_which_box is for debugging
        sizo=((siz-1)//2) 
        if np.prod(box_pix.shape) < 1:
            #print("Too small")
            return 0,0, -997
            
        ind_max = np.unravel_index(np.argmax(box_pix, axis=None), box_pix.shape)
        local_pix=box_pix[ind_max[0]-sizo:ind_max[0]+sizo+1,ind_max[1]-sizo:ind_max[1]+sizo+1]

        if conservative_threshold:
            if np.max(box_pix) - np.mean(box_pix) < 10:
                return np.nan, np.nan, -989     

            # When expanding, use old heuristic that bails (makes NaN) if centroids are too close to the edge to do Gaussian fit.
            if np.any( (ind_max[0]<sizo,ind_max[1]<sizo,ind_max[0]>=box_pix.shape[0]-sizo,ind_max[1] >= box_pix.shape[1]-sizo )
            ):
                return ind_max[1], ind_max[0],-999 # give up if too close to edge
 
        else:
            # Shrinking step
            if np.max(box_pix) - np.mean(box_pix) < 0: # WILL NEVER HAPPEN: allow all boxes
                return np.nan, np.nan, -988
            
        if np.any( (ind_max[0]<sizo,ind_max[1]<sizo,ind_max[0]>=box_pix.shape[0]+sizo,ind_max[1] >= box_pix.shape[1]+sizo )
            ) or np.prod(local_pix.shape) < 25: # Tiny. Use  center of mass
            #idxs = np.arange(box_pix.shape[3]) #-box_pix.shape[1]//2
            XX,YY=np.meshgrid(np.arange(box_pix.shape[0]), np.arange(box_pix.shape[1] ) )
            XXf=XX.flatten()
            YYf=YY.flatten()
            bp1=box_pix.flatten()/255.0
            com_x = np.sum( XXf * bp1 ) / np.sum( bp1 )
            com_y = np.sum( YYf * bp1 ) / np.sum( bp1 )
            #print( "Too small box: ", n_which_box, com_x, com_y )
            return com_x,com_y, 256
            
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
                print( "-998 #1 %d: "%n_which_box + str(lf.min()) + " " + str( lf.max()  ) )
                return ind_max[1], ind_max[0],-998 # give up if too close to edge
        #except ValueError:
            # On the edge maybe?
            #print( "-998 #2 %d:"%n_which_box ) #+ str(lf.min()) + " " + str( lf.max()  ) )
            #return ind_max[1], ind_max[0],-998 # give up if too close to edge

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
            gof = box_pix[yidx,xidx] - np.min(box_pix)
        except:
            gof = 0
            #print("Couldn't COF, n=%d, xidx=%d,yidx=%d"%(n_which_box,xidx,yidx) )

        return goodx,goody,gof

    def get_box_pix(self,nbox):
        box_size_pixel = self.parent.box_size_pixel
        xUL=int( self.parent.box_x[nbox]-box_size_pixel//2 )
        yUL=int( self.parent.box_y[nbox]-box_size_pixel//2 )
        im = self.im #parent.image_bytes
        pix=np.array( im[ yUL:yUL+int(box_size_pixel), xUL:xUL+int(box_size_pixel) ]).copy()
        return pix,xUL,yUL

    def offline_centroids(self,do_apply=True,dark_as_nan=True,conservative_threshold=True):
        num_boxes = self.parent.num_boxes

        self.box_metrics = np.zeros( num_boxes)
        cenx=np.full( num_boxes, np.nan )
        ceny=np.full( num_boxes, np.nan )
        centroids=np.zeros(3)
        box_size_pixel = self.parent.box_size_pixel

        for nbox in np.arange(num_boxes):
            pix,xUL,yUL=self.get_box_pix(nbox)
            #try:
            #    val=self.metric_patch(pix)
            #except ValueError:
            #    val = -999.0
            #self.box_metrics[nbox]=val
            #self.box_metrics[nbox]=BOX_THRESH*2.0
            # Boxes that are off the screen edges
            if ( (self.parent.box_x[nbox]<box_size_pixel//2) or (self.parent.box_y[nbox]<box_size_pixel//2) or
                    (self.parent.box_x[nbox]+box_size_pixel//2>self.im.shape[1]) or 
                    (self.parent.box_y[nbox]+box_size_pixel//2>self.im.shape[0]) ):
                cenx[nbox] = np.nan
                ceny[nbox] = np.nan
                self.box_metrics[nbox]=-990               
            else:
                pix=gaussian_filter(pix,defaults.GAUSS_SD)
                #print( '%03d %s %s'%(nbox, str(pix.shape), str(box_size_pixel) ) )
                centroids=self.box_fit_gauss(pix, defaults.SUBSET_FIT_SIZE, nbox, conservative_threshold=conservative_threshold)
                cenx[nbox] = centroids[0] + xUL
                ceny[nbox] = centroids[1] + yUL
                self.box_metrics[nbox] = centroids[2] # gof

            # Want to keep dark (but in-range) patches for proper optimization
            if (centroids[2] < defaults.BOX_THRESH) and (dark_as_nan):
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

    def offline_auto(self):
        it1=self.offline_stepbox()
        while it1>0:
            it1=self.offline_stepbox()

        #zs = self.zernikes
        #self.shift_search_boxes(zs,from_dialog=False) # Shift by appropriate number

    def offline_stepbox(self):
        self.offline_centroids()
        self.nans_in_prev = np.sum( np.isnan( self.cenx) )
        zs = self.zernikes

        #max_size = self.max_p_diam
        step_size = float(self.parent.ui.it_step.text() )
        ccd_pixel = self.parent.ccd_pixel
        focal = self.parent.focal
        zs_for_extrapolate = np.zeros( 24 ) # Needed for extrapolate function

        self.iterative_size_pixels = self.iterative_size/2.0 * 1000 / ccd_pixel
        if self.iterative_size_pixels < self.iterative_max_pixels: #max_size /2.0 * 1000 / ccd_pixel:
            
            factor = self.iterative_size / (self.iterative_size+step_size)
            num_zs_for_extrapolate = np.min( (defaults.NUM_ZS_FOR_EXTRAPOLATE, len(zs), len(zs_for_extrapolate) ) )
            zs_for_extrapolate *= 0
            zs_for_extrapolate[0:num_zs_for_extrapolate] = zs[0:num_zs_for_extrapolate]
            z_new =  iterative.extrapolate_zernikes(zs_for_extrapolate, factor)

            self.iterative_size += step_size # Diameter
            self.iterative_size_pixels = self.iterative_size/2.0 * 1000 / ccd_pixel # radius
            if self.iterative_size_pixels > self.iterative_max_pixels:
                self.iterative_size = self.iterative_max
                self.iterative_size_pixels = self.iterative_max_pixels

            # Add tip/tilt to the centers
            self.parent.ui.cx -= int( z_new[1] / focal * ccd_pixel )
            self.parent.ui.cy += int( z_new[0] / focal * ccd_pixel )

            self.parent.init_params( {'pupil_diam': self.iterative_size / self.parent.pupil_mag} )
            self.parent.make_searchboxes() #pupil_radius_pixel=self.iterative_size_pixels)
            #print( self.iterative_size_pixels, self.parent.pupil_radius_pixel) # They should match already (debugging)

            # Now shift searchboxes
            zs_for_shift = np.zeros(20) # Needed for shift
            num_zs_for_shift = np.min( (self.parent.zterms_full.shape[0], len(z_new), len(zs_for_shift), defaults.NUM_ZS_FOR_SHIFT) )
            zs_for_shift[0:num_zs_for_shift] = z_new[0:num_zs_for_shift]
            self.parent.shift_search_boxes(zs_for_shift,from_dialog=False) 
            self.offline_centroids(conservative_threshold=True)
            zs = self.zernikes
            self.parent.shift_search_boxes(zs,from_dialog=False)

            # Now shrink the boxes
            self.offline_auto_shrink()

    def offline_serialize(self):
        self.saver.save1(self.parent.ui.offline_curr)
        self.saver.serialize()

    def offline_manual1(self):
        self.parent.offline_frame(self.parent.ui.offline_curr)
        self.iterative_run_good()
        #self.saver.save1(nframe)        

    def offline_autoall(self):
        for nframe in np.arange(self.max_frame):
            # Load
            self.parent.ui.offline_curr=nframe
            self.parent.offline_frame(self.parent.ui.offline_curr)
            #Process
            self.iterative_run_good()
            #self.offline_centroids(conservative_threshold=True) # Now redo, with less conservative
            self.offline_auto_shrink()
            #Save
            self.saver.save1(nframe)
        self.saver.serialize()
    
    def offline_auto_dumb(self):
        self.parent.ui.mode_init()
        for nframe in np.arange(self.max_frame):
            self.parent.ui.offline_curr=nframe
            self.parent.offline_frame(self.parent.ui.offline_curr)
            self.offline_centroids()
            self.saver.save1(nframe)
            
            self.parent.ui.update_ui()
            self.parent.ui.repaint()            
        self.saver.serialize()

# iterative_size is size on sensor
    def offline_reset(self):
        pupil_diam = float(self.parent.ui.it_start.text())
        pupil_diam = pupil_diam * self.parent.pupil_mag
        self.iterative_size = pupil_diam
        self.iterative_size_pixels = self.iterative_size/2.0 * 1000 / self.parent.ccd_pixel
        self.parent.ui.line_pupil_diam.setText('%2.2f'%(self.iterative_size ) ) 
        # pupil_diam is the size on sensor, so divide by mag (because init code multiplies by mag)
        self.parent.init_params( { 'pupil_diam': pupil_diam / self.parent.pupil_mag } ) # Back to real pupil size
        self.parent.make_searchboxes() 

    def fit1(self,r):
        self.pupil_radius_pixel = r * 1000 / self.parent.ccd_pixel 
        self.parent.init_params(
            {'pupil_diam': self.pupil_radius_pixel*2.0/1000.0*self.parent.ccd_pixel } ) #/ self.parent.pupil_mag} )
        self.parent.make_searchboxes(pupil_radius_pixel=self.pupil_radius_pixel)
        self.offline_centroids(dark_as_nan=False)

    def offline_auto_shrink(self):
        size_before = self.parent.box_size_pixel
        while self.parent.box_size_pixel > defaults.SHRINK_MIN:
            self.offline_auto_shrink1()
        self.parent.box_size_pixel = size_before
        
    def offline_auto_shrink1(self):
        #print( self.parent.box_size_pixel )
        #self.offline_centroids()
        #self.parent.make_searchboxes( box_spacing_pixel=self.parent.box_size_pixel-10 )
        #self.parent.shift_search_boxes(-self.zernikes[0:20],from_dialog=False) 

        self.offline_centroids(conservative_threshold=False)
        zs = self.zernikes.copy()
        zs[defaults.NUM_ZS_FOR_SHRINK:] = 0 # Zero out higher-order
        if self.parent.box_size_pixel > 30:
            self.parent.box_size_pixel -= defaults.SHRINK_PIXELS
        elif self.parent.box_size_pixel > defaults.SHRINK_MIN:
            self.parent.box_size_pixel -= defaults.SHRINK_PIXELS
        else:
            pass
        self.parent.shift_search_boxes(zs,from_dialog=False) 
        
        # Dangerous/maybe doesn't work without new thread
        self.parent.ui.update_ui()
        self.parent.ui.repaint()  

    def convex_hull_robust(self,dynamic_threshold=False):
        # Try random subsamples to omit outliers
        im_smooth = gaussian_filter(self.im,defaults.CENTERING_GAUSS_SD)
        if dynamic_threshold:
            maxn=50
            #ncomponents=np.zeros(maxn)
            for thresh_lower in np.arange(2,maxn):
                # Could also check for the area of the ConvexHull, but components seems good
                im_copy=im_smooth.copy()
                im_copy[im_copy<thresh_lower]=0
                labeled_image, num_components = ndimage.label(im_copy)

                points = np.array( np.where( im_copy ) ).T     # Coords of non-zero points
                hull=ConvexHull(points)
                area=hull.area
                print (num_components,area)
                if num_components<defaults.centering_dynamic_ncomponents and area<defaults.centering_dynamic_area:
                    break # Good. First "few enough" components (around # of spots)
            if thresh_lower>=maxn-1:
                print( "Error: couldn't find good dynamic threshold under %d. Using OTSU."%(maxn) ) # DBG
                im_nonsat = im_smooth[im_smooth<defaults.NONSAT_MAX_OTSU]
                self.thresh_lower = filters.threshold_otsu(im_nonsat)
            else:
                self.thresh_lower = thresh_lower
                print( "Dynamic threshold=%d. Num_components=%d. Area=%d"%(thresh_lower,num_components,area) ) # DBG

            im_smooth[im_smooth<thresh_lower] = 0
        else:
            im_nonsat = im_smooth[im_smooth<defaults.NONSAT_MAX_OTSU]
            cutoff = filters.threshold_otsu(im_nonsat)
            im_smooth[im_smooth<cutoff] = 0
        np.save('ims',im_smooth) # DBG
        np.save('im_raw',self.im) # DBG

        points = np.array( np.where( im_smooth ) ).T     # Coords of non-zero points
        hull = ConvexHull(points) # Entire convex hull. Maybe outliers

        nboots=defaults.centering_convex_robust_nboots
        fraction=defaults.centering_convex_robust_fraction
        sample_size=hull.vertices.shape[0]//fraction
        bootres = np.zeros( (nboots,4))
        samples = np.zeros( (nboots,sample_size))

        for nboot in np.arange(nboots):
            hull_sample = np.random.randint(0,hull.vertices.shape,size=sample_size )
            samples[nboot] = hull_sample
            hull_idxs = hull.vertices[hull_sample]
            fit1 = circle_fitter(hull.points[hull_idxs,1], hull.points[hull_idxs,0] ) # Note dimensions switched!
            fit1.solve()
            opt1 = fit1.params
            logloss=np.log10(fit1.circle_err(opt1 ) )
            bootres[nboot]=np.concatenate( (opt1, [logloss]))

        idxs=np.argsort(bootres[:,3])

        # Sort by log total_error of the sample fit.
        # Make sure we don't get a weird sample that fits too well--
        # We want 5 boots to agree.
        nagree=defaults.centering_convex_robust_nagree
        candidates=np.arange(0,len(bootres))
        stds = np.zeros( (len(candidates),4))
        for nstart in candidates:
            # Examine std of fitted centers and radii
            std_boot1 = np.std( bootres[idxs[nstart:nstart+nagree]], 0)
            stds[nstart] = std_boot1
            #print( bootres[idxs[nstart]], std_boot1)
            # radius of 5 pixels of x and y and radii std of<10
            if (std_boot1[0]**2+std_boot1[1]**2 < 50) and (std_boot1[2]<10):
                break

        best_guess = np.mean( bootres[idxs[nstart:nstart+nagree]],0)
        #print( nstart, best_guess)
        return best_guess,im_smooth

    def auto_center(self):
        if defaults.centering_method=='estimate_boxes':
            # First start small
            pupil_radius_small = float(self.parent.ui.it_start.text()) * self.parent.pupil_mag / 2.0
            self.fit1(pupil_radius_small) # 
            
            # Now go big (maximal based on image), but correct from extrapolated small
            pupil_radius_max_image_pixel = np.sqrt(np.sum( (self.dims/2.0)**2))
            pupil_radius_max_image = pupil_radius_max_image_pixel /1000 * self.parent.ccd_pixel
                
            # Extrapolate based on "N" zernikes
            factor = pupil_radius_small / pupil_radius_max_image
            z_subset = np.zeros( 24 ) # Extrapolate wants at least 20
            self.offline_centroids(conservative_threshold=True)
            num_zs = np.min( (len(z_subset), defaults.ZERNIKES_FOR_INITIAL_CENTERING, len(self.zernikes) ) ) 
            z_subset[0:num_zs] = self.zernikes[0:num_zs]
            z_new =  iterative.extrapolate_zernikes(z_subset, factor)
            
            num_z_possible = np.min( (self.parent.zterms_full.shape[0],len(self.zernikes) ) )
            z_full = np.zeros( self.parent.zterms_full.shape[0] )
            z_full[0:num_z_possible] = z_new[0:num_z_possible]

            self.parent.init_params(
                {'pupil_diam': pupil_radius_max_image * 2.0 / self.parent.pupil_mag} )
            self.parent.make_searchboxes(pupil_radius_pixel=pupil_radius_max_image_pixel)
            self.parent.shift_search_boxes(z_full,from_dialog=False) 
            
            self.offline_centroids(dark_as_nan=False,conservative_threshold=True)

            desired = np.all((self.box_metrics > defaults.BOX_THRESH, np.isnan(self.cenx)==False ), 0) *1.0 # binarize 
            self.desired=desired
            
            guess =[ np.sum( desired*self.parent.box_x / np.sum(desired ) ) ,
                np.sum( desired*self.parent.box_y / np.sum(desired ) ),
                6.5*1000/self.parent.ccd_pixel*self.parent.pupil_mag / 2.0 ] #self.pupil_radius_pixel ]
            self.desired=desired

            opt1=minimize( self.circle_err, guess, method='Nelder-Mead', bounds=[[None,None] ,[None,None], [0, None] ] );
            self.opt1=opt1['x']

            if False: # TODO: DEBUG
                np.savez("desired_%d"%self.parent.ui.offline_curr, desired,
                    self.parent.box_x, self.parent.box_y, self.box_metrics, self.cenx, self.ceny, self.opt1, guess )

            # Find closest box center
            distances =(self.parent.box_x - self.opt1[0])**2 + (self.parent.box_y - self.opt1[1])**2 + (
                100 * np.isnan(self.cenx)) # 100=Hack to exclude NaN boxes 
            r_pix = self.opt1[2] - self.parent.box_size_pixel/2 # remove half a box
        elif defaults.centering_method=='convex_hull':
            im_smooth = gaussian_filter(self.im,defaults.CENTERING_GAUSS_SD)
            im_nonsat = im_smooth[im_smooth<defaults.NONSAT_MAX_OTSU]
            cutoff = filters.threshold_otsu(im_nonsat)
            #print( cutoff ) # DBG
            im_smooth[im_smooth<cutoff] = 0
            np.save('ims',im_smooth) # DBG
            np.save('im_raw',self.im) # DBG
            points = np.array( np.where( im_smooth ) ).T     # Coords of non-zero points
            hull = ConvexHull(points)
            fit1 = circle_fitter(hull.points[hull.vertices,1], hull.points[hull.vertices,0] ) # Note dimensions switched!
            fit1.solve()
            self.opt1 = fit1.params

            r_pix = self.opt1[2]
            # Find closest box center
            distances =(self.parent.box_x - self.opt1[0])**2 + (self.parent.box_y - self.opt1[1])**2
        elif 'convex_hull_robust' in defaults.centering_method:
            # Try random subsamples to omit outliers:
            # Get best consensus from random sample of convex hull inliers
            if 'dynamic' in defaults.centering_method:
                do_dynamic=True
            else:
                do_dynamic=False
            self.opt1,im_smooth = self.convex_hull_robust(do_dynamic)
            r_pix = self.opt1[2]
            # Find closest box center
            distances =(self.parent.box_x - self.opt1[0])**2 + (self.parent.box_y - self.opt1[1])**2

        box_min = np.argmin( distances )

        self.parent.ui.cx = self.parent.box_x[box_min]
        self.parent.ui.cy = self.parent.box_y[box_min]
        self.cx_best = self.parent.box_x[box_min]
        self.cy_best = self.parent.box_y[box_min]

        # TODO: Figure out more correct diameter using max outermost box corner
        p_diam = r_pix*2.0/1000.0*self.parent.ccd_pixel

        if self.parent.ui.it_stop_dirty: # If edited in the UI, override.
            p_diam =  float( self.parent.ui.it_stop.text() ) * self.parent.pupil_mag
            print("Dirty:", p_diam)
        elif p_diam > defaults.ITERATIVE_PUPIL_STOP * self.parent.pupil_mag: # Never exceed max.
            p_diam = defaults.ITERATIVE_PUPIL_STOP * self.parent.pupil_mag
            print("TOO BIG:", p_diam)
        else: # Or, use the estimated value
            print("Normal auto:", p_diam)

        # Size on the sensor, max pixel radius in the image
        self.iterative_max = p_diam
        self.iterative_max_pixels = self.iterative_max/2.0 * 1000 / self.parent.ccd_pixel

        if 'convex_hull' in defaults.centering_method:
            crop_left=int( np.max( (0,self.cx_best-r_pix)) )
            crop_top=int( np.max( (0,self.cy_best-r_pix)) )
            crop_right=int( np.min( (self.dims[1],self.cx_best+r_pix)) )
            crop_bottom=int( np.min( (self.dims[0],self.cy_best+r_pix)) )
            self.im_smooth_cropped = im_smooth[crop_top:crop_bottom,crop_left:crop_right]
            np.save('im_crop',self.im_smooth_cropped) # DBG

    def offline_startbox(self):
        try:
            self.parent.box_x # Needed
        except:
            self.parent.ui.mode_init()

        self.iterative_size = float(self.parent.ui.it_start.text()) * self.parent.pupil_mag
        self.iterative_size_pixels = self.iterative_size/2.0 * 1000 / self.parent.ccd_pixel
        self.auto_center()
        # Show the computed max in the box:
        if not self.parent.ui.it_stop_dirty:
            self.parent.ui.it_stop.setText('%2.2f'%(self.iterative_max/ self.parent.pupil_mag) )
        self.parent.ui.mode_offline=True
        self.offline_reset()
        
    def iterative_offline(self):
        pass

    def iterative_step_good(self):
        self.iterative_size_pixels = self.iterative_size/2.0 * 1000 / self.parent.ccd_pixel
        self.offline_stepbox()
        self.parent.ui.line_pupil_diam.setText('%2.2f'%(self.iterative_size / self.parent.pupil_mag) ) #+step) )

    def iterative_run_good(self):
        self.offline_startbox()

        #self.engine.offline.iterative_max_pixels = float(self.it_stop.text())/2.0 * 1000 / self.engine.ccd_pixel
        while self.iterative_size_pixels < self.iterative_max_pixels:
            self.iterative_step_good()
            self.parent.ui.update_ui()
            self.parent.ui.repaint()

            s="Frame %02d/%02d; %04d boxes. %04d zern terms. Pupil: %02.2f/%02.2f"%(self.parent.ui.offline_curr, self.max_frame,
                                                                                    self.parent.num_boxes, self.parent.zterms_full.shape[0], self.iterative_size, self.iterative_max )
            print(s,flush=True)
            

    def offline_navigate(self):
        self.saver.load1(self.parent.ui.offline_curr)

    def offline_goodbox(self,nframe):
        nbox=self.parent.ui.box_info

        GOOD_THRESH=0.25 # TODO
        patch=self.parent.ui.box_pix
        self.good_template=self.metric_patch(self.parent.ui.box_pix)
        #self.good_idx=np.where( self.good_template>GOOD_THRESH)[0][0]
        self.good_idx=int(len(self.good_template)*0.6) # TODO
        #print("Goodbox", nbox,nframe,self.good_idx, patch.shape, self.box_size_pixel)

    def show_dialog_debug(self):
        self.parent.ui.offline_dialog.sc.axes.plot( self.box_metrics, 'x-', label='rands')
        self.parent.ui.offline_dialog.sc.axes.axhline( defaults.BOX_THRESH, color='r' )
        self.parent.ui.offline_dialog.sc.axes.legend(loc='best', fontsize=16)
        #self.parent.ui.offline_dialog.sc.axes.set_xlabel('Frame #', fontsize=16)
        self.parent.ui.offline_dialog.sc.axes.set_ylim([-10, defaults.BOX_THRESH*2] )
        self.parent.ui.offline_dialog.sc.axes.grid()
        self.parent.ui.offline_dialog.sc.draw()
        self.parent.ui.offline_dialog.show()
        
    def show_dialog(self):
        self.parent.ui.offline_dialog.sc.axes.clear();

        diams=np.array([self.saver.data[key1]['pupil_diam'] for key1 in self.saver.data.keys()])
        zerns=np.array([self.saver.data[key1]['zernikes'][0:5] for key1 in self.saver.data.keys()])

        diams = diams / self.parent.pupil_mag # Convert to real "pupil space", not "sensor space"
        
        radius2 = (diams/2) ** 2
        sqrt3=np.sqrt(3.0)
        sqrt6=np.sqrt(6.0)
        z3=zerns[:,3-1]
        z4=zerns[:,4-1]
        z5=zerns[:,5-1]

        J45 =  (-2.0 * sqrt6 / radius2) * z3
        J180 = (-2.0 * sqrt6 / radius2) * z5
        cylinder = (4.0 * sqrt6 / radius2) * np.sqrt((z3 * z3) + (z5 * z5))
        sphere = (-4.0 * sqrt3 / radius2) *z4 - 0.5 * cylinder

        self.parent.ui.offline_dialog.sc.axes.plot( J45, 'x-', label='J45')
        self.parent.ui.offline_dialog.sc.axes.plot( J180, 's-', label='J180')
        self.parent.ui.offline_dialog.sc.axes.plot( sphere, 'o-', label='Sphere')
        self.parent.ui.offline_dialog.sc.axes.legend(loc='best', fontsize=16)
        self.parent.ui.offline_dialog.sc.axes.set_xlabel('Frame #', fontsize=16)
        self.parent.ui.offline_dialog.sc.axes.set_ylabel('Diopters', fontsize=16)
        self.parent.ui.offline_dialog.sc.axes.grid()
        self.parent.ui.offline_dialog.sc.draw()
        self.parent.ui.offline_dialog.show()

"""    # @jit(nopython=True)
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
"""

"""
For each frame:
- Center x,y
- Radius
- Box positions (box_x, box_y), references: 
- Centroids

"""

        
'''
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

            self.parent.make_searchboxes(cx,cy,pupil_radius_pixel=self.iterative_size/2.0*1000/self.ccd_pixel)
            self.parent.init_params( {'pupil_diam': self.iterative_size / self.parent.pupil_mag})

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
'''

            
        
# This was in online_stepbox, after completion (shrink!)        
'''
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
'''   
