import numpy as np
import sys
import os
import time

import matplotlib.cm as cmap
#from numba import jit
from numpy.linalg import svd,lstsq,pinv
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
from offline import NextwaveOffline

#import ffmpegcv # Read AVI... Better than OpenCV (built-in ffmpeg?)

from PIL import Image, TiffImagePlugin

WINDOWS=(os.name == 'nt')


NACT_PER_NZERN=4

GAUSS_SD=3
BOX_THRESH=2.5

OFFLINE_ITERATIVE_START=3.0
OFFLINE_ITERATIVE_STEP=0.25


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
        self.offline = NextwaveOffline(self)
        self.num_boxes = 0
        self.zernikes = None
        self.defocus = 0
        self.aogain = 1.0
        self.dmfill = 1.0
        self.omits = np.zeros( 0, dtype='uint8' ) # Need to do early
        
        self.acts_inside = np.arange(97,dtype='int') # TODO
        self.acts_outside = np.array([],dtype='int')
        self.modes = 97
        self.condition = 9999
        
    def init(self):
        if not self.ui.offline_only:
            self.comm.init()
            self.comm.set_mode(0)
        self.init_params()

    def init_params(self, overrides=None):
    
        print("Init params")

        #self.ccd_pixel = self.ui.get_param("system","pixel_pitch",True)
        #self.pupil_diam = self.ui.get_param("system","pupil_diam",True)
        #self.box_um = self.ui.get_param("system","lenslet_pitch",True)
        #self.focal = self.ui.get_param("system","focal_length",True)
        self.focal =     self.ui.get_param_xml("LENSLETS_LensletFocalLength")/1000.0
        self.box_um =    self.ui.get_param_xml("LENSLETS_LensletPitch")
        self.ccd_pixel = self.ui.get_param_xml("CAMERA1_CameraPixelPitch")
        self.pupil_diam =self.ui.get_param_xml("OPTICS_PupilDiameter")
        self.pupil_mag =self.ui.get_param_xml("OPTICS_PupilMagnificationFactor")

        if overrides:
            self.pupil_diam=overrides.get('pupil_diam',self.pupil_diam)

        # New method, not used much yet:
        self.params = OpticsParams(self.ccd_pixel, self.pupil_diam, self.box_um, self.focal)

        self.pupil_diam = self.pupil_diam * self.pupil_mag

        self.pupil_radius_mm=self.pupil_diam / 2.0
        self.pupil_radius_pixel=self.pupil_radius_mm * 1000 / self.ccd_pixel
        self.box_size_pixel=self.box_um / self.ccd_pixel

        if overrides:
            self.box_size_pixel=overrides.get('box_size_pixel',self.box_size_pixel)

        self.ri_ratio = self.pupil_radius_pixel / self.box_size_pixel
        #print( "Init:", self.box_size_pixel, self.pupil_radius_pixel, self.ri_ratio )

        bytez =np.array([self.ccd_pixel, self.box_um, self.pupil_radius_mm], dtype='double').tobytes() 

        self.send_params_to_engine()

    def send_params_to_engine(self):
        if not self.ui.offline_only:
            self.comm.write_params(True) # "Overrides" : FIXME!! TODO

    def move_searchboxes(self,dx,dy):
        self.box_x += dx
        self.box_y += dy
        self.ref_x += dx
        self.ref_y += dy
        self.initial_x += dx # TODO: confirm good to move these
        self.initial_y += dy # TODO: confirm good to move these

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
        else:
            self.pupil_radius_pixel = pupil_radius_pixel
            self.pupil_radius_mm = self.pupil_radius_pixel / 1000.0 * self.ccd_pixel

        box_size_pixel=self.box_size_pixel
        if box_spacing_pixel is None:
            box_spacing_pixel=box_size_pixel

        ri_ratio = pupil_radius_pixel / box_spacing_pixel

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
        self.centroids_x = np.full(num_boxes, np.nan)
        self.centroids_y = np.full(num_boxes, np.nan)
        
        self.box_spacing_pixel = box_spacing_pixel
        
        self.update_zernike_svd() # Precompute

        print( "Make SB ",pupil_radius_pixel, box_size_pixel, box_spacing_pixel, ri_ratio, num_boxes, self.zterms_full.shape )

        # Determine neighbors (for nan interpolation)
        self.neighbors = np.zeros( (self.num_boxes, 4), dtype='int32')
        for nidx in np.arange(num_boxes):
            distances = (self.box_x - self.box_x[nidx])**2 + (self.box_y - self.box_y[nidx])**2
            self.neighbors[nidx]=np.argsort( distances)[1:5] # Take 4 nearest, excluding self (which will be 0)

        self.update_searchboxes(send_to_engine=True)
        
        self.omits = np.zeros( num_boxes, dtype='uint8' )
        try:
            #fil=open("omits.txt","rt")
            #lins=fil.readlines();
            omits = np.loadtxt("omits.txt", dtype='int')
            print( omits )
            for omit1 in omits:
                self.omits[omit1]=1
        except:
            pass
        
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

    def offline_frame(self,nframe):
        self.offline.offline_frame(nframe)

    def update_searchboxes(self,send_to_engine=True):
        self.update_zernike_svd()
        if not self.ui.offline_only:
            self.update_influence();
            if send_to_engine:
                self.comm.send_searchboxes(self.box_x, self.box_y)
        #self.dump_vars() # DEBUGGING

    def update_zernike_svd(self):
        lefts =    self.norm_x - 0.5/self.ri_ratio # (or 0.5 * self.box_size_pixel/self.pup)
        rights =   self.norm_x + 0.5/self.ri_ratio
        ups =    -(self.norm_y + 0.5/self.ri_ratio)
        downs =  -(self.norm_y - 0.5/self.ri_ratio)

        # Compute all integrals for all box corners 
        lenslet_dx,lenslet_dy=zernike_functions.zernike_integral_average_from_corners(
            lefts, rights, ups, downs, self.pupil_radius_mm / self.pupil_mag )

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
                #print("Est. max zs:%d, Max Z (order):%d"%(nmax_from_boxes, valid_max) )
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
            zterms_inv = pinv(zterms)

            if nsubset==0:
                self.zterms_full=zterms
                self.zterms_full_inv=zterms_inv
                zpoly_full = zpoly.copy()
            elif nsubset==1:
                self.zterms_20=zterms
                self.zterms_20_inv=zterms_inv
                zpoly_20 = zpoly.copy()

        # DBG:
        self.lenslet_dx=lenslet_dx # Debugging
        self.lenslet_dy=lenslet_dy
        np.save('zterms_full.npy',self.zterms_full)
        np.save('zterms_full_inv.npy',self.zterms_full_inv)
        np.save('zterms_20.npy',self.zterms_20)
        np.save('zterms_20_inv.npy',self.zterms_20_inv)
        np.save('zpoly_20.npy',zpoly_20)
        np.save('zpoly_full.npy',zpoly_full)

    def update_influence(self):
        # if False: # Load Miniwave's inverse directly
            # influence = np.loadtxt(self.ui.json_data["params"]["influence_file"], skiprows=0)
            # self.influence[ 0:influence.shape[0] ] = influence
            # self.influence_inv = self.influence.T
            # self.influence = pinv( self.influence_inv ) # New way: read the interaction
        # except:

            # influence = np.random.normal ( loc=0, scale=0.01, size=(97, self.num_boxes * 2)  )
        filename_influence =self.ui.get_param_xml("MIRROR1_MirrorInfluenceMatrix") # TODO: Don't reload every time
        self.influence_full = np.loadtxt(filename_influence, skiprows=1)
        #nonzero_idxs = np.sum(influence_full**2,0)>0 
        #self.influence = influence_full[:,nonzero_idxs]

        max_boxes = np.sqrt( self.influence_full.shape[1] / 2.0 )
        boxes_x = np.arange(-(max_boxes-1)/2,(max_boxes-1)/2+1) # +1 to include +max_boxes number
        boxes_y = np.arange(-(max_boxes-1)/2,(max_boxes-1)/2+1)
        
        # Determine outer edge of each box using corners away from the center:
        # 0.5*sign: positive adds 0.5, negative substracts 0.5
        XX,YY = np.meshgrid(boxes_x, boxes_y )
        aperture = self.pupil_radius_pixel / self.box_spacing_pixel * 1.0
        RR = np.sqrt( (XX+0.5*np.sign(XX))**2 + (YY+0.5*np.sign(YY))**2 )
        valid_boxes = np.where(RR.flatten()<aperture)[0]
        # For each valid index, need to make both the X and Y component
#        valid_indices = np.concatenate( (valid_boxes*2, valid_boxes*2+1) )
        valid_indices = np.vstack( (valid_boxes*2, valid_boxes*2+1) ).T.flatten()       
        self.influence = self.influence_full[:,valid_indices]
        np.save('infl', self.influence )
        
        if len(self.acts_outside)>0:
            self.influence[self.acts_outside,:] = 0 # Zero them out, if any
        U, s, V = np.linalg.svd(self.influence, full_matrices=True)
        s_recip = 1/s
        s_recip[ s<1e-10 ] = 0 # Set any tinies (or 0) to zero
        
        print( max_boxes, len(valid_boxes), self.influence.shape, s.max(), valid_boxes[0:10], RR.shape )

        if self.modes < len(s):
            s_recip[self.modes:] = 0 # Clear all the modes from x up
            
        self.condition = s[0]/s[modes-1]
            
        self.influence_inv = ( (U * s_recip) @ V[0:97,:] ).T
        self.influence_inv[:,self.acts_outside] = 0

        self.nActuators=self.influence.shape[0]
        self.nTerms=self.influence.shape[1]
        np.save('influ', self.influence_inv )

        self.ui.offline_dialog.sc.axes.clear();
        self.ui.offline_dialog.sc.axes.plot(np.arange(len(s))+1, s[0]/s, 'o-')
        self.ui.offline_dialog.sc.axes.axvline( self.modes, color='r' )
        self.ui.offline_dialog.sc.axes.set_ylim(0, s[0]/s[s>1e-10][-1]*1.25 )
        self.ui.offline_dialog.sc.draw()
        self.ui.offline_dialog.show()

    def compute_zernikes(self):
        # find slope
        spot_displace_x =   self.ref_x - self.centroids_x
        spot_displace_y = -(self.ref_y - self.centroids_y)

        # Not sure whether should do this:
        #spot_displace_x -= spot_displace_x.mean()
        #spot_displace_y -= spot_displace_y.mean()
        #print( spot_displace_y.mean(), spot_displace_x.mean() )
        self.mean_displacements = [np.nanmean(spot_displace_x), np.nanmean(spot_displace_y) ]

        self.spot_displace_interpolated = np.zeros( self.num_boxes )

        for nidx in np.arange(self.num_boxes):
            if np.isnan(spot_displace_x[nidx]) or np.isnan(spot_displace_y[nidx]):
                spot_displace_x[nidx] = np.nanmean( spot_displace_x[self.neighbors[nidx]] )
                spot_displace_y[nidx] = np.nanmean( spot_displace_y[self.neighbors[nidx]] )
                self.spot_displace_interpolated[nidx] = 1

            if np.isnan(spot_displace_x[nidx]) or np.isnan(spot_displace_y[nidx]):
                #print( "Still nan", nidx, self.neighbors[nidx])
                spot_displace_x[nidx]=0
                spot_displace_y[nidx]=0
                self.spot_displace_interpolated[nidx] = 2

        slope = np.concatenate( (spot_displace_y, spot_displace_x)) * (self.ccd_pixel/self.focal);

        self.spot_displace_x = spot_displace_x # debugging
        self.spot_displace_y = spot_displace_y
        self.slope = slope

        coeff=np.matmul(self.zterms_full,slope)
        self.zernikes=coeff[zernike_functions.CVS_to_OSA_map[0:len(coeff)]] # Return value will is OSA

        #print ("CompZ spot means:",np.mean(self.spot_displace_x), np.mean(self.spot_displace_y))

    def autoshift_searchboxes(self):
        #shift_search_boxes(self,zs,from_dialog=True):
        pass
        #return

    def get_deltas(self,zs,from_dialog,full=True):
        zs=np.array(zs)
        if len(zs)==20:
            mat1=self.zterms_20_inv
        else:
            mat1=self.zterms_full_inv

        valid_idxs=np.arange(0,mat1.shape[1] )
        idxs_remap  = zernike_functions.OSA_to_CVS_map[valid_idxs]
        zern_new=zs[valid_idxs] [idxs_remap ]
        #zern_new[0:2]=0 # Remove tip/tilt TODO: Need to move the pupil center?

        delta=np.matmul(mat1,zern_new)
        num_boxes = self.box_x.shape[0]
        delta_y = delta[0:num_boxes] / (self.ccd_pixel/self.focal)
        delta_x = delta[num_boxes:] / (self.ccd_pixel/self.focal)
        self.delta_x=delta_x
        self.delta_y=delta_y
        self.zern_new = zern_new
        self.zs = zs

        np.savetxt('zs_delta_x.txt',delta_x );
        np.savetxt('zs_delta_y.txt',delta_y );
        
        return delta_x,delta_y

    def do_ao1(self):
        self.comm.set_mode( 
            self.comm.read_mode() | 0x20 )  # MODE_FORCE_AO_START TODO
            
    def defocus_do(self):
        zs= np.zeros(20)
        zs[3] = self.defocus
        
        if False: # For open loop, doesn't work (not linear)
            dx,dy = self.get_deltas(zs, False)
            
            # As Reference Shift:
            #self.ref_x = self.initial_x - dx
            #self.ref_y = self.initial_y + dy
            #self.comm.send_searchboxes(self.box_x, self.box_y)
            
            # Method2: Take current deltas, then add futher by dx
            dx -= np.mean(dx)
            dy -= np.mean(dy)
            deltas1 = np.vstack( (dx,-dy) ).T.flatten()
            deltas1 /= (self.focal*1000/self.ccd_pixel)
            mirs = np.matmul ( deltas1, self.influence_inv )

            #print( np.mean( mirs), mirs[0:10] )
            self.comm.write_mirrors_offsets(-mirs)
            #print(self.defocus)
        else:
            self.shift_references(zs,False)
            self.comm.send_searchboxes(self.box_x, self.box_y)
            self.do_ao1()
            
    def defocus_start(self):
        # One idea was to save a local copy of the "current" centroids, and use those in entire
        # inverse calculations. Think it's not necessary though; can just add the mirror deltas.
        self.centroids_x_save = self.centroids_x 
        self.centroids_y_save = self.centroids_y
    
    def defocus_plus01(self):
        self.defocus += 0.01
        self.defocus_do();
    def defocus_plus10(self):
        self.defocus += 0.1
        self.defocus_do();
    def defocus_minus01(self):
        self.defocus -= 0.01
        self.defocus_do();
    def defocus_minus10(self):
        self.defocus -= 0.1
        self.defocus_do();

    def defocus_plus(self):
        self.defocus += 0.01
        self.defocus_do();
        
    def defocus_minus(self):
        self.defocus -= 0.01
        self.defocus_do();
        
    def defocus_set(self,val):
        self.defocus = val
        self.defocus_do();

    def aogain_set(self,val):
        self.aogain = val
        self.aogain_do();

    def modes_set(self,val):
        self.modes = int( val )

    def dmfill_set(self,val):
        self.dmfill = val
        filename_pattern =self.ui.get_param_xml("MIRROR1_MirrorPatternArrayFile") # TODO: Don't reload every time
        self.act_positions = np.loadtxt(filename_pattern) 
        
        act_norm_x = (self.act_positions[:,1]-0.5)*2
        act_norm_y =(1-self.act_positions[:,0]-0.5)*2 # Notsure need to negate
        r=np.sqrt( act_norm_x**2 + act_norm_y**2)
        self.acts_inside = np.where( r<= val )[0]
        self.acts_outside = np.where( r>val )[0]
    
    def aogain_do(self):
        self.ui.sockets.centroiding.send(b"G%f\x00"%(self.aogain) )

    def shift_search_boxes(self,zs,from_dialog=True):
        #print( zs )
        dx,dy = self.get_deltas(zs,from_dialog)
        self.box_x = self.initial_x - dx
        self.box_y = self.initial_y + dy
        #self.ui.move_center((round(zs[1]),round(zs[0]),do_update=False)
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
        self.ref_x = self.initial_x
        self.ref_y = self.initial_y

    def receive_image(self):
        return self.comm.receive_image()
    def receive_centroids(self):
        if self.ui.mode_offline==False: # If in offline, don't keep grabbing centroids from C++ engine
            return self.comm.receive_centroids()

    def flat_do(self):
        self.comm.flat_do()

    def zero_do(self):
        self.comm.zero_do()

    def load_mirror_file(self,fname):
        mirrors = np.loadtxt(fname, skiprows=1)
        self.comm.write_mirrors( mirrors )
        
    def flat_do(self):
        self.comm.flat_do()
        
    def loop_changed(self,is_checked):
        if is_checked:
            if self.comm.read_mode() == 4:
                self.comm.set_mode(4+8)
        else:
            if self.comm.read_mode() == 4+8:
                self.comm.set_mode(4)
            
    def flat_save(self):
        self.comm.flat_save()

    def send_quit(self):
        self.comm.set_mode(255) # TODO: Get from file

    def mode_init(self):
        self.mode=1
        self.init_params(False)
        self.mode_snap()
        #time.sleep(0.1)

    def mode_snap(self, reinit=True, allow_AO=True):
        if self.ui.mode_offline:
            self.offline.offline_centroids()
            return

        if reinit:
            #self.init_params()
            self.update_searchboxes()
            
        self.send_params_to_engine()

        val=(2+8) if (self.ui.chkLoop.isChecked() and allow_AO) else 2
        self.mode=2 # Different mode?? TODO
        self.comm.set_mode(val)

    def mode_run(self, reinit=True, numruns=1):
        #if reinit:
        #    self.init_params()
        self.send_params_to_engine()

        val= np.array( numruns, dtype='uint64' )
        #val= np.array( self.ui.edit_num_runs.text(), dtype='uint64' )
        #buf.append(val.tobytes())
        self.comm.set_nframes(val)

        val=(4+8) if self.ui.chkLoop.isChecked() else 4
        self.mode=4 # ?? TODO
        self.comm.set_mode(val)

    def mode_stop(self):
        self.comm.set_mode(0)

    def export_centroids(self,centroids_filename):
        fil = open(centroids_filename,'wt')

        fil.writelines( '[image size = %dx%d]\n'%(self.image_bytes.shape[0],self.image_bytes.shape[1]))
        fil.writelines( '[pupil = %f,%d,%d]\n'%(self.pupil_diam/self.pupil_mag,self.ui.cx,self.ui.cy))
        for nbox in np.arange( len(self.ref_x)):
            # TODO: Valid or invalid
            fil.writelines('%d\t%.12f\t%.12f\t%.12f\t%.12f\t\n'%(
                (self.spot_displace_interpolated[nbox]==0)*1,self.ref_x[nbox], self.ref_y[nbox],
                           self.centroids_x[nbox], self.centroids_y[nbox] ) )
        fil.close()

        filename="zernikes.txt"
        fil = open(filename,'wt')
        for val in self.zernikes:
            fil.writelines('%f\n'%val)
        fil.close()

        np.save('dx',self.spot_displace_x)
        np.save('dy',self.spot_displace_y)
        np.save('slope',self.slope)
        #np.save('zterms_full',self.zterms_full)
        #np.save('zterms_inv',self.zterms_inv)

        try:
            np.savetxt('mirror_voltages.txt',self.comm.mirror_voltages)
        except:
            pass

        self.dump_vars()
