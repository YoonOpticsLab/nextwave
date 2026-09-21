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

from image_helpers import detect_rotation

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
import occupancy

from nextwave_comm import NextwaveEngineComm
import defaults


# Get the directory where your script is running
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the current directory to Windows PATH for this session
os.environ["PATH"] += os.pathsep + current_dir

# Now import your ffmpeg library
import ffmpeg 

import ffmpegcv # Read AVI... Better than OpenCV (built-in ffmpeg?)
from PIL import Image, TiffImagePlugin # Needed

from PyQt5.QtCore import QObject, pyqtSignal

class OfflineSignals(QObject):
    """ Notifications from the offline routines, which can run in a worker QThread.
        These are one-way, for display only: the algorithm keeps its own state in member variables
        and never reads anything back from the UI while running (see NextwaveOffline.__init__).
        The main window connects them to slots (see connect_offline_signals).
    """
    pupil_diam_changed = pyqtSignal(float) # Current pupil diameter (user units)
    pupil_stop_changed = pyqtSignal(float) # Computed max pupil diameter for iterating (user units)
    box_size_changed = pyqtSignal(float)   # Box size (pixels)
    frame_loaded = pyqtSignal(object)      # Image (numpy array) of frame just loaded
    load_progress = pyqtSignal(str, int, int) # While loading a movie: what's being done, n done, n total (total 0: no count)
    first_frame = pyqtSignal(object)       # The first frame of a movie being loaded (numpy array), as soon as it's read

    def __init__(self):
        super().__init__()
        self._last_report = 0.0

    def report(self, stage, done=0, total=0):
        """ Emit load_progress, but no more often than it can be shown: a signal per frame would cost more than it's worth. """
        now = time.monotonic()
        if done == 0 or done >= total or now - self._last_report > 0.05: # Always the first and last of a stage
            self._last_report = now
            self.load_progress.emit(stage, done, total)

def limit_zernikes(zernikes):
    """ Copy of the Zernike coefficients with any whose magnitude is above defaults.ZERNIKE_MAX_ABS set to NaN.
        (Only for saving/exporting: the algorithm itself needs the real values.) """
    limit = float(getattr(defaults, 'ZERNIKE_MAX_ABS', 10.0) or 0)
    zernikes = np.array(zernikes, dtype=float) # A copy
    if limit > 0:
        with np.errstate(invalid='ignore'): # (Already-NaN ones)
            zernikes[np.abs(zernikes) > limit] = np.nan
    return zernikes

# Bits of a frame's FLAGS value (saved with the results, and a column of the exported CSV). 0: nothing to flag.
FLAG_FLASH = 1 # The frame's mean brightness is over defaults.FLASH_RATIO_THRESHOLD times the movie's median: a flash
# (Next flag: 2, then 4, ... so they can be combined)
MAX_EXPORT_ZERNIKES = 65 # Zernike columns (Z1..) of the exported CSV

def classify_frames(movie, flash_ratio, dark_ratio, peak_ratio=0):
    """ (flash frames, dark frames): sets of frame numbers, by mean brightness relative to the movie's median.
        A flash lights up the whole image. Flash frames are over flash_ratio times the median (a strong one is about 60-460x
        on our movies; normal frames stay within 2x); a run of them (within 2 frames of each other) is a flash only if its
        brightest frame is over peak_ratio times the median, so a slow, moderate brightening isn't taken for one.
        A "much darker" frame (eye closed, lights off) is under dark_ratio times the median.
        A ratio of 0 turns that one off. """
    if len(movie) < 3 or not (flash_ratio > 0 or dark_ratio > 0):
        return set(), set()
    means = np.asarray(movie).mean(axis=(1,2), dtype=np.float64)
    median = np.median(means)
    if not median > 0:
        return set(), set()
    ratio = means / median
    flash = set()
    if flash_ratio > 0:
        runs = []
        for n in np.where(ratio > flash_ratio)[0]:
            if runs and n - runs[-1][-1] <= 2:
                runs[-1].append(int(n))
            else:
                runs.append([int(n)])
        for run in runs:
            if not peak_ratio > 0 or ratio[run].max() > peak_ratio:
                flash.update(run)
    dark = {int(n) for n in np.where(ratio < dark_ratio)[0]} if dark_ratio > 0 else set()
    return flash, dark

class info_saver():
    def __init__(self,parent):
        self.parent=parent
        self.offline=parent
        self.engine=parent.parent
        self.ui=self.engine.ui
        self.data = {}

    def save1(self,nframe):
        flash = nframe in self.offline.flash_frames
        dark = nframe in self.offline.dark_frames
        zernikes = limit_zernikes(self.engine.zernikes)
        centroids_x, centroids_y = self.engine.centroids_x, self.engine.centroids_y
        est_x, est_y = self.offline.est_x, self.offline.est_y
        flags = 0
        if flash:
            flags |= FLAG_FLASH
        dark_box_fraction = float(np.mean(np.isnan(centroids_x))) if len(centroids_x) else 0.0 # Boxes that found no spot
        limit = float(getattr(defaults, 'DARK_BOX_FRACTION_THRESHOLD', 0) or 0)
        if flash or dark: # A flash, or much too dark: nothing measured from the image means anything. NaN
            zernikes = np.full(np.shape(zernikes), np.nan)
            centroids_x, centroids_y, est_x, est_y = [np.full(np.shape(a), np.nan) for a in (centroids_x, centroids_y, est_x, est_y)]
        elif limit > 0 and dark_box_fraction > limit: # Too few boxes with a spot for a fit. The centroids that were found are real
            zernikes = np.full(np.shape(zernikes), np.nan)
            est_x, est_y = [np.full(np.shape(a), np.nan) for a in (est_x, est_y)] # (Made from the Zernikes)
        data_record = {
            'box_x':self.engine.box_x,
            'box_y':self.engine.box_y,
            'ref_x':self.engine.ref_x,
            'ref_y':self.engine.ref_y,
            'centroids_x':centroids_x,
            'centroids_y':centroids_y,
            'est_x':est_x,
            'est_y':est_y,
            'cx':self.engine.cx,
            'cy':self.engine.cy,
            'rotation':self.offline.rotations[nframe],
            'pupil_diam':self.engine.pupil_diam / self.engine.pupil_mag, # In pupil coords, not sensor
            'box_size_pixel':float(self.engine.box_size_pixel), # Final search box size, pixels (the boxes shrink while processing)
            'flags':flags, # See FLAG_*. (Not in results saved by older versions.)
            'dark_box_fraction':dark_box_fraction, # Fraction of the boxes that found no spot (NaN centroid)
            'zernikes':zernikes}
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
        self.engine.cx = data_record['cx']
        self.engine.cy = data_record['cy']
        self.engine.pupil_diam = data_record['pupil_diam'] * self.engine.pupil_mag  # TODO: Should we rebuild boxes ?
        self.engine.zernikes = data_record['zernikes']
        if 'box_size_pixel' in data_record: # (Results saved by older versions don't have it: leave the box size as is)
            self.engine.box_size_pixel = data_record['box_size_pixel']
            self.offline.signals.box_size_changed.emit( float(self.engine.box_size_pixel) )

        self.engine.num_boxes = len( self.engine.centroids_x)
        #self.offline.rotations[self.ui.offline_curr]=data_record['rotation']

        self.offline.signals.pupil_diam_changed.emit( self.engine.pupil_diam / self.engine.pupil_mag )

        return data_record

    def printable1(self,nframe,first_frame=0):
        """ One line of the exported CSV. first_frame: the first frame exported (see export_all_zernikes): it is exported frame 1, time 0 """
        data_record=self.load1(nframe)
        n_export = nframe - first_frame + 1 # 1-based, counting from the first frame exported
        t_sec = (n_export-1) / float(getattr(defaults,'FRAME_RATE',100.0))
        if not data_record is None:
            try:
                s=("%s,%s,%s,%0.3f,%0.2f,%0.3f,%d,%d,%d,")%(self.offline.sub_id,self.offline.scan_dir,self.offline.fnames[nframe],t_sec,
                defaults.scan_frame_to_ecc[self.offline.scan_dir][nframe],data_record['pupil_diam'],data_record['cx'],data_record['cy'],
                data_record.get('flags',0))
            except: # without the sub_id params
                s=("%s,%s,%d,%0.3f,%0.2f,%0.3f,%d,%d,%d,")%("","",n_export,t_sec,0.0,data_record['pupil_diam'],data_record['cx'],data_record['cy'],
                data_record.get('flags',0))
            for nz1,z1 in enumerate(limit_zernikes(data_record['zernikes'])): # (Also for results saved before this rule)
                s += "%0.6f,"%(z1)
        else:
            s=',,%d,%0.3f,'%(n_export,t_sec) + ','*(5+MAX_EXPORT_ZERNIKES) # No data (e.g. padding before the movie starts): the rest empty
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
        self.signals = OfflineSignals()
        self.saver = info_saver(self)

        # Algorithm inputs. The UI updates these when the user edits them; the algorithm only reads them here.
        self.it_start = float(defaults.ITERATIVE_PUPIL_START) # Starting pupil diameter
        self.it_step = float(defaults.ITERATIVE_PUPIL_STEP_SIZE)
        self.it_stop = float(defaults.ITERATIVE_PUPIL_STOP) # Max pupil diameter. Holds last computed value unless user-edited
        self.it_stop_dirty = False # User has set it_stop: use it instead of estimating
        self.center_dirty = False # User has set the center: don't autocenter

        self.skip_enlarge = bool(getattr(defaults, 'ITERATIVE_SKIP_ENLARGE', 0)) # Don't grow the pupil in steps: start at the max
        self.autocenter_enabled = bool(getattr(defaults, 'ITERATIVE_AUTOCENTER', 1)) # Find the pupil center in each frame
        self.fixed_center = bool(getattr(defaults, 'ITERATIVE_FIXED_CENTER', 0)) # Same center for every frame: engine.cx, cy stay put

        forced_stop = float(getattr(defaults, 'ITERATIVE_PUPIL_STOP_FORCE', 0) or 0) # getattr: user's defaults file may predate this
        if forced_stop > 0: # Same as the user typing a max pupil into the UI
            self.it_stop = forced_stop
            self.it_stop_dirty = True

        self.offline_curr = 0 # Current frame

        self.flash_frames = set() # Frames that are flashes, and frames that are much darker than usual (see update_frame_classes)
        self.dark_frames = set()

        # Write debug arrays (ims.npy, etc.) to the current directory. Off in the parallel workers, which would collide.
        self.debug_dumps = True

        self._movie_for_ui = None # Set by the loaders: the movie for the frame list. See finish_load
        self.occupancy_template = None # For centering_method 'occupancy_match': see prepare_occupancy_template
        
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

    def offline_frame(self,nframe=None):
        if nframe is None:
            nframe = self.offline_curr
        #dims=np.zeros(2,dtype='uint16')
        #dims[0]=self.offline_movie[nframe].shape[0]
        #dims[1]=self.offline_movie[nframe].shape[1]
        self.dims=np.array( self.offline_movie[nframe].shape, dtype='uint16')  # TODO: np.array( [[shape]], dtype='uint16') seems better
        bytez=self.offline_movie[nframe]
        self.im = bytez
        self.parent.comm.write_image(self.dims,bytez)
        self.signals.frame_loaded.emit(bytez)

    def update_frame_classes(self):
        """ Find the flash frames and the much darker frames of the loaded movie (FLASH_RATIO_THRESHOLD and
            DARK_RATIO_THRESHOLD, FLASH_PEAK_RATIO_THRESHOLD in defaults). Their results are saved as NaN; flashes also get FLAGS 1. """
        flash_ratio = float(getattr(defaults, 'FLASH_RATIO_THRESHOLD', 0) or 0)
        dark_ratio = float(getattr(defaults, 'DARK_RATIO_THRESHOLD', 0) or 0)
        peak_ratio = float(getattr(defaults, 'FLASH_PEAK_RATIO_THRESHOLD', 0) or 0)
        self.flash_frames, self.dark_frames = classify_frames(self.offline_movie[:self.max_frame], flash_ratio, dark_ratio, peak_ratio)
        if self.flash_frames:
            print("Flash frames (0-based): %s"%sorted(self.flash_frames), flush=True)
        if self.dark_frames:
            print("Much darker frames (0-based): %s"%sorted(self.dark_frames), flush=True)

    def _file_kind(self, file_info):
        """ What to load, as text that contains '.avi', '.bmp', etc. From the file's extension, so it works whatever
            wildcard the user chose (e.g. "files (*.*)"); if the extension isn't one we know, from the dialog's filter text. """
        ext = os.path.splitext(file_info[0][0])[1].lower()
        return ext if ext in ('.bin', '.png', '.bmp', '.avi') else file_info[1]

    def _frame_count(self, vidin, limit):
        """ Number of frames we'll read from an open video, for the progress bar. 0 if it can't say. """
        try:
            count = len(vidin)
        except Exception:
            return 0
        if limit > 0:
            count = min(count, limit + 1) # (The read loop stops after frame index `limit`)
        return count

    def _preview(self, frame):
        """ Show the first frame as soon as it has been read, while the rest of the movie is still loading """
        image = frame.copy()
        image[image >= getattr(defaults, 'SATURATION_MINIMUM', 256)] = 0 # As it will be in the loaded movie
        self.signals.first_frame.emit(image)

    def _occupancy_params(self):
        """ The occupancy matching's settings: the defaults in occupancy.py, unless nextwave_defaults.py sets them """
        params = dict(occupancy.PARAMS)
        for key, name in (('max_shift', 'OCCUPANCY_MAX_SHIFT'), ('missing_penalty', 'OCCUPANCY_MISSING_PENALTY'), ('edge_reward', 'OCCUPANCY_EDGE_REWARD'),
                          ('outside_penalty', 'OCCUPANCY_OUTSIDE_PENALTY'), ('min_sites', 'OCCUPANCY_MIN_SITES')):
            if hasattr(defaults, name):
                params[key] = getattr(defaults, name)
        return params

    def prepare_occupancy_template(self):
        """ For centering_method 'occupancy_match': make the reference shape from the loaded movie (see occupancy.py). The center
            is the middle of that shape (see occupancy.py), not the crosshair. Does nothing for the other methods. Needs the whole
            movie, so the main process does it (before a run), and the workers are given the result. """
        if getattr(defaults, 'centering_method', '') != 'occupancy_match' or not hasattr(self.offline_movie, 'shape'):
            return
        params = self._occupancy_params()
        skip = self.flash_frames | self.dark_frames
        n = self.max_frame
        sample = sorted({int(i) for i in np.linspace(0, n - 1, min(n, int(params['sample_frames'])))} - skip)
        self.occupancy_template = occupancy.build_template([(i, self.offline_movie[i]) for i in sample], self.parent.lenslet_size_pixel, params)
        t = self.occupancy_template
        if t is None:
            print("Occupancy template: no frame had enough spots", flush=True)
        else:
            print("Occupancy template: %d sites from %d frames; the center is site %s" % (len(t['sites']), t['n_frames'], t['center_site']), flush=True)

    def occupancy_autocenter(self):
        """ Find the pupil center of this frame by matching its box occupancy to the template """
        params = self._occupancy_params()
        pitch = self.parent.lenslet_size_pixel
        if self.occupancy_template is None: # (Normally made before the run, in the main process.)
            self.prepare_occupancy_template()
        if self.occupancy_template is None: # Nothing to go on but this frame: its own shape
            self.occupancy_template = occupancy.build_template([(self.offline_curr, self.im)], pitch, params)
        template = self.occupancy_template
        if template is not None:
            if self.offline_curr in self.flash_frames or self.offline_curr in self.dark_frames: # Nothing to judge from a flash or a dark frame
                cx, cy = occupancy.template_center(template)
                info = 'flash or dark frame: center unmoved'
            else:
                cx, cy, info = occupancy.find_center(self.im, template, params)
                info = "shift %s, %d spots on sites, %d outside the template" % (info['shift'], info['matched'], info['outside'])
            self.parent.cx, self.parent.cy = cx, cy
            print("Occupancy center: (%.1f, %.1f); %s" % (cx, cy, info), flush=True)
        self.cx_best, self.cy_best = self.parent.cx, self.parent.cy
        # The max pupil isn't estimated by this method: it's the set one (as it is when the center is set by the user)
        self.iterative_max = self.it_stop * self.parent.pupil_mag
        self.iterative_max_pixels = self.iterative_max/2.0 * 1000 / self.parent.ccd_pixel

    def _add_to_ui(self, movie):
        """ The loaders say which movie the frame list should show; finish_load (on the UI thread) does it """
        self._movie_for_ui = movie

    def finish_load(self, restore=True):
        """ The quick part of loading a movie, which makes widgets and so has to run on the UI thread: fill in the frame list, and
            (restore) show the saved results of the first frame. Call it after load_offline_data() or
            load_offline_background_data(), which don't touch the UI, so they can run in a worker thread. """
        if self._movie_for_ui is not None:
            movie, self._movie_for_ui = self._movie_for_ui, None
            self.parent.ui.add_offline(movie) # (Reports its own progress)
        if restore:
            self.saver.load1(0) # Restore if possible

    def load_offline_background(self,file_info):
        """ Load a background, all in this thread. (The UI does load_offline_background_data in a worker, then finish_load.) """
        self.load_offline_background_data(file_info)
        self.finish_load(restore=False)

    def load_offline_background_data(self,file_info):
        # file_info: from dialog. Tuple: (list of files, file types)
        kind = self._file_kind(file_info)
        if '.bin' in kind:
            pass # TODO
        elif '.avi' in kind:
            fname=file_info[0][0]
            #print("Offline movie: ",fname)

            vidin = ffmpegcv.VideoCapture(fname)
            buf_movie=None

            with vidin:
                total = self._frame_count(vidin, 0)
                for nf,frame in enumerate(vidin):
                    #f1=frame.mean(2)[0:1024,0:1024] # Avg RGB. TODO: crop hard-code
                    f1=frame.mean(2)
                    if buf_movie is None:
                        buf_movie=np.zeros( (50,f1.shape[0],f1.shape[1]), dtype='uint8') # TODO: grow new chunk if necessary
                    buf_movie[nf]=f1
                    print('%04d %03d '%(nf,f1.mean() ),end=' ')
                    self.signals.report("Reading background frames", nf+1, total)

            print("Background: read %d frames of %dx%d"%(nf,f1.shape[0],f1.shape[1]) )
            buf_movie=buf_movie[0:nf,:,:] # Trim to correct
            self.offline_background = buf_movie

            self.signals.report("Subtracting background") # (One long step)
            if self.offline_movie.shape[0] != self.offline_background.shape[0]:
                print("Sub average ")
                # Different number of frames in background and movie. Subtract mean background from each frame
                offline_mean = np.array(self.offline_background.mean(0),dtype='int32') # Mean across frames
                self.offline_movie = self.offline_movie - offline_mean
                self.offline_movie[ self.offline_movie<0] = 0
                self.offline_movie = np.array( self.offline_movie, dtype='uint8')
                self._add_to_ui(self.offline_movie)
            else:
                print("Sub whole movie")
                subbed = np.array(self.offline_movie,dtype='int32') - self.offline_background
                subbed[subbed<0]=0
                subbed=np.array( subbed, dtype='uint8')
                self._add_to_ui( subbed)                
        elif '.bmp' in kind:
            buf_movie=None
            nf=0 # USE nf instead of nf_x to allow skipping (e.g. if directory is in there)
            n_files = sum(".bmp" in frame1 for frame1 in file_info[0])
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
                self.signals.report("Reading background frames", nf, n_files)

            print("Read %d frames of %dx%d"%(nf,f1.shape[0],f1.shape[1]) )
            buf_movie=buf_movie[0:nf,:,:] # Trim to correct
            self.offline_background = buf_movie

            self.signals.report("Subtracting background") # (One long step)
            if self.offline_movie.shape[0] != self.offline_background.shape[0]:
                print("Sub average")
                # Different number of frames in background and movie. Subtract mean background from each frame
                offline_mean = np.array(self.offline_background.mean(0),dtype='int32') # Mean across frames
                self.offline_movie = self.offline_movie - offline_mean
                self.offline_movie[ self.offline_movie<0] = 0
                self.offline_movie = np.array( self.offline_movie, dtype='uint8')
                self._add_to_ui(self.offline_movie)
            else:
                print("Sub each frame from each frame")
                subbed = np.array(self.offline_movie,dtype='int32') - self.offline_background
                subbed[subbed<0]=0
                subbed=np.array( subbed, dtype='uint8')
                self.offline_movie = subbed                
                self._add_to_ui( subbed)                

    def load_offline(self,file_info):
        """ Load a movie, all in this thread. (The UI does load_offline_data in a worker, then finish_load.) """
        self.load_offline_data(file_info)
        self.finish_load(restore=True)

    def load_offline_data(self,file_info):
        self.occupancy_template = None # (Made from the movie, so a new movie needs a new one)
        # file_info: from dialog. Tuple: (list of files, file types)
        fname = file_info[0][0]
        self.offline_fname = fname
        kind = self._file_kind(file_info)

        self.parent.mode_offline=True
        
        self.scan_dir ="X"
        self.condition="NONE"
        self.sub_id="NONAME"
        
        if '.bin' in kind:
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

        elif '.png' in kind:
            buf_movie=None
            pathname = file_info[0][0].upper()

            NO_FILENAME=True
            GY_RESTRUCTURE=True
            if NO_FILENAME:
                self.scan_dir ="X"
                self.condition="NONE"
                self.sub_id="NONAME"
            elif GY_RESTRUCTURE:
                # Find the last 3 (from the right) subdirs
                i0=pathname[:].rfind('/')
                i1=pathname[:i0].find('/')
                i2=pathname[:i1].find('/')
                self.scan_dir = pathname[i0+1:]
                self.condition=pathname[i1+1:i0]
                self.sub_id=pathname[i2+1:i1]
                #print( self.scan_dir, self.condition, self.sub_id )
            else:
                idxSub=pathname.find("SWS") # TODO
                if idxSub==-1:
                    self.sub_id="NONAME"
                else:
                    self.sub_id = pathname[idxSub+4:idxSub+8]
                
                # For Chloe's file layout    
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
            self.fnames = ["" for n in np.arange( len(file_info[0]) )]
            n_files = sum(".png" in frame1 for frame1 in file_info[0])

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
                self.fnames[nf] = frame1
                if nf == 0:
                    self._preview(buf_movie[0])
                nf += 1
                self.signals.report("Reading frames", nf, n_files)

            buf_movie = buf_movie[0:nf]
            self.fnames = self.fnames[0:nf]
            self.rotations = [None]*nf
            
            print(pathname, self.condition, self.scan_dir, self.sub_id)
            print("Read %d frames of %dx%d"%(nf,f1.shape[0],f1.shape[1]) )
            buf_movie=buf_movie[0:nf,:,:] # Trim to correct
            self.offline_movie = buf_movie
            self._add_to_ui(buf_movie)
            self.dims=np.array([buf_movie.shape[1],buf_movie.shape[2]])

        elif '.bmp' in kind:
            buf_movie=None
            pathname = file_info[0][0].upper()

            GY_RESTRUCTURE=True
            if GY_RESTRUCTURE:
                # Find the last 3 (from the right) subdirs
                i0=pathname[:].rfind('/')
                i1=pathname[:i0].rfind('/')
                i2=pathname[:i1].rfind('/')
                i3=pathname[:i2].rfind('/')
                self.scan_dir = pathname[i1+1:i0]
                self.condition=pathname[i2+1:i1]
                self.sub_id=pathname[i3+1:i2]
                #print( i0, i1, i2, i3, self.scan_dir, self.condition, self.sub_id )
            else:
                idxSub=pathname.find("SWS") # TODO
                if idxSub==-1:
                    self.sub_id="NONAME"
                else:
                    self.sub_id = pathname[idxSub+4:idxSub+8]
                
                # For Chloe's file layout    
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
            self.fnames = ["" for n in np.arange( len(file_info[0]) )]
            n_files = sum(".bmp" in frame1 for frame1 in file_info[0])
            for nf_x,frame1 in enumerate(file_info[0]):
                if not (".bmp" in frame1):
                    continue
                #print("Offline: ",nf,frame1)
                im = Image.open(frame1)
                f1 = np.array(im) # TODO: assumes Im is already 8bit monochrome
                if buf_movie is None:
                        buf_movie=np.zeros( (50,f1.shape[0],f1.shape[1]), dtype='uint8') # TODO: grow new chunk if necessary
                buf_movie[nf]=f1
                if nf == 0:
                    self._preview(buf_movie[0])
                
                # Assume fname is xxxx_nnn.bmp : extract nnn
                idx_number=frame1.rfind('_')+1
                idx_number_after=frame1[idx_number:].find('.')+idx_number
                #print( frame1, idx_number, frame1[idx_number:idx_number_after] )
                self.fnames[nf] = int(frame1[idx_number:idx_number_after])

                nf += 1
                self.signals.report("Reading frames", nf, n_files)

            print(pathname, self.condition, self.scan_dir, self.sub_id, self.fnames)
            print("Read %d frames of %dx%d"%(nf,f1.shape[0],f1.shape[1]) )

        elif '.avi' in kind:
            fname=file_info[0][0]
            print("Offline movie: ",fname)
            vidin = ffmpegcv.VideoCapture(fname)
            buf_movie=None

            debug_nframes = defaults.AVI_DEBUG_FRAMES

            with vidin:
                total = self._frame_count(vidin, debug_nframes)
                for nf,frame in enumerate(vidin):
                    f1=frame.mean(2)
                    if buf_movie is None:
                        buf_movie=np.zeros( (defaults.MOVIE_MAX_FRAMES,f1.shape[0],f1.shape[1]), dtype='uint8') # TODO: grow new chunk if necessary
                    buf_movie[nf]=f1
                    print('%04d %03d\n'%(nf,f1.mean() ),end=' ', flush=True)
                    if nf == 0:
                        self._preview(buf_movie[0])
                    self.signals.report("Reading frames", nf+1, total)

                    #if nf<100: # For e.g. debugging
                    #    np.save("img_%02d.npy"%nf,f1)
                        
                    if debug_nframes>0 and nf>=debug_nframes:
                        break

            print("Read %d frames of %dx%d"%(nf,f1.shape[0],f1.shape[1]) )
            self.fnames = ["%03d" for n in np.arange(nf)]

        buf_movie=buf_movie[0:nf,:,:] # Trim to correct
        self.fnames = self.fnames[0:nf]
        self.rotations = [None]*nf

        # Threshold anything too bright
        self.signals.report("Removing saturated pixels") # (One long step)
        buf_movie[buf_movie >= getattr(defaults, 'SATURATION_MINIMUM', 256)] = 0 # (256 or more: nothing, for 8 bit)

        self.offline_movie = buf_movie
        self._add_to_ui(buf_movie) # (Shown by finish_load)
        self.dims=np.array([buf_movie.shape[1],buf_movie.shape[2]])

        self.max_frame = buf_movie.shape[0]

        self.signals.report("Looking for flash and dark frames") # (One long step)
        self.update_frame_classes()

        self.signals.report("Loading saved results") # (One long step)
        self.saver.unserialize() # Load previous if they exist
        # (finish_load fills in the frame list and shows the first frame's saved results)

    def export_all_zernikes(self,dir1="."):
        idx=0
        #out_fname = self.offline_fname + "_zern_%02d.csv"%idx        
        stem = None
        if self.offline_fname.lower().endswith('.avi'): # An AVI's CSV is named for it (only an AVI: the others are many files, or not named for the movie)
            stem = "%s/%s"%(dir1,Path(self.offline_fname).stem)
            out_fname = stem + ".csv"
        else:
            out_fname = "%s/zc_%s_%s_%s.csv"%(dir1,self.sub_id,self.condition,self.scan_dir)

        while Path(out_fname).exists():
            idx += 1
            #out_fname = self.offline_fname + "_zern_%02d.csv"%idx
            if stem:
                out_fname = "%s_%02d.csv"%(stem,idx)
            else:
                out_fname = "%s/zc_%s_%s_%s_%02d.csv"%(dir1,self.sub_id,self.condition,self.scan_dir,idx)

        first_frame = 0 # Start this many frames before the first flash, if there is one
        trim = int(getattr(defaults,'TRIM_TO_FIRST_FLASH',0) or 0)
        if trim > 0 and len(self.flash_frames) > 0:
            first_frame = min(self.flash_frames) - trim # (Negative if the flash is sooner than that into the movie: those rows are empty, so time 0 is always trim frames before the flash)
        print("Exporting %s: frames %d-%d%s"%(out_fname, max(0,first_frame)+1, self.max_frame, " (%d empty rows first)"%-first_frame if first_frame < 0 else ""), flush=True)

        self.f_out = open(out_fname,'w')
        s="subject_id,scan_dir,frame_num,time,ecc,pupil_diam_mm,cx,cy,FLAGS,"
        for nz in np.arange(MAX_EXPORT_ZERNIKES):
            s += "Z%d,"%(nz+1)
        s += "\n"
        self.f_out.write(s)

        for nframe in np.arange(first_frame, self.max_frame):
            s=self.saver.printable1(int(nframe), first_frame)
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
            if np.max(box_pix) - np.mean(box_pix) < defaults.threshold_max_minus_mean:
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
            if ((centroids[2] < defaults.BOX_THRESH) and (dark_as_nan)) or (self.parent.omits[nbox]):
                cenx[nbox] = np.nan
                ceny[nbox] = np.nan

        self.cenx = cenx
        self.ceny = ceny

        if do_apply:
            self.parent.centroids_x=self.cenx
            self.parent.centroids_y=self.ceny

        self.parent.compute_zernikes()
        
        if defaults.DO_CORRECT_LCA:
            wfs_wavelength = self.ui.get_param_xml("OPTICS_LaserWavelength")
            ref_wavelength = defaults.LCA_REFERENCE_WAVELENGTH
            pupil_rad = self.engine.pupil_diam / self.engine.pupil_mag / 2.0
            correction = zernike_functions.LCA_z4_correction( wfs_wavelength, ref_wavelength, pupil_rad )
            print( correction, pupil_rad, wfs_wavelength, ref_wavelength)
            self.parent.zernikes[3] += correction
        self.zernikes = self.parent.zernikes 

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
        step_size = self.it_step
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

            # Add tip/tilt to the centers (unless the center is fixed. The boxes are still shifted by the tilt, below.)
            if not self.fixed_center:
                self.parent.cx -= int( z_new[1] / focal * ccd_pixel )
                self.parent.cy += int( z_new[0] / focal * ccd_pixel )

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
        self.saver.save1(self.offline_curr)
        self.saver.serialize()

    def offline_manual1(self):
        self.parent.offline_frame(self.offline_curr)
        self.iterative_run_good()
        #self.saver.save1(nframe)        

    def offline_auto1(self,nframe):
        # Load
        self.offline_curr=nframe
        self.parent.offline_frame(self.offline_curr)
        #Process
        self.iterative_run_good()
        #self.offline_centroids(conservative_threshold=True) # Now redo, with less conservative
        self.offline_auto_shrink()
        #Save
        self.saver.save1(nframe)
            
    def offline_autoall(self):
        for nframe in np.arange(self.max_frame):
            self.offline_auto1(nframe)
        self.saver.serialize()
    
    def offline_auto_dumb(self):
        self.parent.ui.mode_init()
        for nframe in np.arange(self.max_frame):
            self.offline_curr=nframe
            self.parent.offline_frame(self.offline_curr)
            self.offline_centroids()
            self.saver.save1(nframe)         
        self.saver.serialize()

# iterative_size is size on sensor
    def offline_reset(self, pupil_diam=None):
        """ Fresh search boxes at the starting pupil size, or at pupil_diam (size on the sensor) if given """
        if pupil_diam is None:
            pupil_diam = self.it_start
            pupil_diam = pupil_diam * self.parent.pupil_mag
        self.iterative_size = pupil_diam
        self.iterative_size_pixels = self.iterative_size/2.0 * 1000 / self.parent.ccd_pixel
        #self.parent.ui.line_pupil_diam.setText('%2.2f'%(self.iterative_size ) )
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
        #self.parent.box_size_pixel = size_before
        
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
        self.signals.box_size_changed.emit( float(self.parent.box_size_pixel) )

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
                #print (num_components,area)
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
        if self.debug_dumps:
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

    def autocenter(self):
        if defaults.centering_method=='occupancy_match':
            self.occupancy_autocenter()
            return
        if defaults.centering_method=='estimate_boxes':
            # First start small
            pupil_radius_small = self.it_start * self.parent.pupil_mag / 2.0
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
                np.savez("desired_%d"%self.offline_curr, desired,
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
            if self.debug_dumps:
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

        self.parent.cx = self.parent.box_x[box_min]
        self.parent.cy = self.parent.box_y[box_min]
        self.cx_best = self.parent.box_x[box_min]
        self.cy_best = self.parent.box_y[box_min]

        # TODO: Figure out more correct diameter using max outermost box corner
        p_diam = r_pix*2.0/1000.0*self.parent.ccd_pixel

        if self.it_stop_dirty: # If edited in the UI, override.
            p_diam =  self.it_stop * self.parent.pupil_mag
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
            if self.debug_dumps:
                np.save('im_crop',self.im_smooth_cropped) # DBG

    def offline_startbox(self):
       # try:
       #     self.parent.box_x 
       # except:
       #     self.parent.ui.mode_init() # Call init if needed

        self.iterative_size = self.it_start * self.parent.pupil_mag
        self.iterative_size_pixels = self.iterative_size/2.0 * 1000 / self.parent.ccd_pixel
        
        
        if self.autocenter_enabled and not (self.center_dirty or self.fixed_center): # (Otherwise the center is what's already set)
            self.autocenter()
            # Is this circular? Where to get box centers from?
            #self.parent.init_params( { 'pupil_diam': pupil_diam / self.parent.pupil_mag } ) # Back to real pupil size
            #self.parent.make_searchboxes() 

        if defaults.do_auto_rotation_fix:
            self.offline_rotation_fix()

        # Remember the computed max (and show it to the user):
        if not self.it_stop_dirty:
            self.it_stop = self.iterative_max / self.parent.pupil_mag
            self.signals.pupil_stop_changed.emit( self.it_stop )
        self.parent.mode_offline=True
        self.offline_reset()
        
    def iterative_offline(self):
        pass

    def iterative_step_good(self):
        self.iterative_size_pixels = self.iterative_size/2.0 * 1000 / self.parent.ccd_pixel
        self.offline_stepbox()
        self.signals.pupil_diam_changed.emit( self.iterative_size / self.parent.pupil_mag ) #+step) )
        
    def iterative_run_good(self):
        # Size on the sensor, max pixel radius in the image. (it_stop is the pupil size, so scale by magnification.
        # This is the max if autocenter is skipped (user set the center); otherwise autocenter replaces it.)
        self.iterative_max = self.it_stop * self.parent.pupil_mag
        self.iterative_max_pixels = self.iterative_max/2.0 * 1000 / self.parent.ccd_pixel

        self.offline_startbox()

        if self.skip_enlarge:
            # No growing in steps: straight to the max pupil. (The box shrinking is a separate step, after this.)
            self.offline_reset(self.iterative_max)
            self.signals.pupil_diam_changed.emit( self.iterative_size / self.parent.pupil_mag )
            print("Frame %02d/%02d; %04d boxes. Pupil: %02.2f (enlarge steps skipped)"%(self.offline_curr, self.max_frame,
                                                                                        self.parent.num_boxes, self.iterative_size ),flush=True)
            return

        #self.engine.offline.iterative_max_pixels = float(self.it_stop.text())/2.0 * 1000 / self.engine.ccd_pixel
        while self.iterative_size_pixels < self.iterative_max_pixels:
            self.iterative_step_good()

            s="Frame %02d/%02d; %04d boxes. %04d zern terms. Pupil: %02.2f/%02.2f"%(self.offline_curr, self.max_frame,
                                                                                    self.parent.num_boxes, self.parent.zterms_full.shape[0], self.iterative_size, self.iterative_max )
            print(s,flush=True)
            

    def offline_navigate(self):
        self.saver.load1(self.offline_curr)

    def offline_goodbox(self,nframe):
        nbox=self.parent.ui.box_info

        GOOD_THRESH=0.25 # TODO
        patch=self.parent.ui.box_pix
        self.good_template=self.metric_patch(self.parent.ui.box_pix)
        #self.good_idx=np.where( self.good_template>GOOD_THRESH)[0][0]
        self.good_idx=int(len(self.good_template)*0.6) # TODO
        #print("Goodbox", nbox,nframe,self.good_idx, patch.shape, self.box_size_pixel)

    def offline_rotation_fix(self):
        if self.rotations[self.offline_curr] is None:
            angle,img_rotated=detect_rotation(self.im, angls=defaults.rotation_fix_angles,
                ratio_threshold=defaults.rotation_fix_min_peak_ratio)
            self.rotations[self.offline_curr] = angle
            if not (angle==0):
                self.offline_movie[self.offline_curr] = img_rotated # Overwrite
                self.offline_frame( ) # Updates some necessary local variables
            
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
    
        keys = sorted( self.saver.data.keys() )

        diams=np.array([self.saver.data[key1]['pupil_diam'] for key1 in keys ])
        zerns=np.array([self.saver.data[key1]['zernikes'][0:5] for key1 in keys ])

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

        xvalues = np.array(keys) + 1 # To one-based
        self.parent.ui.offline_dialog.sc.axes.plot( xvalues, J45, 'x-', label='J45')
        self.parent.ui.offline_dialog.sc.axes.plot( xvalues, J180, 's-', label='J180')
        self.parent.ui.offline_dialog.sc.axes.plot( xvalues, sphere, 'o-', label='Sphere')
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

            if self.parent.mode_offline:
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

            frame_name = self.offline_fname + "_%02d.png"%self.offline_curr
            self.parent.ui.update_ui()
            self.parent.ui.image.save(frame_name)

            s="%d,%d,%f,%d,%d,"%(self.offline_curr,self.parent.num_boxes,self.iterative_size,self.parent.cx,self.parent.cy)
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
