import numpy as np

############################################################
# From the UI:
############################################################
QIMAGE_HEIGHT=1024
QIMAGE_WIDTH=1024

MAIN_HEIGHT_WIN=1024
MAIN_WIDTH_WIN=1800

# These not used anymore:
SPOTS_HEIGHT_WIN=768
SPOTS_WIDTH_WIN=768

# This is used, but should be based on image, not hard-coded:
SPOTS_WIDTH_WIN_MINIMUM=1024

# TODO
CAM_EXPO_MIN = 32./1000.0 # TODO
CAM_EXPO_MAX = 100000 # TODO
CAM_GAIN_MIN = 0
CAM_GAIN_MAX = 9.83

ui_searchbox_color=(0,128,200)
ui_searchbox_width=1.5

ui_centroid_color=(255,0,255)
ui_centroid_size=3.0

ui_normalize_max = 50

############################################################
# For offline processing: (originally offline.py)
############################################################
ITERATIVE_PUPIL_START=2.65 # TODO: Maybe should determine based on magnification, etc., pupil for minimum # of boxes
ITERATIVE_PUPIL_STEP_SIZE=0.25
ITERATIVE_PUPIL_STOP=8.0
ITERATIVE_AUTOCENTER=1 # 1: find the pupil center in each frame (and snap it to a search box). 0: don't. Use the current center (the UI's, initially cx,cy in config.json). Also then the max pupil isn't estimated: it's ITERATIVE_PUPIL_STOP_FORCE (or ITERATIVE_PUPIL_STOP). Tip/tilt still nudges the center while enlarging: see ITERATIVE_FIXED_CENTER
ITERATIVE_FIXED_CENTER=0 # 1: use the same pupil center for every frame: the UI's current center (initially cx,cy in config.json). Don't find the center in each frame, and don't move it with tip/tilt while enlarging. 0: center each frame
ITERATIVE_SKIP_ENLARGE=1 # 1: skip the "enlarge pupil" steps (growing from ITERATIVE_PUPIL_START by ITERATIVE_PUPIL_STEP_SIZE). Start at the max pupil (see below), then just shrink the search boxes. 0: enlarge in steps
ITERATIVE_PUPIL_STOP_FORCE=4.0 # mm. If >0, always use this as the max pupil (as if typed into the UI's max box) instead of estimating it from each image. 0=estimate

GAUSS_SD=3
BOX_THRESH=2.0
SUBSET_FIT_SIZE=5  # Size of pixel subset to fit a Gaussian to for centroiding
SHRINK_MIN=20 # DEtermined empirically for Chloe's movies TODO
SHRINK_PIXELS=2 # Num. pixels to shrink search boxes by each iteration

threshold_max_minus_mean=-10 # If box max - mean is less than this, make box NaN

scan_frame_to_ecc={'H': np.linspace(-35,35,37),'V': np.linspace(-20,20,27),'D': np.linspace(-28.28,28.28,27),'D2': np.linspace(-28.28,28.28,27)}
  
SATURATION_MINIMUM=0 # When loading a movie, pixels at or above this value are set to 0 (so saturated pixels don't count). 0 = leave them as they are. (Was 255.)
  
#Centering method:
# "estimate_boxes" (use default positions and find best circle to optimize box population)
# "convex_hull"    (gaussian, threshold (otsu's method), convex hull, fit circle)
# "convex_hull_robust"    + outlier detection
# "convex_hull_robust_dynamic" : dynamic threshold based on components(spots), not OTSU
# "occupancy_match" : match which lenslet boxes have a spot to a reference shape made from the movie's frames (see occupancy.py).
#                     Tolerates obstructed parts of the pupil. The center is where the crosshair is (on the frame on screen) when a run
#                     starts. Doesn't estimate the max pupil: it is ITERATIVE_PUPIL_STOP_FORCE (or ITERATIVE_PUPIL_STOP).
#                     Needs ITERATIVE_AUTOCENTER=1. Optional settings: OCCUPANCY_MAX_SHIFT (4), OCCUPANCY_MISSING_PENALTY (0.25),
#                     OCCUPANCY_EDGE_REWARD (0.3), OCCUPANCY_OUTSIDE_PENALTY (1.0), OCCUPANCY_MIN_SITES (12)
centering_method="occupancy_match"
CENTERING_GAUSS_SD=1
NONSAT_MAX_OTSU=100
centering_convex_robust_nboots=100
centering_convex_robust_fraction=3
centering_convex_robust_nagree=5
centering_dynamic_ncomponents=600 # TODO: Better would to base on # of spots/lenslets, known from spacing etc.
centering_dynamic_area=3500 # TODO: Better would be based on image size

# AUTO Rotation detection and correction
do_auto_rotation_fix=False # (The "Rotate this frame" button in the offline panel still works)
rotation_fix_angles=np.linspace(-6,6,100)
rotation_fix_min_peak_ratio=3.0

# For estimate boxes. Max Zernike number to use. 5=all 2rd order. 9=all 3rd order.
ZERNIKES_FOR_INITIAL_CENTERING=9
NUM_ZS_FOR_EXTRAPOLATE=9
NUM_ZS_FOR_SHIFT=9
NUM_ZS_FOR_SHRINK=9

############################################################
# Zernike etc.
############################################################

# Analysis. Minimum # of boxes needed per zernike term
# TODO: Make this a maleable parameter
MIN_BOXES_PER_NZERN=2

ZERNIKE_MAX_ABS=10.0 # um. In saved results, any Zernike coefficient with a magnitude above this is saved as NaN (just that coefficient). 0 = off

FLASH_RATIO_THRESHOLD=4.0 # A frame whose mean brightness is more than this many times the movie's median is a flash (the whole image lights up). Saved with FLAGS 1 (a column of the CSV, and 'flags' in the pickle), and Zernikes/centroids NaN. 0 = off
FLASH_PEAK_RATIO_THRESHOLD=20.0 # A run of flash frames only counts as a flash if its brightest frame is more than this many times the movie's median. Keeps a slow, moderate brightening (a few times the median, for many frames) from being taken for one. Real flashes peak at 60x or more. 0 = any run
DARK_RATIO_THRESHOLD=0.25 # A frame whose mean brightness is less than this fraction of the movie's median is much darker than usual (eye closed, lights off): Zernikes/centroids saved as NaN, but not flagged. 0 = off
DARK_BOX_FRACTION_THRESHOLD=0.5 # If more than this fraction of a frame's boxes found no spot, its Zernikes (and the spot positions estimated from them) are saved as NaN. Not flagged. The measured centroids are kept. 0 = off

FRAME_RATE=100.0 # Frames per second. The exported CSV's "time" column is (exported frame number - 1) / FRAME_RATE
TRIM_TO_FIRST_FLASH=300 # CSV export: start this many frames before the movie's first flash (300 frames = 3 s at 100 FPS), and leave out the frames before that. The first exported frame is frame_num 1, time 0. 0 = export every frame. (A movie with no flash is exported whole.)

MAX_ZERNIKES=65 # Absolute max for 10th order: np.sum( np.arange(10+1+1))-1 .first of 11th order is np.sum(np.arange(12)) )
MAX_ORDER=10

NUM_ZCS=21 # 
START_ZC=1

DO_CORRECT_LCA=0
LCA_REFERENCE_WAVELENGTH=555

#########
## AVIs
#########

MOVIE_MAX_FRAMES=4096 # Maximum loadable
AVI_DEBUG_FRAMES=4096 # Set large (or -1) to load all frames

#########
## Offline processing
#########

OFFLINE_LOAD_FILTER="Movies (*.avi)" # File wildcard selected by default in the "Load Offline Source" dialog. Any Qt filter, e.g. "Movies (*.avi)" or "My movies (native*.avi)"
OFFLINE_BACKGROUND_FILTER="Cam1 Images (sweep_cam1_*.bmp)" # Same, for "Load Offline Background"
OFFLINE_WORKERS=0 # Processes for "Auto process all frames". 0=automatic (cores-1). 1=one frame at a time, in the UI process (old behavior)
