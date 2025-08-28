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

############################################################
# For offline processing: (originally offline.py)
############################################################
ITERATIVE_PUPIL_START=2.65 # TODO: Maybe should determine based on magnification, etc., pupil for minimum # of boxes
ITERATIVE_PUPIL_STEP_SIZE=0.25
ITERATIVE_PUPIL_STOP=8.0

GAUSS_SD=3
BOX_THRESH=2.0
SUBSET_FIT_SIZE=5  # Size of pixel subset to fit a Gaussian to for centroiding
SHRINK_MIN=20 # DEtermined empirically for Chloe's movies TODO
SHRINK_PIXELS=2 # Num. pixels to shrink search boxes by each iteration

scan_frame_to_ecc={'H': np.linspace(-35,35,37),'V': np.linspace(-20,20,27),'D': np.linspace(-28.28,28.28,27),'D2': np.linspace(-28.28,28.28,27)}
  
SATURATION_MINIMUM=255
  
#Centering method:
# "estimate_boxes" (use default positions and find best circle to optimize box population)
# "convex_hull"    (gaussian, threshold (otsu's method), convex hull, fit circle)
centering_method="convex_hull_robust"
CENTERING_GAUSS_SD=10
NONSAT_MAX_OTSU=100
centering_convex_robust_nboots=100
centering_convex_robust_fraction=3
centering_convex_robust_nagree=5

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

MAX_ZERNIKES=65 # Absolute max for 10th order: np.sum( np.arange(10+1+1))-1 .first of 11th order is np.sum(np.arange(12)) )
MAX_ORDER=10

NUM_ZCS=21 # 
START_ZC=1
