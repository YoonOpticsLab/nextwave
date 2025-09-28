import numpy as np

# Rotation fix:
import cv2 as cv
from numpy import fft
from scipy.signal import find_peaks

def detect_rotation(img, angls, ratio_threshold):
    powers=np.zeros( (2,len(angls)))

    height,width=img.shape
    # Find the largest term at 0 degrees to use as freq of interest
    M = cv.getRotationMatrix2D((width/2,height/2),0,1)
    res = cv.warpAffine(img,M, (width,height) )
    sig=np.sum(res,0)
    ft=fft.fft(sig)
    smallest_candidate=10 # Can't be lower than this (large low Freq terms)
    idx_good = np.argmax(ft[smallest_candidate:len(ft)//2])+smallest_candidate

    # Get the right freq from dimensional component 
    for nangle,angl1 in enumerate(angls):
        M = cv.getRotationMatrix2D((width/2,height/2),angl1,1)
        res = cv.warpAffine(img,M, (width,height) )
        for dim in [0]:
            margin_sum=np.sum(res,dim)
            ft=fft.fft(margin_sum);
            powers[dim,nangle] = np.abs(ft[idx_good])

    # For now check horizontal only
    for dim in [0]:
        power1=powers[dim]
        max_loc = np.argmax(power1)
        max_peak = power1[max_loc]
        angl1=angls[max_loc]
        pks=find_peaks(power1)[0]
        if len(pks)==1: # Only the one max
            minor_peaks = power1.min()
            second_peak = power1.min()
        else:
            minor_peaks = np.mean( power1[pks[pks != max_loc ]])
            second_peak = np.max( power1[pks[pks != max_loc ]] )
        ratio = max_peak / second_peak
        if ratio > ratio_threshold:
            print( "ROTATED ", dim, ratio, angl1 )
            im1 = img
            M = cv.getRotationMatrix2D((width/2,height/2),angl1,1)
            rotated = cv.warpAffine(im1,M, (width,height) )
            
            return angl1,rotated
        else:
            return 0, img



