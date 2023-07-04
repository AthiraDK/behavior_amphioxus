import numpy as np
from scipy.signal import savgol_filter
import cv2

def get_curv_savgol(skel_array, wl=5, order=2):
    """ Function to calculate curvatures of a ciona skeleton 
        The derivates are calcuated using savitzky-golay filters
        Keyword arguments:
        skel_array -- 3D array of shape (N,n,2)
            N ------- Number of frames 
            n ------- Number of skeleton points
        wl -- The length of the filter window (i.e. the number of coefficients). wl must be a positive odd integer.
        order -- The order of the polynomial used to fit the samples. It must be less than wl.       
    """
    # first and second derivatives computed using savgol filters
    dx = savgol_filter(skel_array[:,:,0],window_length=wl,polyorder=order,deriv=1,mode='nearest')
    dy = savgol_filter(skel_array[:,:,1],window_length=wl,polyorder=order,deriv=1,mode='nearest')
    ddx = savgol_filter(skel_array[:,:,0],window_length=wl,polyorder=order,deriv=2,mode='nearest')
    ddy = savgol_filter(skel_array[:,:,1],window_length=wl,polyorder=order,deriv=2,mode='nearest')
    # curvature formula adapted from Tierpsy paper 
    #curvatures = np.abs(dx*ddy - dy*ddx)/(dx**2+dy**2)**1.5
    curvatures = (dx*ddy - dy*ddx)/(dx**2+dy**2)**1.5
    return curvatures


def get_quirkiness(skel_array):
    """ Function to eccentricity of a curve 
        Keyword arguments:
        skel_array -- 3D array of shape (N,n,2)      
    """
    dd = [cv2.minAreaRect(x) for x in skel_array.astype(np.float32)]
    dd = [(L,W) if L >W else (W,L) for _,(L,W),_ in dd]
    L, W = list(map(np.array, zip(*dd)))
    quirkiness = np.sqrt(1 - W**2 / L**2)
    return quirkiness

def get_length(skel_array, len_type='nose_to_tail'):
    if len_type == 'nose_to_tail':
        # assume the dorsal skeleton is passed and that tail_tip is the last column
        length = np.linalg.norm(skel_array[:, 0, :] - skel_array[:, -1, :], axis=1)
    else:
        xy_vals = skel_array
        xy_diff = np.diff(xy_vals, prepend=xy_vals[:,-1,:].reshape(-1,1,2), axis=1)
        if len_type == 'mean_point_to_point':
            length = np.nanmean(np.linalg.norm(xy_diff, axis=2), axis=1)
        elif len_type == 'sum_point_to_point':
            length = np.nansum(np.linalg.norm(xy_diff, axis=2), axis=1)
    return length

