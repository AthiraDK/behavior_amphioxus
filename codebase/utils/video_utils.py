import os
import sys
import cv2
import numpy as np
import pandas as pd
from dlc_helper import DLC_tracking


def get_rois(path_to_video, path_to_dlc_coords, frames, crop=True):
    
    

    dlc_folder, dlc_filename = os.path.split(path_to_dlc_coords)
    dlc_obj = DLC_tracking(dlc_filename, dlc_folder)
    
    dict_bbox = dlc_obj.find_bbox_dlc(frame=frames)
    dict_rois = {}
    
    for frame in frames:
        print(frame)
        # extract the frame
        cap = cv2.VideoCapture(path_to_video)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.set(cv2.CAP_PROP_FRAME_COUNT, frame)
        ret, image = cap.read()
        

        if crop:
            # Find coords of the roi corners
            x, y, w, h = dict_bbox[frame]
            mid_x = x + (w/2)
            mid_y = y + (h/2)
            try:
                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)
                x_new, y_new, w_new, h_new = find_square_bounding(x,y,w,h, height_max = 150, width_max = 150)
                roi_box_padded = image[y_new:y_new+h_new, x_new:x_new+w_new]
                dict_rois[frame] = roi_box_padded

            except ValueError:
                print(f'Value Error encountered for frame {f_ind-1}')
            except Exception as e:
                print(f'Error encountered')
                print(e)
                
        else:
            dict_rois[frame] = image
            
        cap.release()
        
    return dict_rois
    
    
def find_square_bounding(x, y, w, h, height_max = 200, width_max = 200):

    true_height = 1024
    true_width = 1280 

    height_diff = height_max - h
    width_diff = width_max - w

    # ideally, I would like to keep the center of the image the same
    if y - (height_diff // 2) < 0 : # not enough space in the top margin
        top_margin = 0
    else:
        top_margin = y - (height_diff // 2)

    if y + h + (height_diff // 2) >= true_height: # not enough in the botton margin
        bottom_margin = true_height
        top_margin = bottom_margin - height_max
    else:
        bottom_margin = top_margin + h

    if x - (width_diff // 2) < 0 : # not enough space in the left margin
        left_margin = 0
    else:
        left_margin = x - (width_diff // 2)

    if x + w + (width_diff // 2) >= true_width: # not enough in the right margin
        right_margin = true_width
        left_margin = right_margin - width_max
    else:
        right_margin = left_margin + w
        


    y_new = top_margin
    x_new = left_margin
#     print(right_margin,left_margin, w, top_margin,bottom_margin, h)

    return x_new, y_new, width_max, height_max