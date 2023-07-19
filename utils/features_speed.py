import numpy as np
import pandas as pd
from hampel import hampel

def get_speeds(df, filt = False):
    list_bp_speeds = []
    bodyparts = []
    for i, col in enumerate(df.columns):
        bp = col.split('_')[0]
        if bp not in bodyparts:
            bodyparts.append(bp)
            xy_data = df.filter(items = [f'{bp}_x',f'{bp}_y'])
            xy_coords = xy_data.values
            time_indices = xy_data.index
            speeds = calc_speeds(xy_coords, time_indices)
            speeds.name = f'speed_{bp}'
            if filt:
                filt_speeds = filter_speeds(speeds)
                list_bp_speeds.append(filt_speeds)
            else:
                list_bp_speeds.append(speeds)
        
    df_speeds = pd.concat(list_bp_speeds, axis=1)
    return df_speeds


def calc_speeds(bp_xycoords, time_indices):
    diff_xy = np.diff(bp_xycoords, prepend = bp_xycoords[1,:].reshape(1,2), axis=0)
    sum_xy = diff_xy[:,0]**2 + diff_xy[:,1]**2
    sqrt_xy = np.sqrt(sum_xy)
    time_int = np.diff(time_indices, prepend = time_indices[1], axis=0)
    speed = sqrt_xy / time_int
    bp_speed = pd.Series(speed, index=time_indices)
    return bp_speed

def filter_speeds(bp_speed):
    from hampel import hampel
    bp_speed_filt = hampel(bp_speed, window_size=15, n=3, imputation=True)
    # bp_speed_filt.name = f'speed_{bp}'
    return bp_speed_filt