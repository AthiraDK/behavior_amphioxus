import os
import cv2
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DLC_tracking:
    def __init__(self, filename, dlc_folder):
        self.filename = filename
        self.filepath = os.path.join(dlc_folder, filename)
        self.df_data = self.read_file()
        self.get_meta_from_dataframe()
        self.df_data = self.df_data.droplevel(level=[0,1], axis=1)
        self.df_data.columns = ['_'.join(col) for col in self.df_data.columns]
#         self.find_outliers()
        self.df_data.reset_index(inplace=True)
        self.df_data.rename(columns={'index':'frame'}, inplace=True)
        self.df_data['n_missing_bodyparts'] = self.df_data.filter(like='_x').isna().sum(axis=1)
        
    def get_meta_from_filename(self, key):
        fname_parts = self.filename.split('_')
        if key == 'video_length':
            return int(fname_parts[3].split('m')[0])
        elif key == 'date_time':
            return '_'.join(fname_parts[0:2])
        elif key == 'dlc_scorer':
            return f"DLC_{'_'.join(fname_parts[-4:-1])}"
        else:
            return None
       
    
    def get_coord_data(self, key='all', likelihood_thresh=0.8):
        if key == 'all':
            return self.df_data.filter(regex="(_x|_y)$").values.reshape(-1,29,2)
        elif key == 'x':
            return self.df_data.filter(like='_x').values
        elif key == 'y':
            return self.df_data.filter(like='_y').values
        elif key in self.list_bodyparts:
            return self.df_data[self.df_data[f"{key}_likelihood"]==1].filter(regex=f"({key}_x|{key}_y)$")            
        else:
            return self.df_data
        
        
    def get_length(self, len_type='nose_to_tail'):
        if len_type == 'nose_to_tail':
            length = np.linalg.norm(self.df_data.loc[:, ['NT_x', 'NT_y']].values - self.df_data.loc[:, ['TT_x', 'TT_y']], axis=1)
        elif len_type == 'point_to_point':
            xy_vals = self.df_data.filter(regex="(_x|_y)$").values.reshape(-1,29,2)
            xy_diff = np.diff(xy_vals, prepend=xy_vals[:,-1,:].reshape(-1,1,2), axis=1)
            length = np.nanmean(np.linalg.norm(xy_diff, axis=2), axis=1)
        return length
    
    def get_speed_avg(self):
        pass 
    
    def find_outliers(self, remove=False):
        self.df_data['avg_node_dist'] = self.get_length(len_type='point_to_point')
        self.df_data['length'] = self.get_length()
#         self.df_data['is_outlier'] = self.df_data['length'].apply(lambda x: (x>20))
        if remove:
            self.df_data = self.df_data[self.df_data['length']<150]
            self.df_data = self.df_data[self.df_data['avg_node_dist']<10]
        return self
    
    def read_file(self):
        df_data = pd.read_hdf(self.filepath)
        return df_data
    
    
    def get_meta_from_dataframe(self):
        self.columns = self.df_data.columns
        self.dlc_scorer = self.columns.levels[0][0]
        self.individual = self.columns.levels[1][0]
        self.list_bodyparts = list(self.columns.levels[2])
        return None
    
    
    def plot_trajectory(self, keys=['NT','TT']):
        fig, axes = plt.subplots(1,len(keys),figsize=(5*len(keys),5))
        for i, k in enumerate(keys):
            ax = axes[i]
            df = self.get_coord_data(key=k)
            coords_xy = df.values
            ax.scatter(coords_xy[:,0], coords_xy[:,1], s=1, c=df.index, cmap='jet')
            ax.set_xlim([0,1500])
            ax.set_ylim([0,1500])
        plt.show()
        return None

    def find_bbox_dlc(self, frame=-1):
        dict_bbox = {}
        
        
        if type(frame) == list:
            list_frames = frame
        if (type(frame) == int) | (type(frame) == float):
            if frame < 0:
                list_frames = list(self.df_data['frame'])
            if frame >= 0:
                list_frames = [frame]
        
        for frame in list_frames:
            
            row = self.df_data[self.df_data['frame'] == frame]
            ind_frame = int(frame)
            x_min = np.nanmin(row.filter(like='_x').values)
            x_max = np.nanmax(row.filter(like='_x').values)
            y_min = np.nanmin(row.filter(like='_y').values)
            y_max = np.nanmax(row.filter(like='_y').values)
            w = x_max - x_min
            h = y_max - y_min
            dict_bbox[ind_frame] = [x_min, y_min, w, h]


        return dict_bbox



