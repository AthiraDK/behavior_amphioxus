import numpy as np
import pandas as pd
from scipy import signal



def get_wv_parameters(fs=30, omg0=6):
    
    # 30 scales linearly spaced, max being sampling frequency  + 1
    f_channels = np.arange(1, fs +1, 1)
    widths = omg0*fs / (2*f_channels*np.pi) 
    
    return omg0, widths
    


def wv_transform(df, widths, omg0):
    
    df_wv = pd.DataFrame()
    df_wv['filename'] = df['filename']
    df_wv['frames'] = df['frame']
    
    df.drop(['filename', 'frame'], axis=1, inplace = True)


    for feat in list(df.columns):


        ts = df[feat].values
        cwtm = signal.cwt(ts, signal.morlet2, widths, w = omg0)

        wv_feat = np.abs(cwtm)
        
        for f in range(len(widths)):
            df_wv[f'{feat}_wv{f}'] = wv_feat[f-1,:]
    
    return df_wv