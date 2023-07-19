import pandas as pd

def interpol_spatial(df, method='pchip', limit=2):
    col_dict = {col: i for i, col in enumerate(df.columns)}
    col_dict_rev = {v:k for k,v in col_dict.items()}
    # assuming dorsal or ventral side to be a cubic-spline 
    df_interp = df.rename(columns=col_dict).interpolate(axis=1, method=method, limit=limit,limit_area='inside').rename(columns=col_dict_rev)
    return df_interp

def interpol_temporal(df, limit=5, direction='both'):
    df_interp = df.interpolate(limit=limit, limit_direction=direction)
    return df_interp