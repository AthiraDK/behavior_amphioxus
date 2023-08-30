import numpy as np
from sklearn.preprocessing import scale, StandardScaler

def obtain_M(X, Y, window):
    """returns normalized embedding matrix M for columns X and Y with the specified window size.
    This matrix can be passed to the get_H function to compute the complexity value"""
    Mx = np.array(X[:window]) #initialize first row of Mx
    My = np.array(Y[:window]) #initialize first row of My
    for ii in range(1, len(X)-window): #skip first entry since we already have that in M
        Mx = np.vstack([Mx, X[ii:ii+window]]) #add new vector to Mx
        My = np.vstack([My, Y[ii:ii+window]]) #add new vector to My
    
    
    Mx = StandardScaler().fit_transform(Mx)
    My = StandardScaler().fit_transform(My)
    
#     cols = Mx.shape[1] #get number of columns from array object
#     for ii in range(cols): #normalize per column:
#         Mx[:,ii] = Mx[:,ii] - np.nanmean(Mx[:,ii])
#         My[:,ii] = My[:,ii] - np.nanmean(My[:,ii])

    
    M = np.dstack([Mx,My]) #stack the arrays Mx and My   
    return M #return M


def get_H(M):
    """Performs singular value decomposition on M, and uses the diagonal matrix S
    to calculate complexity value H as the entropy in the distribution of components of S
    I advise you to read Herbert-Read (2017) on escape path complexity"""
    U,S,V = np.linalg.svd(M) # do singular value decomposition
    hats_array = [s/np.sum(s) for s in S] #make hats array
    local_H = [-np.sum(s*np.log2(s)) for s in hats_array]
#     H = -np.sum([s*np.log2(s) for s in hats_array]) #calculate H
    H = -np.sum(hats_array * np.log2(hats_array))
    return local_H,H
