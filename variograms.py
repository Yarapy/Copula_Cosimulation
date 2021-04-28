#This library was slightly modified
__author__    = "Connor Johnson"
__url__    = "https://github.com/cjohnson318/geostatsmodels"
__date__      = "2019"
__copyright__ = "Copyright (C) 2019 Connor Johnson"
__license__   = "GNU GPL Version 3.0"

import numpy as np
from scipy.spatial.distance import pdist, squareform

def pairwise( data ):
    '''
    Input:  (data) NumPy array where the first two columns
                   are the spatial coordinates, x and y
    '''
    # determine the size of the data
    npoints, cols = data.shape
    # give a warning for large data sets
    if npoints > 10000:
        print("You have more than 10,000 data points, this might take a minute.")
    # return the square distance matrix
    return squareform( pdist( data[:,:2] ) )

def lagindices(pwdist, lag, tol):
    '''
    Input:  (pwdist) square NumPy array of pairwise distances
            (lag)    the distance, h, between points
            (tol)    the tolerance we are comfortable with around (lag)
    Output: (ind)    list of tuples; the first element is the row of
                     (data) for one point, the second element is the row
                     of a point (lag)+/-(tol) away from the first point,
                     e.g., (3,5) corresponds fo data[3,:], and data[5,:]
    '''
    # grab the coordinates in a given range: lag +/- tolerance
    i, j = np.where((pwdist >= lag - tol) & (pwdist < lag + tol))
    # take out the repeated elements,
    # since p is a *symmetric* distance matrix
    indices=np.c_[i, j][np.where(j > i)]
    return indices

def semivariance(data, indices):
    '''
    Input:  (data)    NumPy array where the first two columns
                      are the spatial coordinates, x and y, and
                      the third column is the variable of interest
            (indices) indices of paired data points in (data)
    Output:  (z)      semivariance value at lag (h) +/- (tol)
    '''
    # take the squared difference between
    # the values of the variable of interest
    # the semivariance is half the mean squared difference
    i=indices[:, 0]
    j=indices[:, 1]
    z=(data[i, 2] - data[j, 2])**2.0
    return np.mean(z) / 2.0


def semivariogram(data, lags, tol):
    '''
    Input:  (data) NumPy array where the first two columns
                   are the spatial coordinates, x and y
            (lag)  the distance, h, between points
            (tol)  the tolerance we are comfortable with around (lag)
    Output: (sv)   <2xN> NumPy array of lags and semivariogram values
    '''
    return variogram(data, lags, tol, 'semivariogram')


def covariance(data, indices):
    '''
    Input:  (data) NumPy array where the first two columns
                   are the spatial coordinates, x and y
            (lag)  the distance, h, between points
            (tol)  the tolerance we are comfortable with around (lag)
    Output:  (z)   covariance value at lag (h) +/- (tol)
    '''
    # grab the indices of the points
    # that are lag +/- tolerance apart
    i=indices[:, 0]
    j=indices[:, 1]
    return np.cov(data[i, 2], data[j, 2])[0][1]


def covariogram(data, lags, tol):
    '''
    Input:  (data) NumPy array where the first two columns
                   are the spatial coordinates, x and y
            (lag)  the distance, h, between points
            (tol)  the tolerance we are comfortable with around (lag)
    Output: (cv)   <2xN> NumPy array of lags and covariogram values
    '''
    return variogram(data, lags, tol, 'covariogram')


def variogram(data, lags, tol, method):
    '''
    Input:  (data) NumPy array where the first two columns
                   are the spatial coordinates, x and y
            (lag)  the distance, h, between points
            (tol)  the tolerance we are comfortable with around (lag)
            (method) either 'semivariogram', or 'covariogram'
    Output: (cv)   <2xN> NumPy array of lags and variogram values
    '''
    # calculate the pairwise distances
    pwdist = pairwise(data)
    # create a list of lists of indices of points having the ~same lag
    index = [lagindices(pwdist, lag, tol) for lag in lags]
    # remove empty "lag" sets, prevents zero division error in [co|semi]variance()
    index = list(filter(lambda x: len(x) > 0, index))
    # calculate the variogram at different lags given some tolerance
    if method in ['semivariogram', 'semi', 'sv', 's']:
        v = [semivariance(data, indices) for indices in index]
    elif method in ['covariogram', 'cov', 'co', 'cv', 'c']:
        v = [covariance(data, indices) for indices in index]
    # bundle the semivariogram values with their lags
    return np.c_[lags, v].T

def detrend(data,amplitude=1,dgree=1):
    '''
    Calculates a linear regression and eliminates the trend
    Parameters
    ----------
    data : NumPy array with coordinates and values
        DESCRIPTION.

    Returns
    -------
    res: the residual of var-trend
    m1,m2,n: the coefficient of the trend line such as
    trend(x,y)=m1*x+m2*y+n
    '''
    covz=np.cov(data[:,[1,2]].T)
    varz=np.var(data[:,1])
    if varz!=0:
        m=covz[0][1]/varz
    else:
         m=0
    n=np.mean(data[:,2])-m*np.mean(data[:,1])
    m=amplitude*m
    res=data[:,2]-(n+m*data[:,1])
    
    return res,m,n
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
