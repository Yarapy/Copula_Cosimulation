__author__    = "Yarilis Gómez Martínez (yarilisgm@gmail.com)"
__date__      = "2021"
__copyright__ = "Copyright (C) 2021 Yarilis Gómez Martínez"
__license__   = "GNU GPL Version 3.0"

##Modules##
import numpy as np
import variograms
import model
import ot_copula_conditional_YE as cond
import time

def variogram_parameter(Data_Grid,sill,nugget_var,a_range, lag_number, var_model='spherical',lag_size=0):
    #Variogram Data#
    if lag_size==0:
        X_coord=Data_Grid[:,0]
        Z_coord=Data_Grid[:,1]
        dist_max=np.sqrt((max(X_coord)- min(X_coord))**2 + (max(Z_coord)- min(Z_coord))**2)
        lag_size=0.5*dist_max/lag_number
    else:
        dist_max=lag_size*lag_number*2
    lag_tolerance=lag_size/2
    lags = np.linspace(lag_size*0.75, lag_size*lag_number-lag_size*0.25, lag_number)        
    #Teorical semivariogram#
    if var_model=='Spherical' or var_model=='Esférico':
        M=model.spherical
    elif var_model=='Gaussian' or var_model=='Gaussiano':
        M=model.gaussian
    elif var_model=='Exponential' or var_model=='Exponencial':
        M=model.exponential 
    elif var_model=='Power' or var_model=='Potencia':
        M=model.power
    elif var_model=='Linear' or var_model=='Lineal':
        M=model.linear
    selected_model=model.variogram_combination( model.nugget, M, (0,nugget_var), (a_range,sill-nugget_var), lags)
    svt=selected_model(lags)
    return svt,dist_max, lag_size,lag_tolerance,lags



class funcobj(object):
##Define the objective function##
#Objective function (optimize an input map) where val = variable.
    def __init__(self,w, Data_Grid,svt,lag_tolerance,lags, bivariate_distribution_data, trend_coef=0):       
        self.wO2=w[0]
        self.wO5=w[1]
        self.Data_Grid=Data_Grid
        self.trend=trend_coef
        self.svt=svt
        self.lag_tolerance=lag_tolerance
        self.lags=lags
        self.bivariate_distribution_data=bivariate_distribution_data
      
    #Mean squared error of the variogram
    def funcO2(self,val):
        Ps = np.hstack((self.Data_Grid[:,:2],val.reshape(-1, 1)))
        vdatas = variograms.semivariogram(Ps,self.lags,self.lag_tolerance)
        svs = vdatas[1]-0.5*(self.lags*self.trend)**2
        O2=np.sum(((svs-self.svt)/self.svt)**2)
        return O2
               
    #Mean squared error of the conditional distribution
    def funcO5(self,val):
        Pared_val=np.hstack((self.Data_Grid[:,2].reshape(-1, 1),val.reshape(-1, 1)))
        bivariate_distribution_s=cond.ot_copula_fit(Pared_val) 
        conditioned_dist_t=cond.ot_compute_conditional(val,self.bivariate_distribution_data,self.Data_Grid) 
        conditioned_dist_s=cond.ot_compute_conditional(val,bivariate_distribution_s,self.Data_Grid) 
        O5=np.sum((conditioned_dist_s- conditioned_dist_t)**2)
        return O5
    
    def funcO(self,val):
        if self.wO5==0:        
            r=self.wO2*self.funcO2(val)
        elif self.wO2==0:
            r=self.wO5*self.funcO5(val)
        else:
            r=self.wO2*self.funcO2(val)+self.wO5*self.funcO5(val)
        return r
    

class funcobj_cond(funcobj):
##Define the objective function conditional##
    def __init__(self,w, Data_Grid,Data_Conditioning,svt,lag_tolerance,lags, bivariate_distribution_data, trend_coef=0):       
        funcobj.__init__(self,w, Data_Grid,svt,lag_tolerance,lags, bivariate_distribution_data, trend_coef=0)
        self.Data_Conditioning=Data_Conditioning
        self.trend=trend_coef
        
    def funcO2(self,val):
        Ps = np.hstack((self.Data_Grid[:,:2],val.reshape(-1, 1)))
        Ps = np.vstack((Ps,self.Data_Conditioning[:,[0,1,3]]))
        vdatas = variograms.semivariogram(Ps,self.lags,self.lag_tolerance)
        svs = vdatas[1]-0.5*(self.lags*self.trend)**2
        O2=np.sum(((svs-self.svt)/self.svt)**2)        
        return O2
    
    def funcO5(self,val):
        Pared_val=np.hstack((self.Data_Grid[:,2].reshape(-1, 1),val.reshape(-1, 1)))
        Pared_val = np.vstack((Pared_val,self.Data_Conditioning[:,2:]))
        bivariate_distribution_s=cond.ot_copula_fit(Pared_val)
        conditioned_dist_t=cond.ot_compute_conditional(val,self.bivariate_distribution_data,self.Data_Grid) 
        conditioned_dist_s=cond.ot_compute_conditional(val,bivariate_distribution_s,self.Data_Grid) 
        O5=np.sum((conditioned_dist_s- conditioned_dist_t)**2)
        return O5
