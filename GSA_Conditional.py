###Spatial simulation program using dual annealing_scipy###

##Brief explanation of the method##
"""
scipy.optimize.dual_annealing

"""
__author__    = "Yarilis Gómez Martínez (yarilisgm@gmail.com)"
__date__      = "2020-03-10"
__copyright__ = "Copyright (C) 2020 Yarilis Gómez Martínez"
__license__   = "BSD 3-Clause"

##Modules##
import numpy as np
import time
import dual_annealing as da
import variograms
import LoadSave_data as lsd
import funcionesO as fo
import Grafics as graf
import openturns as ot
import ot_copula_conditional_YE as cond

##Name of the files to save outputs##
#Logger modes: 'w' erase previous file, 'a' appending to the end of the file
output_namefile='Dual_Ann_Lakach_CAP'
log_console = graf.Logger('Results/Log_'+output_namefile+'.log', mode="w") 

##Load data## 
Data_nc=lsd.Load_columns(7,[1,2,3,4,5,6,7],windowname='Select no Conditioning Data File',delimiter=',')
Condition=lsd.Load_columns(7,[1,2,3,4,5,6,7],windowname='Select Conditioning Data File',delimiter=',')
Data=np.vstack((Data_nc,Condition))
Data_Grid_nc=np.hstack((Data_nc[:,0].reshape(-1, 1),Data_nc[:,2].reshape(-1, 1),Data_nc[:,4].reshape(-1, 1))) 
Data_Conditioning=np.hstack((Condition[:,0].reshape(-1, 1),Condition[:,2].reshape(-1, 1),Condition[:,4].reshape(-1, 1),Condition[:,3].reshape(-1, 1))) 
X=Data[:,0]
Z=Data[:,2]
Phit=Data[:,3]
Phit_nc=Data_nc[:,3]
Ip_nc=Data_nc[:,4]
Ip=Data[:,4]
Data_Grid=np.hstack((X.reshape(-1, 1),Z.reshape(-1, 1),Ip.reshape(-1, 1)))

##Variogram##
P=np.hstack((X.reshape(-1, 1),Z.reshape(-1, 1),Phit.reshape(-1, 1)))
#Input parameters#
#BF3
model='gaussian';sill=0.0035;nugget_var=0.002;a_range=55;amplitude=0.8;w0=[1,0.001] 
Nobs=len(Data_Grid_nc)
lag_number= 10
lag_size=0 #If lag_size=0 it is calculated automatically.
Phit_detrend,pend,zero=variograms.detrend(P,amplitude=amplitude)###Trend amplitude=0, Detrend amplitude=1
#Teorical input semivariogram#
svt,dist_max,lag_size,lag_tolerance,lags=fo.variogram_parameter(Data_Grid,sill,nugget_var,a_range,lag_number,var_model=model,lag_size=lag_size)

##Bivariate input distribution##
#BF3
beta1 = 2458.48;gamma1 = 28953.5
marginal1= ot.Gumbel(beta1, gamma1)
beta2 = 0.0489963;gamma2 = 0.156505
marginal2= ot.Gumbel(beta2, gamma2)
theta = -5.21511
copula= ot.FrankCopula(theta)
bivariate_distribution_data = ot.ComposedDistribution([marginal1,marginal2], copula)
marginal_data=[bivariate_distribution_data.getMarginal(i) for i in [0,1]]
copula_data=bivariate_distribution_data.getCopula()

#Weights#
w1=[1,1]
w=[w0[0]*w1[0],w0[1]*w1[1]] 
Phit_min=min(Phit)
Phit_max=max(Phit)

##Objective function##
F=fo.funcobj_cond(w, Data_Grid_nc,Data_Conditioning,svt,lag_tolerance,lags, bivariate_distribution_data, trend_coef=pend)


print("----------------------------------------------------------------------")
print("Input Objective Function:                                             ")
print("----------------------------------------------------------------------")
print("Variogram parameters:                                                 ")
print("----------------------------------------------------------------------")
print("Model=",model)
print("Observation number =", Nobs)
print("Max distance  =", dist_max)
print("Lag number =", lag_number)
print("Lag size =", lag_size)
print("Lag tolerance =", lag_tolerance)
print("Sill =", sill)
print("Nugget efect value =", nugget_var)
print("Range =", a_range)
print("Trend slope =", pend)
print("Trend zero =", zero)
print("Trend amplitude=",amplitude)
print("----------------------------------------------------------------------")
print("Bivariate Distributions from DATA:                                    ")
print("----------------------------------------------------------------------")
print("Marginal for Ip(data_grid) =",marginal_data[0]) 
print("Marginal for Phit(Var) =",marginal_data[1]) 
print("Copula for marginals =",copula_data)
print("----------------------------------------------------------------------")
print("Weights:                                                              ")
print("----------------------------------------------------------------------")
print("Variogram weight =",w[0]) 
print("Distribution weight =",w[1]) 
print("----------------------------------------------------------------------")

"""-----------------------Method-Dual-Ann-----------------------------------"""

##Method input parameters##
bounds=[(Phit_min,Phit_max) for i in range(Nobs)] #Variable search space
maxiter=300 #25 iter para cualquier int_temp evalute 4801,1 iter 23 callbak y 193 evalute 
#args
#local_search_options
initial_temp=16 #Calcular temperatura por Dreo
restart_temp_ratio=2e-5
visit=2.7 #The value range is (0, 3]. The value 1 is classic SA and 2 is fast SA.
accept=-5 #Default value is -5.0 with a range (-1e4, -5].
maxfun=1e7
#seed
no_local_search=True
epsilon=1e-3 #Value from which the optimization is stopped
def callback_epsilon(x, f, context):
    if f<epsilon:
        return True
#Create a random array wich preserve conditional distribution from Pared values
Phit_ini=cond.ot_sample_conditional(bivariate_distribution_data,Data_Grid_nc)
x0=np.array(Phit_ini)

print("----------------------------------------------------------------------")
print("Dual_Anneling parameters:                                             ")
print("----------------------------------------------------------------------")
print("Objective Function: Variogram and Bivariate Distribution Function     ")
print("Bounds for variables =", bounds[0])
#print("args =", args)
print("Maximum number of global search iterations  =", maxiter)
#print("Local search options =", local_search_options)
print("Initial temperature =", initial_temp)
print("Restart temperature ratio=", restart_temp_ratio)
print("Parameter for visiting distribution =", visit)
print("Parameter for acceptance distribution =", accept)
print("Number of objective function calls=", maxfun)
#print("Seed =", seed)
print("No local search =", no_local_search)
print("Minimization halted value =", epsilon)
print("x0 =", x0)
print("----------------------------------------------------------------------")
print("Initial value of objective functions")
print("  fun O1:",F.funcO2(x0)*w[0])
print("  fun O2:",F.funcO5(x0)*w[1])
print("  fun:",F.funcO(x0))
print("----------------------------------------------------------------------")

##Result##
start_time = time.time()
ResultDA = da.dual_annealing(F.funcO,bounds,maxiter=maxiter, initial_temp=initial_temp,restart_temp_ratio=restart_temp_ratio,
                                visit=visit,accept=accept,maxfun=maxfun,no_local_search=no_local_search, callback=callback_epsilon,x0=x0 )
end_time=time.time() - start_time
ResultDAx=np.hstack((ResultDA.x,Condition[:,3]))

print("----------------------------------------------------------------------")
print("Dual_Anneling result:                                                 ")
print("----------------------------------------------------------------------")
print("Objective functions")
print("  fun O1:",F.funcO2(ResultDA.x)*w[0])
print("  fun O2:",F.funcO5(ResultDA.x)*w[1])
print(str(ResultDA).split('\n       x:')[0])
print("Execution time: %s seconds" % end_time)
print("----------------------------------------------------------------------")
print("Optimal solution \n x= \n[",end='')
print(*ResultDAx,sep=', ',end=']\n')
print("----------------------------------------------------------------------")

"""------------------------------------End-Dual-Ann-------------------------"""

###Analysis of the result##
#Load saved data (only if necessary)#
#Ps=lsd.Load_columns(3,[1,2,3],delimiter=' ',windowname='Select Results Data File')

#Variable#
Ps = np.hstack((Data_Grid[:,:2],ResultDAx.reshape(-1, 1)))
Phits=Ps[:,2]
Pared_s=np.hstack((Ip.reshape(-1, 1),Phits.reshape(-1, 1) ))
bivariate_distribution_s=cond.ot_copula_fit(Pared_s)
marginal_s=[bivariate_distribution_s.getMarginal(i) for i in [0,1]]
copula_s=bivariate_distribution_s.getCopula()

#Save variable#
lsd.Save_Data('Results/Result_'+ output_namefile +'.dat',Ps, columns=["X", "Z", "Phits"])

#Descriptive Univariate Statistics#
T_Ip=graf.Stats_Univariate(Ip)
T_Ip.histogram_boxplot(Ip,xlabel='Ip',marginal=marginal_data[0])
T_Phit=graf.Stats_Univariate(Phit)
T_Phit.histogram_boxplot(Phit,xlabel='Phit',marginal=marginal_data[1])
T_Phits=graf.Stats_Univariate(Phits)
T_Phits.histogram_boxplot(Phits, xlabel='Phits',marginal=marginal_s[1])

print("----------------------------------------------------------------------")
print("Descriptive Univariate Statistics:                                    ")
print("----------------------------------------------------------------------")
print("Statistics for Ip ")
T_Ip.Table()
print("----------------------------------------------------------------------")
print("Statistics for Phit ")
T_Phit.Table()
print("----------------------------------------------------------------------")
print("Statistics for Phits")
T_Phits.Table()
print("----------------------------------------------------------------------")

#Descriptive Bivariate Statistics#
graf.Scater1(Ip,Phit,labelx='Ip',labely='Phit', color='black')
graf.pseudo_obs_scater1(marginal_data,Ip,Phit)
Ip_Phit=np.hstack((Ip.reshape(-1, 1),Phit.reshape(-1, 1)))
TB=graf.Stats_Bivariate(Ip_Phit)
Ip_Phits=Pared_s
TBs=graf.Stats_Bivariate(Ip_Phits)


print("----------------------------------------------------------------------")
print("Descriptive Bivariate Statistics:                                     ")
print("----------------------------------------------------------------------")
print("Statistics for Phit_Ip ")
TB.Table()
print("----------------------------------------------------------------------")
print("Statistics for Phits_Ip")
TBs.Table()
print("----------------------------------------------------------------------")

#Variogram calculation#
#Variogram initial# 
svt_smooth,_,_,_,lag_smooth=fo.variogram_parameter(Data_Grid,sill,nugget_var,a_range,lag_number*4,var_model=model,lag_size=lag_size/4)
vdata = variograms.semivariogram(P, lags, lag_tolerance)
h, sv = vdata[0], vdata[1] 
sv=sv-0.5*(h*pend)**2 
graf.Experimental_Variogram(lags, [svt_smooth, sv], sill, a_range,var_model=model,variance=T_Phit.variance_value,color_svt='red',lags_svt=lag_smooth)

#Variogram simulado#   
vdatas = variograms.semivariogram(Ps, lags, lag_tolerance)
hs, svs = vdatas[0], vdatas[1] 
svs=svs-0.5*(hs*pend)**2 
graf.Experimental_Variogram(lags, [svt_smooth, svs], sill, a_range,variance=T_Phit.variance_value,var_model=model,lags_svt=lag_smooth)

#Porosity# 
graf.Scater(Ip,Phits,Phit,labelx='Ip',labely1='Phits',labely2= 'Phit',Condition=Data_Conditioning[:,[2,3]])
graf.pseudo_obs_scater(marginal_data,marginal_s,Ip,Phit,Phits)

#Error of porosity#
error=(Phits-Phit)
Te=graf.Stats_Univariate(error)
Te.histogram_boxplot(error,xlabel='Error')

print("----------------------------------------------------------------------")
print("Error Statistics:                                    ")
print("----------------------------------------------------------------------")
Te.Table()
print("----------------------------------------------------------------------")

#Log_well
Data_well=Data[:,2:]
tracks=[[3,4],[2],[1]]
limits=[[(np.max(Data_well[:,3]).round(decimals=2),np.min(Data_well[:,3]).round(decimals=2)),(np.max(Data_well[:,4]).round(decimals=2),np.min(Data_well[:,4]).round(decimals=2))],[(np.max(Ip).round(decimals=2),np.min(Ip).round(decimals=2))],[(np.max(Phit).round(decimals=2),np.min(Phit).round(decimals=2))]]
labels=[['Rhob','Vp'],['Ip'],['Phit']]
color=[['orange','gray'],['k'],['r']]
graf.logview(Data_well,tracks,labels,title='Log well',limits=limits,colors=color)

#Log_porosity
Phit_log=np.array([Z,Phit,Phits,error]).T
Phit_log_cond=np.array([Data_Conditioning[:,1],Data_Conditioning[:,3],Data_Conditioning[:,3],0*Data_Conditioning[:,2]]).T
tracks=[[1,2],[3]]
limits=[[(np.min(Phit).round(decimals=2),np.max(Phit).round(decimals=2)),(np.min(Phit).round(decimals=2),np.max(Phit).round(decimals=2))],[(np.min(error).round(decimals=2),np.max(error).round(decimals=2))]]
labels=[['Phit','Phits'],['Error']]
color=[['black','lime'],['black']]
mean_log=[[T_Phits.mean,''],[Te.mean]]
median_log=[[T_Phits.median,''],[Te.median]]
graf.logview(Phit_log,tracks,labels,title='Log of porosity',limits=limits,colors=color,mean=mean_log,median=median_log,Condition=Phit_log_cond)


#Marginal,copula and bivariate distributions plots
graf.four_axis (marginal_data,Ip,Phit,copula_data, bivariate_distribution_data)
graf.four_axis (marginal_s,Ip,Phits,copula_s, bivariate_distribution_s) 
graf.cumul_four_axis (marginal_data,Ip,Phit,copula_data, bivariate_distribution_data) 
graf.cumul_four_axis (marginal_s,Ip,Phits,copula_s, bivariate_distribution_s) 

print("----------------------------------------------------------------------")
print("Bivariate Distributions from DATA:                                    ")
print("----------------------------------------------------------------------")
print("Marginal for Ip(data_grid) =",marginal_data[0]) 
print("Marginal for Phit(Var) =",marginal_data[1]) 
print("Copula for marginals =",copula_data)
print("----------------------------------------------------------------------")
print("Simulate Bivariate Distributions information:                         ")
print("----------------------------------------------------------------------")
print("Estimate marginal for Ip(data_grid) =",marginal_s[0]) 
print("Estimate marginal for Phit(Var) =",marginal_s[1])
print("Estimate copula for marginals =",copula_s)
print("----------------------------------------------------------------------")

#Conditional_PDF#
conditioned_pdf_s=cond.ot_compute_conditional(Phits,bivariate_distribution_s,Data_Grid) 
conditioned_pdf=cond.ot_compute_conditional(Phits,bivariate_distribution_data,Data_Grid)
graf.Scater(Phits,conditioned_pdf_s,conditioned_pdf)

#Conditional_CDF#
conditioned_cdf_s=cond.ot_compute_conditional_cdf(Phits,bivariate_distribution_s,Data_Grid) 
conditioned_cdf=cond.ot_compute_conditional_cdf(Phits,bivariate_distribution_data,Data_Grid)
graf.Scater(Phits,conditioned_cdf_s,conditioned_cdf, labely1='conditioned_cdf_s', labely2='conditioned_cdf')

#Empirical CDF
graf.emprical_CDF([Phits],[marginal_data[1]],colors=['r','k'])

#Save figures and log to files#
graf.multipage('Results/Figures_'+ output_namefile +'.pdf')
log_console.close()
