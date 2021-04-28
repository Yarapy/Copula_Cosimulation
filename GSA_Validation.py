###Spatial simulation program using dual annealing_scipy###

##Brief explanation of the method##
"""
scipy.optimize.dual_annealing

"""
__author__    = "Yarilis Gómez Martínez (yarilisgm@gmail.com)"
__date__      = "2021"
__copyright__ = "Copyright (C) 2021 Yarilis Gómez Martínez"
__license__   = "GNU GPL Version 3.0"

##Modules##
import numpy as np
import time
import dual_annealing as da
import variograms
import LoadSave_data as lsd
import funcionesO as fo
import Grafics as graf
import ot_copula_conditional_YE as cond
import openturns as ot

#Name of the files to save outputs##
#Logger modes: 'w' erase previous file, 'a' appending to the end of the file
output_namefile='GSA_Validation_Noruego2'
log_console = graf.Logger('Results/Log_'+output_namefile+'.log', mode="w")

##Load data## 
Data=lsd.Load_columns(7,[1,2,3,4,5,6,7],windowname='Select Data File',delimiter=',')
X=Data[:,0]
Z=Data[:,2]
Ip=Data[:,4]
Phit=Data[:,3] 
Data_Grid=np.hstack((X.reshape(-1, 1),Z.reshape(-1, 1),Ip.reshape(-1, 1)))
Ip_Phit=np.hstack((Ip.reshape(-1, 1),Phit.reshape(-1, 1)))
 
##Variogram##
P=np.hstack((X.reshape(-1, 1),Z.reshape(-1, 1),Phit.reshape(-1, 1)))###
#Input parameters#
##Lakach1 
#model='Spherical';sill=0.00112;nugget_var=0.0007;a_range=52;amplitude=0 ;w0=[1,0.001];U='(m/s.g/cm3)';limit_Ip=(5324,11612);limit_Phit=(0.05,0.29);limit_error=(-0.15,0.15)#w0=[1.0173e-1,1.0723e-3]
#Well2Noruego 
model='Spherical';sill=0.00093;nugget_var=0.0004;a_range=110;amplitude=0;w0=[1,0.001];U='(ft/s.g/cm3)';limit_Ip=(13981,26727);limit_Phit=(0.21,0.4);limit_error=(-0.13,0.13) 
#BF3
#model='Gaussian';sill=0.0035;nugget_var=0.002;a_range=55;amplitude=0.8;w0=[1,0.001];U='(ft/s.g/cm3)';limit_Ip=(24289,46654);limit_Phit=(0.05,0.34);limit_error=(-0.2,0.2) 
Nobs=len(Data_Grid)
lag_number= 10
lag_size=0
Phit_detrend,pend,zero=variograms.detrend(P,amplitude=amplitude)###Trend amplitude=0, Detrend amplitude=1
#Teorical input semivariogram#
svt,dist_max,lag_size,lag_tolerance,lags=fo.variogram_parameter(Data_Grid,sill,nugget_var,a_range,lag_number,var_model=model,lag_size=lag_size)

##Bivariate input distribution##
#Lakach
# muLog = 7.43459;sigmaLog = 0.555439;gamma = 4977.04
# marginal1 =ot.LogNormal(muLog, sigmaLog, gamma)
# mu = 0.165352;beta = 0.0193547;
# marginal2 =ot.Logistic(mu, beta)
# theta = -4.2364
# copula = ot.FrankCopula(theta)

#Noruego
mu = 21929.5;sigma = 2251.15
marginal1= ot.Normal(mu,sigma)
muLog = -1.49583;sigmaLog = 0.134007;gamma = 0.0770454
marginal2= ot.LogNormal(muLog,sigmaLog,gamma)
theta =-7.83752
copula = ot.FrankCopula(theta )

#BF3
# beta1 = 2458.48;gamma1 = 28953.5
# marginal1= ot.Gumbel(beta1, gamma1)
# beta2 = 0.0489963;gamma2 = 0.156505
# marginal2= ot.Gumbel(beta2, gamma2)
# theta = -5.21511
# copula= ot.FrankCopula(theta)

#bivariate_distribution_data=cond.ot_kernel_copula_fit(Ip_Phit) #Nonparametric variant
bivariate_distribution_data = ot.ComposedDistribution([marginal1,marginal2], copula) #Parametric variant
marginal_data=[bivariate_distribution_data.getMarginal(i) for i in [0,1]]
copula_data=bivariate_distribution_data.getCopula()

#Weights#
w1=[1,1]
w=[w0[0]*w1[0],w0[1]*w1[1]] 

##Objective function##
F=fo.funcobj(w, Data_Grid,svt,lag_tolerance,lags, bivariate_distribution_data, trend_coef=pend)

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
print("Weights:                                                              ")
print("----------------------------------------------------------------------")
print("Variogram weight =",w[0]) 
print("Distribution weight =",w[1]) 
print("----------------------------------------------------------------------")

"""-----------------------Method-Dual-Ann-----------------------------------"""
##Method input parameters##
bounds=[limit_Phit for i in range(Nobs)] #Variable search space
maxiter=5 #Recommended maxiter=500 
#args
#local_search_options
initial_temp=16 #Temperature for Dreo
restart_temp_ratio=2e-5
visit=2.7 #The value range is (0, 3]. The value 1 is classic SA and 2 is fast SA.
accept=-5 #Default value is -5.0 with a range (-1e4, -5!!]. 
maxfun=1e7
#seed
no_local_search=True
epsilon=1e-3 #Value from which the optimization is stopped
call_iter=[]
def callback_epsilon(x, f, context, iteration):
    call_iter.append(iteration)
    if f<epsilon:
        return True
#Create a random array wich preserve conditional distribution from Pared values
Phit_ini=cond.ot_sample_conditional(bivariate_distribution_data,Data_Grid)  
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
ResultDAx=ResultDA.x

print("----------------------------------------------------------------------")
print("Dual_Anneling result:                                                 ")
print("----------------------------------------------------------------------")
print("Objective functions")
print("  fun O1:",F.funcO2(ResultDAx)*w[0])
print("  fun O2:",F.funcO5(ResultDAx)*w[1])
print(str(ResultDA).split('\n       x:')[0])
print("Execution time: %s seconds" % end_time)
print("----------------------------------------------------------------------")
print("Optimal solution \n x= \n[",end='')
print(*ResultDAx,sep=', ',end=']\n')
print("----------------------------------------------------------------------")

"""------------------------------------End-Dual-Ann-------------------------"""

###Analysis of the result##
#Variable#
#Load saved data (only if necessary)#
#Ps=lsd.Load_columns(3,[1,2,3],delimiter=' ',windowname='Select Results Data File')
Ps = np.hstack((Data_Grid[:,:2],ResultDAx.reshape(-1, 1)))
Phits=Ps[:,2]
Pared_s=np.hstack((Ip.reshape(-1, 1),Phits.reshape(-1, 1))) 
bivariate_distribution_s=cond.ot_copula_fit(Pared_s)
marginal_s=[bivariate_distribution_s.getMarginal(i) for i in [0,1]]
copula_s=bivariate_distribution_s.getCopula()

#Save variable#
lsd.Save_Data('Results/Result_'+output_namefile+'.dat',Ps, columns=["X", "Z", "Phits"])

#Descriptive Univariate Statistics#
T_Ip=graf.Stats_Univariate(Ip)
T_Ip.histogram_boxplot(Ip,xlabel='Ip '+U,marginal=marginal_data[0],limit_x=limit_Ip)
T_Phit=graf.Stats_Univariate(Phit)
T_Phit.histogram_boxplot(Phit,xlabel='Phit (v/v)',marginal=marginal_data[1],limit_x=limit_Phit)
T_Phits=graf.Stats_Univariate(Phits)
T_Phits.histogram_boxplot(Phits, xlabel='Phits (v/v)',marginal=marginal_s[1],limit_x=limit_Phit)

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
graf.Scater1(Ip,Phit,labelx='Ip '+U,labely='Phit (v/v)', color='black')
graf.pseudo_obs_scater1(marginal_data,Ip,Phit)
TB=graf.Stats_Bivariate(Ip_Phit)
Ip_Phits=Pared_s
TBs=graf.Stats_Bivariate(Ip_Phits)

print("----------------------------------------------------------------------")
print("Descriptive Bivariate Statistics:                                     ")
print("----------------------------------------------------------------------")
print("Statistics for Ip_Phit ")
TB.Table()
print("----------------------------------------------------------------------")
print("Statistics for Ip_Phits")
TBs.Table()
print("----------------------------------------------------------------------")

#Variogram calculation#
#Variogram initial
lag_number= 50
svt,dist_max,lag_size,lag_tolerance,lags=fo.variogram_parameter(Data_Grid,sill,nugget_var,a_range,lag_number,var_model=model)   
svt_smooth,_,_,_,lag_smooth=fo.variogram_parameter(Data_Grid,sill,nugget_var,a_range,lag_number*4,var_model=model,lag_size=lag_size/4)
vdata = variograms.semivariogram(P, lags, lag_tolerance)
h, sv = vdata[0], vdata[1] 
sv=sv-0.5*(h*pend)**2 
graf.Experimental_Variogram(lags, [svt_smooth, sv], sill, a_range,var_model=model,variance=T_Phit.variance_value,color_svt='red',lags_svt=lag_smooth)

#Variogram simulated
lag_number= 10
svt,dist_max,lag_size,lag_tolerance_1,lags=fo.variogram_parameter(Data_Grid,sill,nugget_var,a_range,lag_number,var_model=model)   
svt_smooth,_,_,_,lag_smooth=fo.variogram_parameter(Data_Grid,sill,nugget_var,a_range,lag_number*4,var_model=model,lag_size=lag_size/4)
vdatas = variograms.semivariogram(Ps, lags, lag_tolerance_1)
hs, svs = vdatas[0], vdatas[1] 
svs=svs-0.5*(hs*pend)**2 
graf.Experimental_Variogram(lags, [svt_smooth, svs], sill, a_range,variance=T_Phit.variance_value,var_model=model,lags_svt=lag_smooth)

#Porosity# 
graf.Scater(Ip,Phits,Phit,labelx='Ip '+U,labely1='Phits (v/v)',labely2= 'Phit (v/v)')
graf.pseudo_obs_scater(marginal_data,marginal_s,Ip,Phit,Phits)

#Error of porosity
error=(Phits-Phit)
Te=graf.Stats_Univariate(error)
Te.histogram_boxplot(error,xlabel='Error',limit_x=limit_error)

print("----------------------------------------------------------------------")
print("Error Statistics:                                    ")
print("----------------------------------------------------------------------")
Te.Table()
print("----------------------------------------------------------------------")

#Log_well
Data_well=Data[:,2:]
tracks=[[3,4],[2],[1]]
limits=[[(np.min(Data_well[:,3]).round(decimals=2),np.max(Data_well[:,3]).round(decimals=2)),(np.min(Data_well[:,4]).round(decimals=2),np.max(Data_well[:,4]).round(decimals=2))],[limit_Ip],[limit_Phit]]
labels=[['Rhob (g/cm3)','Vp (ft/s)'],['Ip (ft/s.g/cm3)'],['Phit (v/v)']]
color=[['orange','gray'],['k'],['r']]
#For Lakach
# tracks=[[2],[1]] 
# limits=[[limit_Ip],[limit_Phit]
# labels=[['Ip (m/s.g/cm3)'],['Phit (v/v)']]
# color=[['k'],['r']] 
graf.logview(Data_well,tracks,labels,title='Log well',limits=limits,colors=color)

#Log_porosity
Data_log=np.array([Z,Phit,Phits,error]).T
tracks=[[1,2],[3]]
limits=[[limit_Phit,limit_Phit],[limit_error]]
labels=[['Phit (v/v)','Phits (v/v)'],['Error']]
color=[['black','lime'],['black']]
mean_log=[[T_Phits.mean,''],[Te.mean]]
median_log=[[T_Phits.median,''],[Te.median]]
graf.logview(Data_log,tracks,labels,title='Log of Porosity ',limits=limits,colors=color,mean=mean_log,median=median_log)

#Marginal,copula and bivariate distributions plots#
graf.four_axis (marginal_data,Ip,Phit,copula_data, bivariate_distribution_data,U)
graf.four_axis (marginal_s,Ip,Phits,copula_s, bivariate_distribution_s,U) 
graf.cumul_four_axis (marginal_data,Ip,Phit,copula_data, bivariate_distribution_data,U) 
graf.cumul_four_axis (marginal_s,Ip,Phits,copula_s, bivariate_distribution_s,U) 

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

#Conditional#
#Conditional_PDF
conditioned_pdf_s=cond.ot_compute_conditional(Phits,bivariate_distribution_s,Data_Grid) 
conditioned_pdf=cond.ot_compute_conditional(Phits,bivariate_distribution_data,Data_Grid)
graf.Scater(Phits,conditioned_pdf_s,conditioned_pdf,limit_x=limit_Phit)

#Conditional_CDF
conditioned_cdf_s=cond.ot_compute_conditional_cdf(Phits,bivariate_distribution_s,Data_Grid) 
conditioned_cdf=cond.ot_compute_conditional_cdf(Phits,bivariate_distribution_data,Data_Grid)
graf.Scater(Phits,conditioned_cdf_s,conditioned_cdf, labely1='cdf_conditional_s', labely2='cdf_conditional',limit_x=limit_Phit)

#Empirical CDF
graf.emprical_CDF([Phits],[marginal_data[1]],colors=['r','k'], limit_x=limit_Phit)

#Save figures and log to files#
graf.multipage('Results/Figures_'+ output_namefile +'.pdf')
log_console.close()
