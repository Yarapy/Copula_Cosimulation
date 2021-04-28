###Spatial simulation program using differential evolution_scipy###

##Brief explanation of the method##
"""
scipy.optimize.differential_evolution

References:
1.-Storn, R and Price, K, Differential Evolution - a Simple and Efficient Heuristic 
for Global Optimization over Continuous Spaces, Journal of Global Optimization, 1997, 11, 341 - 359.
2.-http://www1.icsi.berkeley.edu/~storn/code.html
3.- http://en.wikipedia.org/wiki/Differential_evolution
4.- Wormington, M., Panaccione, C., Matney, K. M., Bowen, D. K., - Characterization of structures 
from X-ray scattering data using genetic algorithms, Phil. Trans. R. Soc. Lond. A, 1999, 357, 2827-2848
5.- Lampinen, J., A constraint handling approach for the differential evolution algorithm. 
Proceedings of the 2002 Congress on Evolutionary Computation. CEC‘02 (Cat. No. 02TH8600). 
Vol. 2. IEEE, 2002.

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

##Name of the files to save outputs##
#Logger modes: 'w' erase previous file, 'a' appending to the end of the file
output_namefile='GSA_Aplication_Noruego3'
log_console = graf.Logger('Results/Log_'+output_namefile+'.log', mode="w") 

##Load data## 
Data=lsd.Load_columns(6,[1,2,3,4,5,6],windowname='Select Data File',delimiter=',') #Data (Well 3 Noruego)
Data_r=lsd.Load_columns(7,[1,2,3,4,5,6,7],windowname='Select Data File reference',delimiter=',') #Data reference (Well 2Noruego) 
Data=Data[46:,:]
X=Data[:,0]
Z=Data[:,2]
Ip=Data[:,3]
Ip_r=Data_r[:,4]
Phit_r=Data_r[:,3]
Data_Grid=np.hstack((X.reshape(-1, 1),Z.reshape(-1, 1),Ip.reshape(-1, 1)))

##Variogram##
#Input parameters#
##Well2Noruego 
model='Spherical';sill=0.00093;nugget_var=0.0004;a_range=110;amplitude=0;w0=[1,0.001];U='(ft/s.g/cm3)';limit_Ip=(13981,26727);limit_Phit=(0.21,0.4);limit_error=(-0.13,0.13) 
Nobs=len(Data_Grid)
lag_number= 10
lag_size=0 #If lag_size=0 it is calculated automatically.
P=np.hstack((X.reshape(-1, 1),Z.reshape(-1, 1),Ip.reshape(-1, 1)))
Ip_detrend,pend_Ip,zero_Ip=variograms.detrend(P,amplitude=amplitude)
pend_Phit=pend_Ip/np.mean(Ip)
zero_Phit=np.mean(Phit_r)-pend_Phit*np.mean(Z)
#Teorical input semivariogram#
svt,dist_max,lag_size,lag_tolerance,lags=fo.variogram_parameter(Data_Grid,sill,nugget_var,a_range,lag_number,var_model=model,lag_size=lag_size)

##Bivariate input distribution##
#Noruego
marginal1=cond.ot_Marginals1(Ip)
muLog = -1.49583;sigmaLog = 0.134007;gamma = 0.0770454
marginal2= ot.LogNormal(muLog,sigmaLog,gamma)
theta =-7.83752
copula = ot.FrankCopula(theta)
bivariate_distribution_data = ot.ComposedDistribution([marginal1,marginal2], copula)
marginal_data=[bivariate_distribution_data.getMarginal(i) for i in [0,1]]
copula_data=bivariate_distribution_data.getCopula()

#Weights#
w1=[1,1]
w=[w0[0]*w1[0],w0[1]*w1[1]] 

##Objective function##
F=fo.funcobj(w, Data_Grid,svt,lag_tolerance,lags, bivariate_distribution_data, trend_coef=pend_Phit)

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
print("Trend slope =", pend_Phit)
print("Trend zero =", zero_Phit)
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
bounds=[limit_Phit for i in range(Nobs)] #Variable search space
maxiter=5 #Recommended maxiter=500 
#args
#local_search_options
initial_temp=16 #Temperature for Dreo
restart_temp_ratio=2e-5
visit=2.7 #The value range is (0, 3]. The value 1 is classic SA and 2 is fast SA.
accept=-5 #Default value is -5.0 with a range (-1e4, -5!!]. debería ser hasta 1
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
T_Ip.histogram_boxplot(Ip,xlabel='Ip (ft/s.g/cm3)',marginal=marginal_data[0],limit_x=limit_Ip)
T_Phit_r=graf.Stats_Univariate(Phit_r)
T_Phit_r.histogram_boxplot(Phit_r,xlabel='Phit_reference (v/v)',marginal=marginal_data[1],limit_x=limit_Phit)
T_Phits=graf.Stats_Univariate(Phits)
T_Phits.histogram_boxplot(Phits, xlabel='Phits (v/v)',marginal=marginal_s[1],limit_x=limit_Phit)

print("----------------------------------------------------------------------")
print("Descriptive Univariate Statistics:                                    ")
print("----------------------------------------------------------------------")
print("Statistics for Ip ")
T_Ip.Table()
print("----------------------------------------------------------------------")
print("Statistics for Phit of reference ")
T_Phit_r.Table()
print("----------------------------------------------------------------------")
print("Statistics for Phits")
T_Phits.Table()
print("----------------------------------------------------------------------")

#Descriptive Bivariate Statistics#
graf.Scater1(Ip_r,Phit_r,labelx='Ip_reference (ft/s.g/cm3)',labely='Phit_reference (v/v)', color='black')
graf.pseudo_obs_scater1(marginal_data,Ip_r,Phit_r,labelx='v(Ip_reference) (ft/s.g/cm3)',labely1='u(Phit_reference) (v/v)')
Ipr_Phitr=np.hstack((Ip_r.reshape(-1, 1),Phit_r.reshape(-1, 1))) 
TB=graf.Stats_Bivariate(Ipr_Phitr)
Ip_Phits=Pared_s
TBs=graf.Stats_Bivariate(Ip_Phits)


print("----------------------------------------------------------------------")
print("Descriptive Bivariate Statistics:                                     ")
print("----------------------------------------------------------------------")
print("Statistics for Ip_Phit of reference  ")
TB.Table()
print("----------------------------------------------------------------------")
print("Statistics for Ip_Phits")
TBs.Table()
print("----------------------------------------------------------------------")

#Variogram calculation#
#Variogram initial# 
svt_smooth,_,_,_,lag_smooth=fo.variogram_parameter(Data_Grid,sill,nugget_var,a_range,lag_number*4,var_model=model,lag_size=lag_size/4)
graf.Teorical_variogram(lag_smooth,svt_smooth,sill, a_range, var_model=model)

#Variogram simulado#  
svt_smooth,_,_,_,lag_smooth=fo.variogram_parameter(Data_Grid,sill,nugget_var,a_range,lag_number*4,var_model=model,lag_size=lag_size/4)
vdatas = variograms.semivariogram(Ps, lags, lag_tolerance)
hs, svs = vdatas[0], vdatas[1] 
svs=svs-0.5*(hs*pend_Phit)**2 
graf.Experimental_Variogram(lags, [svt_smooth, svs], sill, a_range,variance=T_Phits.variance_value,var_model=model,lags_svt=lag_smooth)

#Porosity
graf.Scater1(Ip,Phits,labelx='Ip (ft/s.g/cm3)',labely='Phits (v/v)', color='red')
graf.pseudo_obs_scater1(marginal_data,Ip,Phits,labely1='u(Phits)', color='red')
graf.pseudo_obs_scater1(marginal_data,Ip,Phits,labely1='u(Phits)', color='red')

#Log_well
Data_well=np.hstack((Data[:,2:],Phits.reshape(-1,1)))
tracks=[[2,3],[1]]
limits=[[(np.min(Data_well[:,2]).round(decimals=2),np.max(Data_well[:,2]).round(decimals=2)),(np.min(Data_well[:,3]).round(decimals=2),np.max(Data_well[:,3]).round(decimals=2))],[limit_Ip]]
labels=[['Rhob (g/cm3)','Vp (ft/s)'],['Ip (ft/s.g/cm3)']]
color=[['orange','gray'],['k']]
graf.logview(Data_well,tracks,labels,title='Log well',limits=limits,colors=color)

#Log_porosity
Data_log=np.array([Z,Ip,Phits]).T
tracks=[[1],[2]]
limits=[limit_Ip],[limit_Phit]
labels=[['Ip (ft/s.g/cm3)'],['Phits (v/v)']]
color=[['k'],['lime']]
mean_log=[[T_Ip.mean],[T_Phits.mean]]
median_log=[[T_Ip.median],[T_Phits.median]]
graf.logview(Data_log,tracks,labels,title='Log of porosity',limits=limits,colors=color,mean=mean_log,median=median_log)

#Marginal,copula and bivariate distributions plots
graf.four_axis (marginal_data,Ip_r,Phit_r,copula_data, bivariate_distribution_data,U)
graf.four_axis (marginal_s,Ip,Phits,copula_s, bivariate_distribution_s,U) 
graf.cumul_four_axis (marginal_data,Ip_r,Phit_r,copula_data, bivariate_distribution_data,U) 
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

#Conditional_PDF#
conditioned_pdf_s=cond.ot_compute_conditional(Phits,bivariate_distribution_s,Data_Grid) 
conditioned_pdf=cond.ot_compute_conditional(Phits,bivariate_distribution_data,Data_Grid)
graf.Scater(Phits,conditioned_pdf_s,conditioned_pdf,limit_x=limit_Phit)

#Conditional_CDF#
conditioned_cdf_s=cond.ot_compute_conditional_cdf(Phits,bivariate_distribution_s,Data_Grid) 
conditioned_cdf=cond.ot_compute_conditional_cdf(Phits,bivariate_distribution_data,Data_Grid)
graf.Scater(Phits,conditioned_cdf_s,conditioned_cdf, labely1='cdf_conditional_s', labely2='cdf_conditional',limit_x=limit_Phit)

#Empirical CDF
graf.emprical_CDF([Phits],[marginal_data[1]],colors=['r','k'],limit_x=limit_Phit)

#Save figures and log to files#
graf.multipage('Results/Figures_'+ output_namefile +'.pdf')
log_console.close()
