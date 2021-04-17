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
__date__      = "2020-03-10"
__copyright__ = "Copyright (C) 2020 Yarilis Gómez Martínez"
__license__   = "BSD 3-Clause"

##Modules##
import numpy as np
import time
from scipy.optimize import differential_evolution
import variograms
import LoadSave_data as lsd
import funcionesO as fo
import Grafics as graf
import openturns as ot
import ot_copula_conditional_YE as cond

##Name of the files to save outputs##
#Logger modes: 'w' erase previous file, 'a' appending to the end of the file
output_namefile='Dif_Evol_Noruego3_Ap'
log_console = graf.Logger('Results/Log_'+output_namefile+'.log', mode="w") 

##Load data## 
Data=lsd.Load_columns(6,[1,2,3,4,5,6],windowname='Select Data File',delimiter=',')
Data_r=lsd.Load_columns(7,[1,2,3,4,5,6,7],windowname='Select Data File reference',delimiter=',')
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
model='spherical';sill=0.00093;nugget_var=0.0004;a_range=110;amplitude=1;w0=[1,0.0001] 
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
Phit_min=0.1
Phit_max=0.4

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


"""-----------------------Method-Dif-Evol-----------------------------------"""

##Method input parameters##
bounds=[(Phit_min,Phit_max) for i in range(Nobs)] #Variable search space
#args
strategy='best1bin' #Default strategy
max_generations=3000
population_size=35 #Minimum is 5
tolerance=1e-10
mutation=0.5 #between (0,2)
recombination=0.3 #Probability 
#seed
disp=False
epsilon=1e-3 #Value from which the optimization is stopped
def callback_epsilon(xk,convergence):
    if F.funcO(xk)<epsilon:
        return True
#polish (The L-BFGS-B minimization method is used to polish the last member of the population.)
#Create a random array wich preserve conditional distribution from Pared values  
Phit_ini=[cond.ot_sample_conditional(bivariate_distribution_data,Data_Grid) for i in range(population_size)] 
initial=Phit_ini#It can be 'latinhypercube' (default), 'random' or array list.
initial=np.array(Phit_ini)#It can be 'latinhypercube' (default), 'random' or array list.
#atol
updating='deferred' #'deferred' or 'immediate'
workers=3 #-1 to use all available CPU cores
#constraints

print("----------------------------------------------------------------------")
print("Differential_evolution parameters:                                    ")
print("----------------------------------------------------------------------")
print("Objective Function: Variogram and Bivariate Distribution Function     ")
print("Bounds for variables =", bounds[0])
#print("args =", args)
print("Strategy =", strategy)
print("Maximum number of generations  =", max_generations)
print("Total population size =", population_size)
print("Relative tolerance =", tolerance)
print("Mutation constant =", mutation)
print("Recombination constant =", recombination)
#print("Seed =", seed)
print("Prints the evaluated func at every iteration. =", disp)
print("Minimization halted value =", epsilon)
#print("polish =", polish)
print("Type of population initialization =", initial)
print("----------------------------------------------------------------------")
print("Initial value of objective Initial")
print("  fun O1:",F.funcO2(initial[0])*w[0])
print("  fun O2:",F.funcO5(initial[0])*w[1])
print("  fun:",F.funcO(initial[0]))
print("----------------------------------------------------------------------")
#print("atol =", atol)
print("Updating =", updating)
print("Workers =", workers)
#print("constraints =", constraints)
print("----------------------------------------------------------------------")

##Result##
start_time = time.time()
ResultDE = differential_evolution(F.funcO,bounds,strategy=strategy, maxiter=max_generations,popsize=population_size,
                                tol=tolerance,mutation=mutation,recombination=recombination,disp=disp, callback=callback_epsilon,polish=False,
                                init=initial,updating=updating,workers=workers)
end_time=time.time() - start_time
ResultDEx=ResultDE.x

print("----------------------------------------------------------------------")
print("Differential_evolution result:                                        ")
print("----------------------------------------------------------------------")
print("Objective functions")
print("  fun O1:",F.funcO2(ResultDEx)*w[0])
print("  fun O2:",F.funcO5(ResultDEx)*w[1])
print(str(ResultDE).split('\n       x:')[0])
print("Execution time: %s seconds" % end_time)
print("----------------------------------------------------------------------")
print("Optimal solution \n x= \n[",end='')
print(*ResultDEx,sep=', ',end=']\n')
print("----------------------------------------------------------------------")

"""------------------------------------End-Dif-Evol-------------------------"""

###Analysis of the result##
#Load saved data (only if necessary)#
#Ps=lsd.Load_columns(3,[1,2,3],delimiter=' ',windowname='Select Results Data File')

#Variable#
Ps = np.hstack((Data_Grid[:,:2],ResultDEx.reshape(-1, 1)))
Phits=Ps[:,2]
Pared_s=np.hstack((Ip.reshape(-1, 1),Phits.reshape(-1, 1))) 
bivariate_distribution_s=cond.ot_copula_fit(Pared_s)
marginal_s=[bivariate_distribution_s.getMarginal(i) for i in [0,1]]
copula_s=bivariate_distribution_s.getCopula()

#Save variable#
lsd.Save_Data('Results/Result_'+output_namefile+'.dat',Ps, columns=["X", "Z", "Phits"])

#Descriptive Univariate Statistics#
T_Ip=graf.Stats_Univariate(Ip)
T_Ip.histogram_boxplot(Ip,xlabel='Ip',marginal=marginal_data[0])
T_Phit_r=graf.Stats_Univariate(Phit_r)
T_Phit_r.histogram_boxplot(Phit_r,xlabel='Phit_reference',marginal=marginal_data[1])
T_Phits=graf.Stats_Univariate(Phits)
T_Phits.histogram_boxplot(Phits, xlabel='Phits',marginal=marginal_s[1])

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
graf.Scater1(Ip_r,Phit_r,labelx='Ip_reference',labely='Phit_reference', color='black')
graf.Scater1(Ip,Phits,labelx='Ip',labely='Phits', color='red')
graf.pseudo_obs_scater1(marginal_data,Ip,Phits,labely1='u(Phits)', color='red')
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
#Variogram simulado#  
svt_smooth,_,_,_,lag_smooth=fo.variogram_parameter(Data_Grid,sill,nugget_var,a_range,lag_number*4,var_model=model,lag_size=lag_size/4)
vdatas = variograms.semivariogram(Ps, lags, lag_tolerance)
hs, svs = vdatas[0], vdatas[1] 
svs=svs-0.5*(hs*pend_Phit)**2 
graf.Experimental_Variogram(lags, [svt_smooth, svs], sill, a_range,variance=T_Phits.variance_value,var_model=model,lags_svt=lag_smooth)

#Log_well
Data_well=np.hstack((Data[:,2:],Phits.reshape(-1,1)))
tracks=[[2,3],[1]]
limits=[[(np.max(Data_well[:,2]).round(decimals=2),np.min(Data_well[:,2]).round(decimals=2)),(np.max(Data_well[:,3]).round(decimals=2),np.min(Data_well[:,3]).round(decimals=2))],[(np.max(Ip).round(decimals=2),np.min(Ip).round(decimals=2))]]
labels=[['Rhob','Vp'],['Ip']]
color=[['orange','gray'],['k']]
graf.logview(Data_well,tracks,labels,title='Log well',limits=limits,colors=color)

#Log_porosity
Data_log=np.array([Z,Ip,Phits]).T
tracks=[[1],[2]]
limits=[(np.min(Ip).round(decimals=2),np.max(Ip).round(decimals=2))],[(np.min(Phits).round(decimals=2),np.max(Phits).round(decimals=2))]
labels=[['Ip'],['Phits']]
color=[['k'],['lime']]
mean_log=[[T_Ip.mean],[T_Phits.mean]]
median_log=[[T_Ip.median],[T_Phits.median]]
graf.logview(Data_log,tracks,labels,title='Log of porosity',limits=limits,colors=color,mean=mean_log,median=median_log)

#Marginal,copula and bivariate distributions plots
graf.four_axis (marginal_data,Ip_r,Phit_r,copula_data, bivariate_distribution_data)
graf.four_axis (marginal_s,Ip,Phits,copula_s, bivariate_distribution_s) 
graf.cumul_four_axis (marginal_data,Ip_r,Phit_r,copula_data, bivariate_distribution_data) 
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
