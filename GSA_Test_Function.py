###Differential Evolution Validation###
##Brief explanation of the method##
"""
Validation will be done for various functions.
"""
__author__    = "Yarilis Gómez Martínez (yarilisgm@gmail.com)"
__date__      = "2021"
__copyright__ = "Copyright (C) 2021 Yarilis Gómez Martínez"
__license__   = "GNU GPL Version 3.0"


##Modules##
import numpy as np
import dual_annealing as da
import time
import Grafics as graf

##Name of the files to save outputs##
#Logger modes: 'w' erase previous file, 'a' appending to the end of the file
output_namefile='GSA_Test_Function'
log_console = graf.Logger('Results/Log_'+output_namefile+'.log', mode="w") 


##Define the objective function##
#val=variable of the function to be optimized 
#Ackley Function
solA=np.array([ 0, 0])
fA=0
boundsA = [(-5, 5), (-5, 5)]
def Ackley(valA):
    x=valA
    arg1 = -0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))
    arg2 = 0.5 * (np.cos(2. * np.pi * x[0]) + np.cos(2. * np.pi * x[1]))
    zA=-20. * np.exp(arg1) + 20. + np.e - np.exp(arg2) 
    return zA-fA

#Rastrigin Function
solR=np.array([ 0, 0])
fR=0
n=2
boundsR=[(-5.12, 5.12) for i in range(0,n,1)]
def Rastrigin (valR):
    x=valR
    n=len(x)
    zR=10*n+np.sum(x**2-10*np.cos(2*np.pi*x))
    return zR-fR

#Rosenbrok Function
solB=np.array([ 1, 1])
fB=0
def Rosenbrok (valB):
    x0=valB[:-1]
    x1=valB[1:]
    zB=np.sum(100*(x1-x0**2)**2+(1-x0)**2)
    return zB-fB

##Method input parameters##
maxiter=2000
#args
#local_search_options
initial_temp=25
initial_tempB=10000
restart_temp_ratio=2e-7
visit=2.7 #The value range is (0, 3]. The value 1 is classic SA and 2 is fast SA.
accept=-5 #Default value is -5.0 with a range (-1e4, -5].
#maxfun
#seed
no_local_search=True
##Callback##
epsilon=1e-16

eA=[]
end_timeA=[]
call_iterA=[]
def callback_A(x, f, context, iteration):
    eA.append(np.sum((x-solA)**2))
    end_timeA.append (time.time() - start_time)
    call_iterA.append(iteration)
    if np.abs(f)<epsilon:
        return True
eR=[]
end_timeR=[]
call_iterR=[]
def callback_R(x, f, context, iteration):
    eR.append(np.sum((x-solR)**2))
    end_timeR.append (time.time() - start_time)
    call_iterR.append(iteration)
    if np.abs(f)<epsilon:
        return True
eB=[]
end_timeB=[]
call_iterB=[]
def callback_B(x, f, context, iteration):
    eB.append(np.sum((x-solB)**2))
    end_timeB.append (time.time() - start_time)
    call_iterB.append(iteration)
    if np.abs(f)<epsilon:
        return True
#x0
        
print("----------------------------------------------------------------------------------")
print("Dual_Anneling parameters:                                                ")
print("----------------------------------------------------------------------------------")
print("Objective Function: Peaks, Ackley, Rastrigin and Rosenbrok Functions")
print("Bounds for the variables of the Ackley function =", boundsA[0])
print("Bounds for the variables of the Rastrigin and Rosenbrok function =", boundsR[0])
#print("args =", args)
print("Maximum number of global search iterations  =", maxiter)
#print("Local search options =", local_search_options)
print("Initial temperature of the Peaks, Ackley and Rastrigin function  =", initial_temp)
print("Initial temperature of the Rosenbrok function =", initial_tempB)
print("Restart temperature ratio=", restart_temp_ratio)
print("Parameter for visiting distribution =", visit)
print("Parameter for acceptance distribution =", accept)
#print(" maxfun=", maxfun)
#print("Seed =", seed)
print("No local search =", no_local_search)
print("Minimization halted value =", epsilon)
#print("x0 =", x0)
print("----------------------------------------------------------------------------------")


##Result##

#Result for the  Ackley function 
start_time = time.time()
ResultA = da.dual_annealing(Ackley,boundsA,maxiter=maxiter, initial_temp=initial_temp,restart_temp_ratio=restart_temp_ratio,
                                visit=visit,accept=accept,no_local_search=no_local_search, callback=callback_A)
end_timeA_Mem=time.time() - start_time

#Result for the  Rastrigin function #iter=151 for initial_temp=22
start_time = time.time()
ResultR = da.dual_annealing(Rastrigin,boundsR,maxiter=maxiter, initial_temp=initial_temp,restart_temp_ratio=restart_temp_ratio,
                                visit=visit,accept=accept,no_local_search=no_local_search, callback=callback_R)
end_timeR_Mem=time.time() - start_time

#Result for the  Rosenbrok function 
start_time = time.time()
ResultB = da.dual_annealing(Rosenbrok,boundsR,maxiter=maxiter, initial_temp=initial_tempB,restart_temp_ratio=restart_temp_ratio,
                                visit=visit,accept=accept,no_local_search=no_local_search, callback=callback_B)
end_timeB_Mem=time.time() - start_time
   
print("----------------------------------------------------------------------------------")
print("Dual_Annealing result:                                                    ")
print("----------------------------------------------------------------------------------")
print("Result for the  Ackley function\n", ResultA)
print("----------------------------------------------------------------------------------")
print("Result for the  Rastrigin function\n", ResultR)
print("----------------------------------------------------------------------------------")
print("Result for the  Rosenbrok function\n", ResultB)
print("----------------------------------------------------------------------------------")

##Error graph##
n_iterations_A=call_iterA#[10::10]#np.array(range(10,len(eA),10))
eA_list=eA#[10::10]
endtimeA_list=end_timeA#[10::10]
text=graf.message_convert(ResultA,epsilon)
graf.Two_axes_plot(n_iterations_A,eA_list,endtimeA_list,"Error graph for the Ackley function",
                   'n_iterations',y1label='error', y2label='Time (s)', text=text)

n_iterations_R=call_iterR#[10::10]#np.array(range(10,len(eR),10))
eR_list=eR#[10::10]
endtimeR_list=end_timeR#[10::10]
text=graf.message_convert(ResultR,epsilon)
graf.Two_axes_plot(n_iterations_R,eR_list,endtimeR_list,"Error graph for the Rastrigin function",
                   'n_iterations',y1label='error', y2label='Time (s)', text=text)


n_iterations_B=call_iterB#[10::10]#np.array(range(10,len(eB),10))
eB_list=eB#[10::10]
endtimeB_list=end_timeB#[10::10]
text=graf.message_convert(ResultB,epsilon)
graf.Two_axes_plot(n_iterations_B,eB_list,endtimeB_list,"Error graph for the Rosenbrok function",
                   'n_iterations',y1label='error', y2label='Time (s)', text=text)

##Table##
##Analyzing max_generations##
#Initialize#
n=50
j=0
resultAT=[0]*n
end_timeAT=[0]*n
resultRT=[0]*n
end_timeRT=[0]*n
resultBT=[0]*n
end_timeBT=[0]*n
resultET=[0]*n
end_timeET=[0]*n

#Iterating max_generations#
for i in range(0,n):
    
    #Result for the  Ackley function 
    start_time = time.time()
    resultAT[j] = da.dual_annealing(Ackley,boundsA,maxiter=maxiter, initial_temp=initial_temp,restart_temp_ratio=restart_temp_ratio,
                                visit=visit,accept=accept,no_local_search=no_local_search, callback=callback_A)
    end_timeAT[j]=time.time() - start_time
    
    #Result for the  Rastrigin function 
    start_time = time.time()
    resultRT[j] = da.dual_annealing(Rastrigin,boundsR,maxiter=maxiter, initial_temp=initial_temp,restart_temp_ratio=restart_temp_ratio,
                                visit=visit,accept=accept,no_local_search=no_local_search, callback=callback_R)
    end_timeRT[j]=time.time() - start_time
        
    #Result for the  Rosenbrok function 
    start_time = time.time()
    resultBT[j] = da.dual_annealing(Rosenbrok,boundsR,maxiter=maxiter, initial_temp=initial_tempB,restart_temp_ratio=restart_temp_ratio,
                                visit=visit,accept=accept,no_local_search=no_local_search, callback=callback_B)
    end_timeBT[j]=time.time() - start_time
    j=j+1

xA=np.array([res.x for res in resultAT])
eAT=np.sum((xA-solA)**2, axis=1)
funA=np.array([res.fun for res in resultAT])
nitA=np.array([res.nit for res in resultAT])
TeA=graf.Stats_Univariate(eAT)
TfunA=graf.Stats_Univariate(funA)
TnitA=graf.Stats_Univariate(nitA)
Tend_timeA=graf.Stats_Univariate(end_timeAT)

xR=np.array([res.x for res in resultRT])
eRT=np.sum((xR-solR)**2, axis=1)
funR=np.array([res.fun for res in resultRT])
nitR=np.array([res.nit for res in resultRT])
TeR=graf.Stats_Univariate(eRT)
TfunR=graf.Stats_Univariate(funR)
TnitR=graf.Stats_Univariate(nitR)
Tend_timeR=graf.Stats_Univariate(end_timeRT)

xB=np.array([res.x for res in resultBT])
eBT=np.sum((xB-solB)**2, axis=1)
funB=np.array([res.fun for res in resultBT])
nitB=np.array([res.nit for res in resultBT])
TeB=graf.Stats_Univariate(eBT)
TfunB=graf.Stats_Univariate(funB)
TnitB=graf.Stats_Univariate(nitB)
Tend_timeB=graf.Stats_Univariate(end_timeBT)


print("-------------------------------------------------------------------------")
print("Descriptive statistics                                                   ")
print("-------------------------------------------------------------------------")
print("Ackley Function                                                          ")
print("-------------------------------------------------------------------------")
print("Error")
TeA.Table()
print("-------------------------------------------------------------------------")
print("Function value")
TfunA.Table()
print("-------------------------------------------------------------------------")
print("Iterations number")
TnitA.Table()
print("-------------------------------------------------------------------------")
print("Execution time")
Tend_timeA.Table()
print("-------------------------------------------------------------------------")
print("Rastrigin Function                                                       ")
print("-------------------------------------------------------------------------")
print("Error")
TeR.Table()
print("-------------------------------------------------------------------------")
print("Function value")
TfunR.Table()
print("-------------------------------------------------------------------------")
print("Iterations number")
TnitR.Table()
print("-------------------------------------------------------------------------")
print("Execution time")
Tend_timeR.Table()
print("-------------------------------------------------------------------------")
print("Rosenbrok Function                                                       ")
print("-------------------------------------------------------------------------")
print("Error")
TeB.Table()
print("-------------------------------------------------------------------------")
print("Function value")
TfunB.Table()
print("-------------------------------------------------------------------------")
print("Iterations number")
TnitB.Table()
print("-------------------------------------------------------------------------")
print("Execution time")
Tend_timeB.Table()

#Save figures and log to files#
graf.multipage('Results/Figures_'+ output_namefile +'.pdf')
log_console.close()
