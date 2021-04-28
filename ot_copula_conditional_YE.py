###Create conditionl distributions using estimte marginal and copula_openturns###


##Brief explanation of the method##
"""
Estimate a multivariate distribution_http://openturns.github.io/openturns/master/examples/data_analysis/estimate_multivariate_distribution.html?highlight=getcontinuousunivariatefactories

"""
__author__    = "Yarilis Gómez Martínez (yarilisgm@gmail.com)"
__date__      = "2021"
__copyright__ = "Copyright (C) 2021 Yarilis Gómez Martínez"
__license__   = "GNU GPL Version 3.0"


##Modules##
import numpy as np
import openturns as ot


def best_fit_distribution(sample):
    
    ##Best holders##
    best_distribution_ot = ot.NormalFactory()
    best_BIC_ot = np.inf
    marginalFactories = [] # the next cicle generate all available marginals in OT
    
    ##Create list distributions##
    #List of admisible Distribution:
    #[class=DistributionFactory implementation=class=ArcsineFactory,
    #class=DistributionFactory implementation=class=BetaFactory,
    #class=DistributionFactory implementation=class=ChiFactory,
    #class=DistributionFactory implementation=class=ExponentialFactory,
    #class=DistributionFactory implementation=class=GammaFactory,
    #class=DistributionFactory implementation=class=GumbelFactory,
    #class=DistributionFactory implementation=class=LaplaceFactory,
    #class=DistributionFactory implementation=class=LogisticFactory,
    #class=DistributionFactory implementation=class=LogNormalFactory,
    #class=DistributionFactory implementation=class=NormalFactory,
    #class=DistributionFactory implementation=class=RayleighFactory,
    #class=DistributionFactory implementation=class=RiceFactory,
    #class=DistributionFactory implementation=class=UniformFactory]
    for factory in ot.DistributionFactory.GetContinuousUniVariateFactories():
        if str(factory).startswith('Histogram'):
            # ~ non-parametric
            continue
        if str(factory).startswith('Burr'):
            # Error
            continue
        if str(factory).startswith('ChiSquare'):
            # Error
            continue
        if str(factory).startswith('Dirichlet'):
            # Error
            continue
        if str(factory).startswith('FisherSnedecor'):
            #Error
            continue
        if str(factory).startswith('GeneralizedPareto'):
            #Error
            continue
        if str(factory).startswith('InverseNormal'):
            # Error
            continue
        if str(factory).startswith('LogUniform'):
            #Error
            continue
        if str(factory).startswith('MeixnerDistribution'):
            # Error
            continue
        if str(factory).startswith('Pareto'):
            # Error
            continue
        if str(factory).startswith('Rice'): 
            # Error
            continue
        if str(factory).startswith('Triangular'):
            # Error
            continue
        if str(factory).startswith('Frechet'):
            # Error
            continue
        if str(factory).startswith('Student'):
            # Error
            continue
        if str(factory).startswith('Trapezoidal'):
            # Error
            continue
        if str(factory).startswith('TruncatedNormal'):
            # Error
            continue
        if str(factory).startswith('WeibullMax'):
            # Error
            continue
        if str(factory).startswith('WeibullMin'):
            # Error
            continue
        marginalFactories.append(factory)
            
    for distribution in marginalFactories:
        # Calculate Bayesian information criterion
        ot_dist,BIC=ot.FittingTest.BIC(sample, distribution)
        # identify if this distribution is better
        if best_BIC_ot > BIC:# > 0:
            best_distribution_ot = ot_dist
            best_BIC_ot = BIC
    best_distribution=best_distribution_ot
    best_BIC=best_BIC_ot 
    return (best_distribution, best_BIC)

def ot_Marginals1(Var):
    Pared=np.hstack((Var.reshape(-1, 1),0*Var.reshape(-1, 1)))
    sample=ot.Sample(Pared)
    marginals = best_fit_distribution(sample.getMarginal(0))[0] 
    return marginals

def ot_Marginals(sample):
    dimension = sample.getDimension()
    marginals = [best_fit_distribution(sample.getMarginal(i))[0] for i in range(dimension)]
    return marginals

def ot_copula_fit(Pared): 
    sample_Pared = ot.Sample(Pared)
    marginals=ot_Marginals(sample_Pared)
    
    blocs = [0, 1]
    copulaFactories = []
    
    ##Create list coula##
    #List of admisible Copulas:
    #[class=DistributionFactory implementation=class=ClaytonCopulaFactory,
    #class=DistributionFactory implementation=class=FrankCopulaFactory,
    #class=DistributionFactory implementation=class=NormalCopulaFactory,
    #class=DistributionFactory implementation=class=PlackettCopulaFactory]
    for factory in ot.DistributionFactory.GetContinuousMultiVariateFactories():
        if not factory.build().isCopula():
            continue
        if factory.getImplementation().getClassName()=='BernsteinCopulaFactory':
            continue
        if factory.getImplementation().getClassName()=='AliMikhailHaqCopulaFactory':
            # This Copula throw a WARNING
            continue
        if factory.getImplementation().getClassName()=='GumbelCopulaFactory':
            # This Copula throw a WARNING
            continue
        if factory.getImplementation().getClassName()=='FarlieGumbelMorgensternCopulaFactory':
            # This Copula throw a WARNING
            continue
        #if factory.getImplementation().getClassName()=='FrankCopulaFactory':
            # This Copula throw a WARNING
         #   continue
        if factory.getImplementation().getClassName()=='ClaytonCopulaFactory':
            # This Copula throw a WARNING
           continue
        copulaFactories.append(factory)
        
    #Fit copula    
    copulas_fit = ot.FittingTest.BestModelBIC(sample_Pared.getMarginal(blocs), copulaFactories)[0]
     
    # Build joint distribution from marginal distributions and dependency structure
    bivariate_distribution_fit = ot.ComposedDistribution(marginals, copulas_fit)
    
    return bivariate_distribution_fit

#-----------------------------------------------------------------------------#
    
def kernel_fit_distribution(sample):
    var=sample[:,0]
    #Define the type of kernel
    kernel_distribution = ot.Epanechnikov()
    # Estimate Kernel Smoothing marginals
    kernel_function = ot.KernelSmoothing(kernel_distribution)
    kernel_distribution = kernel_function.build(var)
    return kernel_distribution


def ot_kernel_Marginals(sample):
    dimension = sample.getDimension()
    marginals = [kernel_fit_distribution(sample.getMarginal(i)) for i in range(dimension)]
    return marginals

def ot_kernel_copula_fit(Pared): 
    kernel_distribution = ot.Epanechnikov()
    sample_Pared = ot.Sample(Pared)
    marginals=ot_kernel_Marginals(sample_Pared)
    KernelSmoothing_copula_distribution = ot.KernelSmoothing(kernel_distribution).build(sample_Pared).getCopula()
    bivariate_distribution=ot.ComposedDistribution(marginals,KernelSmoothing_copula_distribution)
    return bivariate_distribution

def ot_bernstein_copula_fit(Pared): 
    sample_Pared = ot.Sample(Pared)
    marginals=ot_kernel_Marginals(sample_Pared)
    ranksTransf = ot.MarginalTransformationEvaluation(marginals, ot.MarginalTransformationEvaluation.FROM)
    rankSample = ranksTransf(sample_Pared)
    bernstein_copula_distribution = ot.BernsteinCopulaFactory().build(rankSample)
    bivariate_distribution=ot.ComposedDistribution(marginals,bernstein_copula_distribution)
    return bivariate_distribution

#-----------------------------------------------------------------------------#

def ot_compute_PDF(vals,distribution): 
    #vals=np.atleast_2d(vals)
    pdf=0.0*vals[0]
    for i in range(len(pdf)):
        pdfval=distribution.computePDF([float(v[i]) for v in vals])
        pdf[i]=pdfval 
    return pdf

def ot_compute_conditional(val,bivariate_distribution,Data_Grid,order=0):
    condition=Data_Grid[:,2]
    marginal_cond=bivariate_distribution.getMarginal(order) 
    pdf_C=ot_compute_PDF([condition],marginal_cond) 
    if order==0:
        pdf_2=ot_compute_PDF([condition,val],bivariate_distribution) 
    else:
        pdf_2=ot_compute_PDF([val,condition],bivariate_distribution) 
    conditioned_pdf=pdf_2/pdf_C
    return conditioned_pdf

def ot_compute_conditional_cdf(val,bivariate_distribution,Data_Grid):
    condition=Data_Grid[:,2]
    conditioned_cdf=0.0*val
    for i in range(len(conditioned_cdf)):
        cdfval=bivariate_distribution.computeConditionalCDF(float(val[i]),[float(condition[i])])
        conditioned_cdf[i]=cdfval 
    return conditioned_cdf

def ot_sample_conditional(bivariate_distribution,Data_Grid): 
    Var1=Data_Grid[:,2]
    uniformSample = np.random.random(len(Var1))
    marginal1=bivariate_distribution.getMarginal(0)
    marginal2=bivariate_distribution.getMarginal(1)
    copula=bivariate_distribution.getCopula()
    Var2=[]
    for i in range(len(Var1)):
        condition=ot.Point(1,marginal1.computeCDF(Var1[i]))
        inverse_cond=copula.computeConditionalQuantile(uniformSample[i],condition)    
        condRealization = marginal2.computeQuantile(inverse_cond)
        Var2.append(condRealization[0])
    sample_array=Var2 
    return sample_array

def ot_sample(bivariate_distribution,N=1): 
    uniformSample1 = np.random.random(N)
    uniformSample2 = np.random.random(N)
    marginal1=bivariate_distribution.getMarginal(0)
    marginal2=bivariate_distribution.getMarginal(1)
    copula=bivariate_distribution.getCopula()
    Var1=[]
    Var2=[]
    for i in range(N):
        usample2=ot.Point(1,uniformSample2[i])
        inverse_cond=copula.computeConditionalQuantile(uniformSample1[i],usample2)    
        Realization1 = marginal1.computeQuantile(inverse_cond)
        Realization2 = marginal2.computeQuantile(uniformSample2[i])
        Var1.append(Realization1[0])
        Var2.append(Realization2[0])
    sample_array=np.hstack((np.array(Var1).reshape(-1,1),np.array(Var2).reshape(-1,1))) 
    return sample_array
