##Modules##
import sys, os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
import openturns as ot
import openturns.viewer as otv
import ot_copula_conditional_YE as cond

Global_resolution=1.5


def multipage(filename, figs=None, dpi=300):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

    
class Logger(object): # Lumberjack class - duplicates sys.stdout to a log file;
                      # source: https://stackoverflow.com/a/24583265/5820024
    def __init__(self, filename="Log", mode="a", buff=1):
        self.stdout = sys.stdout
        self.file = open(filename, mode, buff)
        sys.stdout = self
    def __del__(self):
        self.close()
    def __enter__(self):
        pass
    def __exit__(self, *args):
        pass
    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
    def flush(self):
        self.stdout.flush()
        self.file.flush()
        os.fsync(self.file.fileno()) # The written string is in program buffer but it might not actually be
                                     # writed to disk until file is closed, 'fsync' force real time writing
    def close(self):
        if self.stdout != None:
            sys.stdout = self.stdout
            self.stdout = None
        if self.file != None:
            self.file.close()
            self.file = None


#Figures_Validation_Funtion Test
def Two_axes_plot (x,y1,y2,title,xlabel,y1label='error', y2label='Time (s)'):
    fig, ax1 = plt.subplots(figsize=(6*Global_resolution,4*Global_resolution))
    ax1.semilogy(x, y1, 'ko-')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(y1label)
    ax2 = ax1.twinx()
    ax2.plot(x, y2, 'ro-')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylabel(y2label,color='red')
    plt.title(title)

  
#Descriptive statistics#
#Var=Variable
class Stats_Univariate(object):
    def __init__(self,Var):
        self.N=len(Var)
        self.min_value=np.amin(Var) 
        self.max_value=np.amax(Var)
        self.range_value=self.max_value-self.min_value
        self.mean=np.mean(Var)
        self.median=np.median(Var)
        self.Frist_quartile=np.percentile(Var,25)
        self.Third_quartile=np.percentile(Var,75)
        self.IR=self.Third_quartile-self.Frist_quartile
        self.variance_value=np.var(Var)
        self.std_value=np.std(Var)
        self.symmetry=stats.skew(Var)
        self.kurtosis=stats.kurtosis(Var)
        
    def Table(self):
        Stadistic=np.hstack((self.N,self.min_value, self.max_value,self.range_value,self.mean,self.median,self.Frist_quartile,self.Third_quartile,self.IR,self.variance_value,self.std_value,self.symmetry,self.kurtosis))#.reshape(1,-1)
        Stadistic_names = ["N","Min", "Max","Range","Mean","Median","Frist quartile","Third quartile","IR","Variance","Std","Symmetry","Kurtosis"]
        Table=pd.DataFrame(Stadistic, Stadistic_names)
        print(Table)

    def histogram_boxplot(self,Var,xlabel="Porosity",marginal=None,nbins=10):
        #Histogram and boxplot
        fig, (ax_box, ax_hist) = plt.subplots(2, gridspec_kw={"height_ratios": (.15, .85)},figsize=(6*Global_resolution,4*Global_resolution))
        
        #Boxplot#
        #To change the Boxplot
        outlires = dict(markeredgecolor='black',markerfacecolor='white', marker='o')#outlires
        caps=dict(color='black')
        whiskers=dict(linewidth=1,color='black')
        box = dict( facecolor='lightgray', color='black')
        medians = dict(linestyle='-', linewidth=1, color='blue')
        means = dict(marker='X',markeredgecolor='black',markerfacecolor='red')
        #Plot
        ax_box.boxplot(Var,vert=False, widths=[0.8],patch_artist=True,showfliers=True, showcaps=True, showbox=True,
                             showmeans=True,meanline=False,flierprops=outlires,capprops=caps, whiskerprops=whiskers,
                             boxprops=box, medianprops=medians,meanprops=means)
        #To change axes
        ax_box.set_yticks([])
        ax_box.set_xticks([])
        #Another way to change the boxplot
        #box['fliers'][1].set_color('blue')
        #box['caps'][0].set_color('blue')
        #box['boxes'][0].set_facecolor('gray')
        #box['whiskers'][0].set_color('blue')
        #box['medians'][0].set_color('yellow')
        #box['means'][0].set_marker('x')
        
        #Histogram#
        #Plot
        n,bins,_=ax_hist.hist(Var,bins=nbins, edgecolor='black', facecolor='lightgray')
        if marginal!=None:
           x=np.linspace(self.min_value,self.max_value,num=200)
           y=ot_compute_PDF(x,marginal)
           ax_hist.plot(x,y*(bins[-1]-bins[0])*self.N/nbins, color='red')
           #Legend
           legend = ['Density','Median', 'Mean']
           color=['red','blue', 'red']
        else:
           #Legend
           legend = ['Median', 'Mean']
           color=['blue', 'red']
        #To put count label
        count, bin_value = np.histogram(Var)
        ax_hist.set_ybound(upper=ax_hist.get_ybound()[1]+0.5)
        for p,i in zip(ax_hist.patches,count):
            posx=p.get_x()+p.get_width()/3
            posy=p.get_height()+0.22
            ax_hist.text(posx,posy,i)
        #Median and median lines   
        plt.axvline(self.median, ls='--', color='blue')
        plt.axvline(self.mean, ls='--', color='red')
        #Add
        plt.xlabel(xlabel)
        plt.ylabel("Frequency")
        plt.legend(legend)
        plt.title("")

        
#Stadistic bivariate
class Stats_Bivariate(object):
    def __init__(self,Var):
        Var_pd=pd.DataFrame(Var)#{'Ip': Var[:,0], 'Phit': Var[:,1]})
        self.covariance=np.array(Var_pd.cov())
        self.Pearson=np.array(Var_pd.corr(method='pearson'))
        self.Spearman=np.array(Var_pd.corr(method='spearman'))
        self.Kendall=np.array(Var_pd.corr(method='kendall'))
        
    def Table(self):
        Stadistic=np.transpose(np.hstack((self.covariance,self.Pearson, self.Spearman,self.Kendall)).reshape(2,-1))
        Stadistic_names = ["Covariance","", "Pearson","","Spearman","","Kendall",""]
        Table=pd.DataFrame(Stadistic, Stadistic_names)
        print(Table)

        
#Density marginal histogram        
def ot_compute_PDF(val,distribution):
    pdf=0*val
    for i in range(len(val)):
        pdfval=distribution.computePDF(val[i])
        pdf[i]=pdfval 
    return pdf


#Histogam marginal
def histogram_marginals(marginal, Var, ax=None, orientation='vertical'):
    #Distributions and histogram for marginals
    #Plot
    if ax==None:
        fig, ax = plt.subplots(1,1,figsize=(6*Global_resolution,4*Global_resolution))
    ax.hist(Var,bins=20, edgecolor='black', facecolor='lightgray', density=True, orientation=orientation)
    count, bin_value = np.histogram(Var)
    #Distributions
    x=np.linspace(bin_value[0],bin_value[-1],num=200)
    y=ot_compute_PDF(x,marginal)
    #To put count label
    if orientation=='horizontal':
        for p,i in zip(ax.patches,count):
            posy=p.get_y()#+p.get_width()/3
            posx=p.get_height()+0.22
        ax.plot(y,x,color='red')
    else:
        for p,i in zip(ax.patches,count):
            posx=p.get_x()+p.get_width()/3
            posy=p.get_height()+0.22
        ax.plot(x,y,color='red')

#To label over the level curve
def ot_drawcontour(ot_draw):
    ot.ResourceMap_SetAsUnsignedInteger("Contour-DefaultLevelsNumber",7)
    drawables = ot_draw.getDrawables()
    levels = []
    for i in range(len(drawables)):
        contours = drawables[i]
        levels.append(contours.getLevels()[0])
    ot.ResourceMap.SetAsUnsignedInteger('Drawable-DefaultPalettePhase', len(levels))#Colors
    palette = ot.Drawable.BuildDefaultPalette(len(levels))#Colors
    newcontour = ot_draw.getDrawable(0)
    drawables = list()
    for i in range(len(levels)):
        newcontour.setLevels([levels[i]]) # Inline the level values
        newcontour.setDrawLabels(True)
        newcontour.setLabels([str("{:.3e}".format(levels[i]))])
        # We have to copy the drawable because a Python list stores only pointers
        drawables.append(ot.Drawable(newcontour))
    graphFineTune = ot.Graph("", "", "", True, '')
    graphFineTune.setDrawables(drawables)
    graphFineTune.setColors(palette) # Add colors
    return graphFineTune

# Pseudovservations cloud in the rank space
def pseudo_obs_draw(marginal,Var1, Var2, ax=None, color='k'):
    if ax==None:
        fig, ax = plt.subplots(1,1,figsize=(6*Global_resolution,4*Global_resolution))
    #Plot copula
    U=ot_compute_CDF(Var1,marginal[0])
    V=ot_compute_CDF(Var2,marginal[1])
    ax.scatter(U,V,marker='.',color=color)

#Copula pdf        
def copula_draw(copula, ax=None):
    if ax==None:
        fig, ax = plt.subplots(1,1,figsize=(6*Global_resolution,4*Global_resolution))
    #Plot copula
    ot_draw=copula.drawPDF([0,0], [1,1], [201,201])
    otv.View(ot_draw, axes=[ax] ,add_legend=False)
    #Color scale 
    drawables = ot_draw.getDrawables()
    levels = []
    for i in range(len(drawables)):
        contours = drawables[i]
        levels.append("{:.2f}".format(contours.getLevels()[0]))
    ax.legend(levels,loc='center right',bbox_to_anchor=(-0.25/Global_resolution, 0.5),fontsize=7.5)


#Bivariate distribution PDF
def bivariate_distribution_draw(bivariate_distribution,ax=None):
    if ax==None:
        fig, ax = plt.subplots(1,1,figsize=(6*Global_resolution,4*Global_resolution))
    #Plot bivariate distributions
    ot_draw=bivariate_distribution.drawPDF()
    otv.View(ot_draw, axes=[ax] ,add_legend=False)
    #Color scale
    drawables = ot_draw.getDrawables()
    levels = []
    for i in range(len(drawables)):
        contours = drawables[i]
        levels.append("{:.2e}".format(contours.getLevels()[0]))
    ax.legend(levels,loc='center left',bbox_to_anchor=(1, 0.5),fontsize=7.5)


def four_axis(marginal,Var1, Var2,copula, bivariate_distribution):
    fig,((ax1,ax2),(ax3,ax4))=plt.subplots(2,2,figsize=(6*Global_resolution,4*Global_resolution))
    
    histogram_marginals(marginal[1], Var2, ax=ax1,orientation='horizontal') 
    histogram_marginals(marginal[0], Var1, ax=ax4) 
    copula_draw(copula, ax=ax3)
    pseudo_obs_draw(marginal,Var1, Var2, ax=ax3)
    bivariate_distribution_draw(bivariate_distribution,ax=ax2)
    ax2.scatter(Var1, Var2,marker='.',color='black')
    
    fig.suptitle('Bivarite analysis PDF', fontsize=16)
    
    #Equal scale
    ax2.set_xlim(ax4.get_xlim())
    ax2.set_ylim(ax1.get_ylim())
    ax2.locator_params(nbins=4) #nbins don't set the number, but the maximum number of bins
    ax1.locator_params(nbins=4) 
    #ax2.label_outer()
    ax2.set_title('Join PDF',fontsize=10)
    ax1.yaxis.grid(True)
    ax4.locator_params(nbins=4) 
    ax4.xaxis.grid(True)    
    ax3.grid(False)
    ax3.locator_params(nbins=5) 
    
    #Ticks for variable
    ax1.set_ylabel('Phit',fontsize=10)
    ax4.set_xlabel('Ip',fontsize=10)
    ax3.set_xlabel('Copula PDF',fontsize=10)
    ax3.set_ylabel('')
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    
    fig.tight_layout(rect=[0, 0, 1, 0.95]) 


#Distribution marginal histogram         
def ot_compute_CDF(val,distribution):
    cdf=0*val
    for i in range(len(val)):
        cdfval=distribution.computeCDF(val[i])
        cdf[i]=cdfval 
    return cdf


#Histogam marginal cumulative
def cumul_histogram_marginals(marginal, Var, ax=None, orientation='vertical'):
    #Distributions and histogram for marginals
    #Plot
    if ax==None:
        fig, ax = plt.subplots(1,1,figsize=(6*Global_resolution,4*Global_resolution))
    ax.hist(Var,bins=20, edgecolor='black', facecolor='lightgray', density=True, orientation=orientation, cumulative=True)
    count, bin_value = np.histogram(Var)
    #Distributions
    x=np.linspace(bin_value[0],bin_value[-1],num=200)
    y=ot_compute_CDF(x,marginal)
    #To put count label
    if orientation=='horizontal':
        for p,i in zip(ax.patches,count):
            posy=p.get_y()#+p.get_width()/3
            posx=p.get_height()+0.22
        ax.plot(y,x,color='red')
    else:
        for p,i in zip(ax.patches,count):
            posx=p.get_x()+p.get_width()/3
            posy=p.get_height()+0.22
        ax.plot(x,y,color='red')

      
#Copula cdf         
def cumul_copula_draw(copula, ax=None):
    if ax==None:
        fig, ax = plt.subplots(1,1,figsize=(6*Global_resolution,4*Global_resolution))
    #Plot copula
    ot_draw=copula.drawCDF([0,0], [1,1], [201,201])
    otv.View(ot_draw, axes=[ax] ,add_legend=False)
    #Color scale 
    drawables = ot_draw.getDrawables()
    levels = []
    for i in range(len(drawables)):
        contours = drawables[i]
        levels.append("{:.2f}".format(contours.getLevels()[0]))
    ax.legend(levels,loc='center right',bbox_to_anchor=(-0.25/Global_resolution, 0.5),fontsize=7.5)


#Bivariate distribution CDF
def cumul_bivariate_distribution_draw(bivariate_distribution,ax=None):
    if ax==None:
        fig, ax = plt.subplots(1,1,figsize=(6*Global_resolution,4*Global_resolution))
    #Plot bivariate distributions
    ot_draw=bivariate_distribution.drawCDF()
    otv.View(ot_draw, axes=[ax] ,add_legend=False)
    #Color scale 
    drawables = ot_draw.getDrawables()
    levels = []
    for i in range(len(drawables)):
        contours = drawables[i]
        levels.append("{:.2e}".format(contours.getLevels()[0]))
    ax.legend(levels,loc='center left',bbox_to_anchor=(1, 0.5),fontsize=7.5)
  
      
def cumul_four_axis(marginal,Var1, Var2,copula, bivariate_distribution):
    fig,((ax1,ax2),(ax3,ax4))=plt.subplots(2,2,figsize=(6*Global_resolution,4*Global_resolution))
    
    cumul_histogram_marginals(marginal[1], Var2, ax=ax1,orientation='horizontal') 
    cumul_histogram_marginals(marginal[0], Var1, ax=ax4) 
    cumul_copula_draw(copula, ax=ax3)
    pseudo_obs_draw(marginal,Var1, Var2, ax=ax3)
    cumul_bivariate_distribution_draw(bivariate_distribution,ax=ax2)
    ax2.scatter(Var1, Var2,marker='.',color='black')
    
    fig.suptitle('Bivariate analysis CDF', fontsize=16)
    
    #Equal scale
    ax2.set_xlim(ax4.get_xlim())
    ax2.set_ylim(ax1.get_ylim())
    ax2.locator_params(nbins=4) #nbins don't set the number, but the maximum number of bins
    #ax2.label_outer()
    ax2.set_title('Join CDF',fontsize=10)
    ax1.locator_params(axis='x',nbins=5) 
    ax1.locator_params(axis='y',nbins=4) 
    ax1.yaxis.grid(True)
    ax4.locator_params(axis='x',nbins=4) 
    ax4.locator_params(axis='y',nbins=5) 
    ax4.xaxis.grid(True)   
    ax3.grid(False)
    ax3.locator_params(nbins=5)
    
    #Ticks variable
    ax1.set_ylabel('Phit',fontsize=10)
    ax4.set_xlabel('Ip', fontsize=10)
    ax3.set_xlabel('Copula CDF', fontsize=10)
    ax3.set_ylabel('')
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    
    fig.tight_layout(rect=[0, 0, 1, 0.95]) 
    
   
def Teorical_variogram(lags,svt,sill, a_range, var_model='Spherical'):
    #Plot_teorical_vaiogram#  
    plt.figure(figsize=(6*Global_resolution,4*Global_resolution))
    plt.plot(lags, svt, 'r-', label=var_model)
    #Lines
    plt.axhline(sill, ls='--', color='blue', label='Sill')
    plt.axvline(a_range, ls='--', color='green', label='Range')
    #Labels
    plt.legend()
    plt.xlabel("Distance(m)")
    plt.ylabel("Semivariogram")

  
def Experimental_Variogram(lags, sv_list, sill, a_range, variance=0,var_model='Spherical',color_svt='red',sv_plot=1,lags_svt=[]):
    #Simulated experimental variogram#
    if lags_svt==[]:
        lags_svt=lags
    #Plot
    plt.figure(figsize=(5*Global_resolution,5*Global_resolution))
    plt.plot(lags_svt, sv_list[0], '-',color=color_svt, label=var_model)
    if len(sv_list)==3:
        plt.scatter(lags, sv_list[1], marker="o",color='black', label='Reference experimental')
        plt.scatter(lags, sv_list[2],marker="o",color='green', label='Simulated experimental')
        plt.ylim(bottom=0,top=0.0016)
    elif len(sv_list)==2:
        plt.scatter(lags, sv_list[1],marker="o",color='black', label='Experimental')
    #Lines
    plt.axhline(sill, ls='--', color='blue', label='Sill')
    plt.axvline(a_range, ls='--', color='green', label='Range')
    if variance>0:
        plt.axhline(variance, ls='--', color='red', label='Variance')
    #Labels
    plt.legend()
    plt.xlabel("Distance(m)")
    plt.ylabel("Semivariogram")

    
def Scater(x, y1, y2,options='scater',labelx='Phit',labely1='conditioned_pdf_s',labely2= 'conditioned_pdf', Condition=[],label_Condition='conditioning', color=['black','red']):
    plt.figure(figsize=(5*Global_resolution,5*Global_resolution))
    if options=='scater':
        #Plot   
        plt.scatter(x,y2,label=labely2, color=color[0],marker='.') #alpha:transparence 
        plt.scatter(x,y1,label=labely1,alpha=0.8, color=color[1],marker='.')
    elif options=='plot':
        #Plot
        order=np.argsort(x)
        plt.plot(x[order],y1[order],label=labely1)  
        plt.plot(x[order],y2[order],label=labely2,alpha=0.8) 
    if len(Condition)>0: 
        plt.scatter(Condition[:,0],Condition[:,1],label=label_Condition,marker='o',facecolors='none',edgecolors='blue') 
    #Labels
    plt.legend()
    plt.xlabel(labelx,fontsize=10)
    plt.ylabel(labely2,fontsize=10)

def pseudo_obs_scater(marginal1,marginal2,Var1,Var2,Var3,ax=None,labely1='u (Phit)',labely2='u (Phits)',labelx='v (Ip)', color=['k','r']):
    if ax==None:
        fig, ax = plt.subplots(1,1,figsize=(Global_resolution*5,Global_resolution*5))
    #Plot copula
    U=ot_compute_CDF(Var1,marginal1[0])
    V=ot_compute_CDF(Var2,marginal1[1])
    Vs=ot_compute_CDF(Var3,marginal2[1])
    ax.scatter(U,V,marker='.',label=labely1,color=color[0])
    ax.scatter(U,Vs,marker='.',label=labely2,color=color[1])
    #Labels
    ax.set_xlabel(labelx,fontsize=10)
    ax.set_ylabel(labely1,fontsize=10)
    plt.legend()

    
def Scater1(x, y,labelx='Phit',labely='Ip', color='orange',ax=None):
    if ax==None:
        fig, ax = plt.subplots(1,1,figsize=(Global_resolution*5,Global_resolution*5))
    #Plot
    ax.scatter(x,y,color=color,marker='.')  
    #Labels
    ax.set_xlabel(labelx,fontsize=10)
    ax.set_ylabel(labely,fontsize=10)
    
def pseudo_obs_scater1(marginal1,Var1,Var2,ax=None,labely1='u (Phit)',labelx='v (Ip)', color='k'):
    if ax==None:
        fig, ax = plt.subplots(1,1,figsize=(Global_resolution*5,Global_resolution*5))
    #Plot copula
    U=ot_compute_CDF(Var1,marginal1[0])
    V=ot_compute_CDF(Var2,marginal1[1])
    ax.scatter(U,V,marker='.',label=labely1,color=color)
    #Labels
    ax.set_xlabel(labelx,fontsize=10)
    ax.set_ylabel(labely1,fontsize=10)
    plt.legend()
    
def compute_conditional_cdf(val,bivariate_distribution,condition):
    conditioned_cdf=0.0*val
    for i in range(len(conditioned_cdf)):
        cdfval=bivariate_distribution.computeConditionalCDF(float(val[i]),[float(condition)])
        conditioned_cdf[i]=cdfval 
    return conditioned_cdf
    
def conditionalCDF(Pared,bivariate_distribution_dataT,bivariate_distribution_sT,conditionalVar, labely1='conditioned_cdf_s', labely2='conditioned_cdf'):
    plt.figure()
    Np=300
    Var=Pared[:,0]
    Cond=Pared[:,1]
    Xmin,Xmax=min(Var),max(Var)
    Var_space=np.linspace(Xmin,Xmax,Np)
    for condition in conditionalVar:
        indice = np.where(abs(Cond-condition)<100)
        encontrados=Var[indice]
        linea_s=compute_conditional_cdf(Var_space,bivariate_distribution_sT,condition)
        puntos_s=compute_conditional_cdf(encontrados,bivariate_distribution_sT,condition)
        plt.plot(Var_space,linea_s,color='blue')
        plt.scatter(encontrados,puntos_s,color='blue')
        linea_data=compute_conditional_cdf(Var_space,bivariate_distribution_dataT,condition)
        puntos_data=compute_conditional_cdf(encontrados,bivariate_distribution_dataT,condition)
        plt.plot(Var_space,linea_data,color='orange',alpha=0.8)
        plt.scatter(encontrados,puntos_data,color='orange',alpha=0.8)
    plt.legend([labely1,labely2])
    
def conditionalCDF1(Pared,bivariate_distribution_dataT,conditionalVar, labely='conditioned_cdf'):
    plt.figure()
    Np=300
    Var=Pared[:,1]
    Cond=Pared[:,0]
    Xmin,Xmax=min(Var),max(Var)
    Var_space=np.linspace(Xmin,Xmax,Np)
    for condition in conditionalVar:
        indice = np.where(abs(Cond-condition)<100)
        encontrados=Var[indice]
        linea_data=compute_conditional_cdf(Var_space,bivariate_distribution_dataT,condition)
        puntos_data=compute_conditional_cdf(encontrados,bivariate_distribution_dataT,condition)
        plt.plot(Var_space,linea_data,color='orange',alpha=0.8)
        plt.scatter(encontrados,puntos_data,color='orange',alpha=0.8)
    plt.legend([labely])

def Three_axes_conditionalCDF1(Pared_data,bivariate_distribution_dataT,condition, labelx1='Ip',labelx2='Phit',labely1='Phit',labely2='conditioned_cdf'):
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10*Global_resolution,5*Global_resolution))
    Np=300
    Var=Pared_data[:,1]
    Cond=Pared_data[:,0]
    tol=100
    
    Xmin,Xmax=min(Var),max(Var)
    Var_space=np.linspace(Xmin,Xmax,Np)
    indice = np.where(abs(Cond-condition)<100)
    encontrados=Var[indice]
    
    linea_data=compute_conditional_cdf(Var_space,bivariate_distribution_dataT,condition)
    puntos_data=compute_conditional_cdf(encontrados,bivariate_distribution_dataT,condition)
    count, bin_value = np.histogram(encontrados,bins=20)
    ax2.plot(Var_space,linea_data,color='red',alpha=0.8)
    ax2.scatter(encontrados,puntos_data,color='red',alpha=0.8)
    ax2.set_xlabel(labelx2,fontsize=10)
    ax2.set_ylabel(labely2,fontsize=10)
    
        
    Vmin,Vmax=min(Var),max(Var)
    ax1.scatter(Cond,Var,color='black')  
    ax1.fill([condition-tol,condition+tol,condition+tol,condition-tol],
             [0,0,Vmax,Vmax],'r',alpha=0.4)
    ax1.set_xlabel(labelx1,fontsize=10)
    ax1.set_ylabel(labely1,fontsize=10)
    
    
def Three_axes_conditionalCDF(Pared_data,Pared_Var,bivariate_distribution_dataT,bivariate_distribution_sT,condition, labely1='conditioned_cdf_s', labely2='conditioned_cdf'):
    fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(12*Global_resolution,4*Global_resolution))
    Np=300
    Var=Pared_Var[:,0]
    Vard=Pared_data[:,0]
    Cond=Pared_Var[:,1]
    tol=100
    
    Xmin,Xmax=min(Var),max(Var)
    Var_space=np.linspace(Xmin,Xmax,Np)
    indice = np.where(abs(Cond-condition)<100)
    encontrados=Var[indice]
    
    linea_s=compute_conditional_cdf(Var_space,bivariate_distribution_sT,condition)
    puntos_s=compute_conditional_cdf(encontrados,bivariate_distribution_sT,condition)
    ax3.plot(Var_space,linea_s,color='blue')    
    ax3.scatter(encontrados,puntos_s,color='blue')
    
    linea_data=compute_conditional_cdf(Var_space,bivariate_distribution_dataT,condition)
    puntos_data=compute_conditional_cdf(encontrados,bivariate_distribution_dataT,condition)
    ax2.hist(encontrados,bins=20, edgecolor='black', facecolor='lightgray', density=True, cumulative=True)
    count, bin_value = np.histogram(encontrados,bins=20)
    ax2.plot(Var_space,linea_data,color='orange',alpha=0.8)
    ax2.scatter(encontrados,puntos_data,color='orange',alpha=0.8)
    
    encontrados_data=Vard[indice]
    
    linea_s=compute_conditional_cdf(Var_space,bivariate_distribution_sT,condition)
    puntos_s=compute_conditional_cdf(encontrados,bivariate_distribution_sT,condition)
    ax3.hist(encontrados_data,bins=bin_value, edgecolor='black', facecolor='lightgray', density=True, cumulative=True)
    ax3.plot(Var_space,linea_s,color='blue')    
    ax3.scatter(encontrados,puntos_s,color='blue')
    
    Vmin,Vmax=min(Vard),max(Vard)
    ax1.scatter(Cond,Var,color='blue')  
    ax1.scatter(Cond,Vard,color='orange',alpha=0.8) #alpha:transparence 
    ax1.fill([condition-tol,condition+tol,condition+tol,condition-tol],
             [0,0,Vmax,Vmax],'r',alpha=0.4)



def logview(Data,tracks,labels,title='Pozo',limits=[],colors=[],mean=[],median=[],Condition=[]):
    #tracks has the form [[1],[1,...,n],[2]] where the numbers are the columns
    #colors and limits should have the same form as tracks
    #each limit in limits is a tuple (a,b)
    
    if len(colors)==0: #if colors are not deffined set all to black
        colors = [['k' for element in t] for t in tracks]
    
    z=Data[:,0]
    order=np.argsort(z)
    ntrack=len(tracks)
    fig,axis=plt.subplots(1,ntrack,figsize=(ntrack*2*Global_resolution,5*Global_resolution))#Equal width tracks
    
    for n in range(ntrack):
        #Plot each carril
        ax=axis[n]
        ax.label_outer() #only axis z, for the frist track
        ax.set_xticklabels([]) #Remove an axis x extra
        #add the grid
        ax.grid(which='both') 
        ax.xaxis.set_major_locator(matplotlib.ticker.LinearLocator(4))
        # ax.xaxis.set_minor_locator(matplotlib.ticker.LinearLocator(5))
        ax.invert_yaxis()
        
        nplots=len(tracks[n])
        spin_position=10
        for i in range(nplots):
            # ax.xaxis.set_visible(False)
            newax=ax.twiny() #Duplicate axis x.
            newax.plot(Data[order,tracks[n][i]],z[order],label=labels[n][i],color=colors[n][i]) #Plot
            if len(Condition)>0:
                z_cond=Condition[:,0]
                newax.scatter(Condition[:,tracks[n][i]],z_cond,marker='o',facecolors='none',edgecolors='blue')
            newax.xaxis.tick_top() #axis in botton position
            #Scale top
            newax.spines['top'].set_position(('outward', spin_position))
            spin_position=spin_position+35
            newax.spines['top'].set_color(colors[n][i])
            newax.set_xlabel(labels[n][i],color=colors[n][i])
            if len(limits)>0:
                newax.set_xlim(limits[n][i][0],limits[n][i][1])
                
            ## The next three lines align all the ticks to the same grid ##
            #https://stackoverflow.com/questions/45037386/trouble-aligning-ticks-for-matplotlib-twinx-axes
            l, l2 = ax.get_xlim(), newax.get_xlim()
            f = lambda x : l2[0]+(x-l[0])/(l[1]-l[0])*(l2[1]-l2[0]); ticks = f(ax.get_xticks())
            newax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
            #Median and median lines   
            if len(median)>0 and median[n][i]!='':
                m1=[median[n][i] for z_index in z]
                c='blue'
                renewax=ax.twiny()
                renewax.plot(m1,z[order], ls='--', color=c)
                #Scale top
                renewax.spines['top'].set_position(('outward', spin_position))
                spin_position=spin_position+20
                renewax.spines['top'].set_color(c)
                renewax.set_xlabel('median',color=c)
                renewax.set_xticks([])
                renewax.set_xlim(l2)
            if len(mean)>0 and mean[n][i]!='':
                m0=[mean[n][i] for z_index in z]
                c='red'
                renewax=ax.twiny()
                renewax.plot(m0,z[order], ls='--', color=c)
                #Scale top
                renewax.spines['top'].set_position(('outward', spin_position))
                spin_position=spin_position+20
                renewax.spines['top'].set_color(c)
                renewax.set_xlabel('mean',color=c)
                renewax.set_xticks([])
                renewax.set_xlim(l2)
    
    fig.suptitle(title,  fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95]) 
    
def logview1(Data,track,labels,title='Pozo',limits=[],colors=[],mean=[],median=[],Condition=[]):
    #tracks has the form [[1],[1,...,n],[2]] where the numbers are the columns
    #colors and limits should have the same form as tracks
    #each limit in limits is a tuple (a,b)
    
    if len(colors)==0: #if colors are not deffined set all to black
        colors = ['k' for element in track]
    
    z=Data[:,0]
    order=np.argsort(z)
    fig,ax=plt.subplots(1,1,figsize=(2*Global_resolution,5*Global_resolution))#Equal width tracks
    
    #Plot each carril
    ax.label_outer() #only axis z, for the frist track
    ax.set_xticklabels([]) #Remove an axis x extra
    #add the grid
    ax.grid(which='both') 
    ax.invert_yaxis()
    
    nplots=len(track)
    spin_position=10
    for i in range(nplots):
        newax=ax.twiny() #Duplicate axis x.
        newax.plot(Data[order,track[i]],z[order],label=labels[i],color=colors[i]) #Plot
        if len(Condition)>0:
            z_cond=Condition[:,0]
            newax.scatter(Condition[:,track[i]],z_cond,marker='o',facecolors='none',edgecolors='blue')
        newax.xaxis.tick_top() #axis in botton position
        #Scale top
        newax.spines['top'].set_position(('outward', spin_position))
        spin_position=spin_position+35
        newax.spines['top'].set_color(colors[i])
        newax.set_xlabel(labels[i],color=colors[i])
        if len(limits)>0:
            newax.set_xlim(limits[i][0],limits[i][1])
            
        ## The next three lines align all the ticks to the same grid ##
        #https://stackoverflow.com/questions/45037386/trouble-aligning-ticks-for-matplotlib-twinx-axes
        l, l2 = ax.get_xlim(), newax.get_xlim()
        f = lambda x : l2[0]+(x-l[0])/(l[1]-l[0])*(l2[1]-l2[0]); ticks = f(ax.get_xticks())
        newax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        #Median and median lines   
        if len(median)>0 and median[i]!='':
            m1=[median[i] for z_index in z]
            c='blue'
            renewax=ax.twiny()
            renewax.plot(m1,z[order], ls='--', color=c)
            #Scale top
            renewax.spines['top'].set_position(('outward', spin_position))
            spin_position=spin_position+20
            renewax.spines['top'].set_color(c)
            renewax.set_xlabel('median',color=c)
            renewax.set_xticks([])
            renewax.set_xlim(l2)
        if len(mean)>0 and mean[i]!='':
            m0=[mean[i] for z_index in z]
            c='red'
            renewax=ax.twiny()
            renewax.plot(m0,z[order], ls='-', color=c)
            #Scale top
            renewax.spines['top'].set_position(('outward', spin_position))
            spin_position=spin_position+20
            renewax.spines['top'].set_color(c)
            renewax.set_xlabel('mean',color=c)
            renewax.set_xticks([])
            renewax.set_xlim(l2)
    
    fig.suptitle(title,  fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95]) 
    
def emprical_CDF(Vars,marginals,ax=None,colors=[],labelx='Phit',labely1=['empirical Phits'],labely2=['marginal Phit']):
    if ax==None:
        fig, ax = plt.subplots(1,1,figsize=(Global_resolution*5,Global_resolution*5))
        
    if len(colors)==0: #if colors are not deffined set all to black
        colors = ['k' for v in Vars]
    for i in range(len(Vars)): 
        #Calculate empirical
        order=np.argsort(Vars[i]) #Orden
        n=len(Vars[i])
        x=Vars[i][order] #Variable ordenada
        y=(np.array(range(n))+1)/n #Conteo de x
        ax.scatter(x,y,color=colors[i],label=labely1[i])
        
        #Draw mariganl
        mx=np.linspace(x[0],x[-1],num=200)#Muestrea 200 puntos de x
        my=ot_compute_CDF(mx,marginals[i])#estima la distribucion en esos puntos.
        ax.plot(mx,my,color=colors[i+1],label=labely2[i])
    ax.legend()
    plt.xlabel(labelx,fontsize=10)
    plt.ylabel(labely2[0],fontsize=10)
