import numpy as np
import os
from time import time
import colossus.cosmology.cosmology as cosmo
from colossus.halo import concentration, mass_so
from colossus.lss.mass_function import massFunction
from scipy.interpolate import interp1d

cosmol=cosmo.setCosmology('planck15')    #cosmology for colossus. By default hmf uses planck15

def fred(x,bpar=1,beta=0.68):   # need to update this with new fit
    
    return 1./(1+beta*1.e12/10**x)




#ToDo put size things into a class
def sizes_from_lambda(R):

    ## Bullock
    lambda0 = -1.459
    sigma = 0.268
    fact = 10**(np.random.normal(lambda0,sigma,size=len(R)))
    return R*fact


def Schechter_like(x,width):

    ###peebles spin
    alpha =-4.126
    beta=0.610
    lambda0 = -2.919

    first = (x/10**lambda0)**(-alpha)
    second = -(x/10**lambda0)**(beta)
    final = first*np.exp(second)

    area = np.sum(final*width)
    return final/area


def Schechter_cumul(x,width):

    y = Schechter_like(x,width)
    cumul = np.cumsum(y)

    return cumul/np.max(cumul)

def extract_lambdas(R):

    x=np.linspace(1.e-4,0.5,1000000)
    width =x[1]-x[0]
    cumul = Schechter_cumul(x,width)
    ff = interp1d(cumul,x)
    xx = np.random.uniform(0,1,size= len(R))
    lambdas = ff(xx)
    return lambdas
    
def sizes_from_lambda_skewed(R):

    lambdas = extract_lambdas(R)
    return np.log10(R*lambdas)


def get_Rh(halos, redshift,mdef='vir'):
    halonew = np.array(10**halos)*cosmol.h #converts from Mvir to Mvir/h
    rhalo = mass_so.M_to_R(halonew,redshift,mdef=mdef)/cosmol.h  
    return rhalo
    
def get_sizefunction(halos, redshift, V,A_K, sigma_K, stars, masslow,massup,model='K13',sigmaK_evo=False, mdef='vir', bins=np.arange(-2,3,0.05)):
    rhalo = get_Rh(halos, redshift, mdef=mdef)

    if model=='K13':
        inp = rhalo*A_K
        if sigmaK_evo:
            sigma_K= 0.15+0.05*redshift
        Re = Kravtsov(inp, A=A_K, scatt=sigma_K)
        #print(Re)
    if model=='MMW':
        Re = sizes_from_lambda_skewed(rhalo)
        
    mask = np.ma.masked_inside(stars, masslow, massup).mask
    Re = Re[mask]
    occupation_distr = halos[mask]
    binwidth = bins[1]-bins[0]
  #  bins = np.arange(-2,3,binwidth)
    try:
        hist = np.histogram(Re, bins=bins)[0]
        return occupation_distr, Re,np.array([bins[1:]-0.5*binwidth, hist/V/(binwidth)])
    except:
        m = (masslow+massup)/2
        print('Could not compute size function for mass='+str(m)+'and z='+str(redshift))
        return 0,0,np.zeros(len(bins)-1)
    
    
def get_SigmaGargiulofunction(halos, redshift, V,A_K, sigma_K, stars, masslow,massup,model='K13',sigmaK_evo=False, mdef='vir', bins=np.arange(2,5,0.05)):
    rhalo = get_Rh(halos, redshift, mdef=mdef)

    if model=='K13':
        inp = rhalo*A_K
        if sigmaK_evo:
            sigma_K= 0.15+0.05*redshift
        Re = Kravtsov(inp, A=A_K, scatt=sigma_K)
        #print(Re)
    if model=='MMW':
        Re = sizes_from_lambda_skewed(rhalo)
        
    mask = np.ma.masked_inside(stars, masslow, massup).mask
    Re = Re[mask]
    stars = stars[mask]
    occupation_distr = halos[mask]
    binwidth = bins[1]-bins[0]
    print(bins[1:]-0.5*binwidth)
    Sigma = stars-2*Re-np.log10(2*np.pi) - 6 # -6 to convert from pc^-2 to kpc^-2
  #  bins = np.arange(-2,3,binwidth)
    try:
        hist = np.histogram(Sigma, bins=bins)[0] #Gargiulo definition of compacntess, Mstar/(2pi*R^2)>2000 msun/pc^2
        return occupation_distr,Sigma,np.array([bins[1:]-0.5*binwidth, hist/V/(binwidth)])
    except:
        m = (masslow+massup)/2
        print('Could not compute size function for mass='+str(m)+'and z='+str(redshift))
        return 0,0,np.zeros(len(bins)-1)
    
def get_gammafunction(halos, redshift, V,A_K, sigma_K, stars, masslow,massup,model='K13',sigmaK_evo=False, mdef='vir', bins=np.arange(-2,3,0.05)):
    rhalo = get_Rh(halos, redshift, mdef=mdef)

    if model=='K13':
        inp = rhalo*A_K
        if sigmaK_evo:
            sigma_K= 0.15+0.05*redshift
        Re = Kravtsov(inp, A=A_K, scatt=sigma_K)
        #print(Re)
    if model=='MMW':
        Re = sizes_from_lambda_skewed(rhalo)
        
    mask = np.ma.masked_inside(stars, masslow, massup).mask
    gamma = Re[mask]-0.75*(stars[mask]-11)
    
    occupation_distr = halos[mask]
    binwidth = bins[1]-bins[0]
  #  bins = np.arange(-2,3,binwidth)
    try:
        hist = np.histogram(gamma, bins=bins)[0]
        return occupation_distr, gamma,np.array([bins[1:]-0.5*binwidth, hist/V/(binwidth)])
    except:
        m = (masslow+massup)/2
        print('Could not compute size function for mass='+str(m)+'and z='+str(redshift))
        return 0,0,np.zeros(len(bins)-1)

def get_mean_size(halos,redshift,A_K,sigma_K,stars,masslow,massup, sigmaK_evo=False):
    rhalo = get_Rh(halos, redshift)
    inp = rhalo*A_K
    if sigmaK_evo:
        sigma_K= 0.15+0.05*redshift
    Re_ = Kravtsov(inp, A=A_K, scatt=sigma_K)
    mask = np.ma.masked_inside(stars, masslow, massup).mask
    
    try:
        return 10**np.percentile(Re_[mask],50)  #mean size evolution
    except:
        m = (masslow+massup)/2
        print('Could not compute size evolution for mass='+str(m)+'and z='+str(redshift))
        return np.nan
    
def Kravtsov(RA,A,scatt=0.1, redshift=None):
    RA = np.log10(RA)
    return np.random.normal(RA, scale=scatt)
        
def extract_catalog(N,M):
    f=interp1d(N,M)
    array_cumul=np.arange(min(N),max(N))
    cat=f(array_cumul)
    
    return cat


def get_halos(z, Vol,dlog10m=0.005,hmf_choice='despali16',mdef='vir' ):
    ####### set cosmology #########
    cosmol=cosmo.setCosmology('planck15')  
    cosmo.setCurrent(cosmol)
        ##############################
        
    Mvir=10**np.arange(10.,16,dlog10m) #Mh/h
    if hmf_choice=='despali16':
        massfunct =  massFunction(x=Mvir, z=z, mdef=mdef, model='despali16', q_out='dndlnM')*np.log(10)  #dn/dlog10M
       
            
    elif hmf_choice=='rodriguezpuebla16':
        massfunct = hmf_rp16(Mvir,z)*Mvir/np.log10(np.exp(1))

    massfunct = massfunct*(cosmol.h)**3 #convert  from massf*h**3
    total_haloMF=massfunct.copy()
    #massfunct =  massFunction(x=Mvir, z=z, q_in='M',mdef='vir', model='despali16', q_out='dndlnM')*np.log(10)  #dn/dlog10M
        
    Mvir=np.log10(Mvir)
    Mvir=Mvir-np.log10(cosmol.h)  #convert from M/h

        
    Ncum=Vol*(np.cumsum((total_haloMF*dlog10m)[::-1])[::-1])
    halos=extract_catalog(Ncum,Mvir)

    return halos


class get_SMHM:
    
    def __init__(self):
        np.random.seed(int(time()+os.getpid()*1000))

    class moster13:
        
        def __init__(self, scatteron=True, scatterevol=False):
            self.scatteron = scatteron
            self.scatterevol = scatterevol
        def make(self, halos, z):
            zparameter = np.divide(z, z+1)
            M10, SHMnorm10, beta10, gamma10, Scatter = 11.590, 0.0351, 1.376, 0.608, 0.15
            M11, SHMnorm11, beta11, gamma11 = 1.195, -0.0247, -0.826, 0.329
        
            M = M10 + M11*zparameter
            N = SHMnorm10 + SHMnorm11*zparameter
            b = beta10 + beta11*zparameter
            g = gamma10 + gamma11*zparameter
            stars =  np.power(10, halos) * (2*N*np.power( (np.power(np.power(10,halos-M), -b) + np.power(np.power(10,halos-M), g)), -1))
    
           
            if self.scatteron:
                if self.scatterevol:
                    if z<0.5:
                        scatt = 0.1
                        print('here')
                    else:
                        scatt =0.3
                else:
                    scatt=0.3
                stars = np.random.normal(np.log10(stars),scale=scatt)
                return stars
            return np.log10(stars)
        
        def __call__(self,halos,z):
            return self.make(halos,z)
        
         
    class grylls19:  
        def __init__(self, scatteron=True, SE=False, PyMorph=False, cmodel=False):
            self.scatteron = scatteron
            self.SE = SE
            self.PyMorph = PyMorph
            self.cmodel = cmodel
        def make(self, halos,z, constant):
            zparameter = np.divide(z-0.1, z+1)   
            if self.SE:
                M10, SHMnorm10, beta10, gamma10, Scatter = 12.0,0.032,1.5,0.56,0.15 
                M11, SHMnorm11, beta11, gamma11 = 0.6,-0.014,-0.7,0.08
            if self.PyMorph:
                print('pymorph')
                M10, SHMnorm10, beta10, gamma10, Scatter = 11.92,0.032,1.64,0.53,0.15
                M11, SHMnorm11, beta11, gamma11 = 0.58,-0.014,-0.69,0.03
            if self.cmodel:
                print('cmodel')
                M10, SHMnorm10, beta10, gamma10, Scatter =11.91,0.029,2.09,0.64,0.15
                M11, SHMnorm11, beta11, gamma11 = 0.52,-0.018,-1.03,0.084
        
            if constant:
                zparameter = 0.
                
            M = M10 + M11*zparameter
            N = SHMnorm10 + SHMnorm11*zparameter
            b = beta10 + beta11*zparameter
            g = gamma10 + gamma11*zparameter
            stars =  np.power(10, halos) * (2*N*np.power( (np.power(np.power(10,halos-M), -b) + np.power(np.power(10,halos-M), g)), -1))
        
            if self.scatteron:
                scatt=0.15
                stars = np.random.normal(np.log10(stars),scale=scatt)
                return stars
            return np.log10(stars)
        
        def __call__(self, halos, z,constant=False):
            return self.make(halos,z, constant)
        
        
    class rodriguezpuebla17:
        
        def __init__(self, scatteron=True, scatterevol =True):
            self.scatteron = scatteron
            self.scatterevol = scatterevol
        def P(self,x,y,z):
            return y*z-x*z/(1+z)

        def a(self,z):
            return 1./(1.+z)

        def nu(self,z):
            nu=np.e**(-4*self.a(z)**2)  
            return nu

        def m1(self,z):
            m1=10**(11.548+self.P(-1.297,-0.026,z)*self.nu(z))
            return m1

        def eps(self,z):
            eps=10**(-1.758+self.P(0.11,-0.061,z)*self.nu(z)+self.P(-0.023,0,z))
            return eps

        def alfa(self,z):
            alfa=1.975+self.P(0.714,0.042,z)*self.nu(z)
            return alfa

        def delta(self,z):
            delta=3.390+self.P(-0.472,-0.931,z)*self.nu(z)
            return delta

        def gamma(self,z):
            gamma=0.498+self.P(-0.157,0,z)*self.nu(z)
            return gamma

        def scattRP17(self,z):
            var=0.1+0.05*z
            scatt=np.sqrt(0.15**2+np.power(var,2))
            return scatt

        def make(self, halos, z, constant):
            if constant:
                z = 0.1
            
            x=halos-np.log10(self.m1(z))
            first=-np.log10(10**(-self.alfa(z)*x)+1)
            second= self.delta(z)*(np.log10(1+np.e**x))**self.gamma(z)
            third=1+np.e**(10**(-x))
            f=first+second/third

            first=-np.log10(10**(-self.alfa(z)*0.)+1)
            second= self.delta(z)*(np.log10(1+np.e**-0.))**self.gamma(z)
            third=1+np.e**(10**(0.))
            f0=first+second/third
    
            stars=np.log10(self.eps(z)*self.m1(z))+ f-f0
        
            
            if self.scatteron:
                if self.scatterevol:
                    scatt = self.scattRP17(z)
                else:
                    scatt = 0.15
                stars = np.random.normal(stars,scale=scatt)
                return stars
            return stars  
        
        def __call__(self, halos,z, constant=False):
            return self.make(halos,z,constant)
        
        
    class Z19_LTGs_ETGs:  #SHMR functional form; Behroozi+2010
        def __init__(self, scatteron=True, choice='All'):
            self.scatteron = scatteron
            self.choice = choice
 
        def g(self, x, a, g, d):
            return (-np.log10(10**(-a*x)+1.) +
                d*(np.log10(1.+np.exp(x)))**g/(1.+np.exp(10**(-x))))
     
        def scatt(self, log10Mvir):
            if self.choice=='All':
                return 0.15 #needs change
            elif self.choice=='LTGs':
                return 0.12
            elif self.choice=='ETGs':
                return 0.14
        
        def make(self,log10Mvir,z=None):
            if self.choice=='LTGs':
                alpha, delta, gamma, log10eps, log10M1 = 1.47439,4.26527,0.314474,-1.70524,11.4935
            elif self.choice=='ETGs':
                alpha, delta, gamma, log10eps, log10M1 = 6.10862,5.2244,0.336961,-2.19148,11.7417
            elif self.choice=='All':
                alpha, delta, gamma, log10eps, log10M1 = 1.72338,4.43454,0.422541,-1.88375,11.5095
            elif self.choice=='Z19':
                alpha, delta, gamma, log10eps, log10M1 = 2.352,3.797,0.6,-1.785,11.632
            x = log10Mvir - log10M1
            g1 = self.g(x, alpha, gamma, delta)
            g0 = self.g(0, alpha, gamma, delta)
            log10Ms = log10eps + log10M1 + g1 - g0
     
            if self.scatteron:
                log10Ms = np.random.normal(log10Ms, self.scatt(log10Mvir))
            return log10Ms
    
        def __call__(self, halos,z=None):
            return self.make(halos,z=None) #z is dummy here for consistency with the call to other SMHM
 
    class constantSMF:
        
        def __init__(self, scatteron=True):
            self.scatteron = scatteron
        def a(self,z):
            return 1./(1.+z)

        def nu(self,z):
            nu=np.e**(-4*self.a(z)**2)
            return nu

        def m1(self,z,M10=11.7845,M1a=1.074,M1z=-1.0596):
            m1=10**(M10+(M1a*(self.a(z)-1)+M1z*z)*self.nu(z))
            return m1

        def eps(self,z,eps0=-1.8456,epsa=-0.9274,epsz=0.1595,epsa2=0.7849):
            eps=10**(eps0+(epsa*(self.a(z)-1)+epsz*z)*self.nu(z)+epsa2*(self.a(z)-1))
            return eps

        def alfa(self,z,alfa0=-2.0105,alfaa=-0.0158):
            alfa=alfa0+(alfaa*(self.a(z)-1))*self.nu(z)
            return alfa

        def delta(self,z,delta0=3.6179,deltaa=-1.3933,deltaz=-2.2852):
            delta=delta0+(deltaa*(self.a(z)-1)+deltaz*z)*self.nu(z)
            return delta

        def gamma(self,z,gamma0=0.4822,gammaa=1.0908,gammaz=0.1906):
            gamma=gamma0+(gammaa*(self.a(z)-1)+gammaz*z)*self.nu(z)
            return gamma
        
        def scatter(self,z):
            var=0.1+0.05*z
            scatt=np.sqrt(0.15**2+np.power(var,2))
            return scatt       


        def make(self,halos,z):
            x=halos-np.log10(self.m1(z))
            first=-np.log10(10**(self.alfa(z)*x)+1)
            second= self.delta(z)*(np.log10(1+np.e**x))**self.gamma(z)
            third=1+np.e**(10**(-x))
            f=first+second/third
            first=-np.log10(10**(self.alfa(z)*0.)+1)
            second= self.delta(z)*(np.log10(1+np.e**-0.))**self.gamma(z)
            third=1+np.e**(10**(0.))
            f0=first+second/third
    
            stars=np.log10(self.eps(z)*self.m1(z))+ f-f0
        
            if self.scatteron:
                scatt = self.scatter(z)
                stars = np.random.normal(stars,scale=scatt)
                return stars
            return stars  
    
        def __call__(self,halos,z):
            return self.make(halos,z)
        

 
        
        
        
        

def DarkMatterToStellarMass(DM, z, Paramaters, ScatterOn = False, Scatter = 0.001, Pairwise = True):
    """ 
    This funtion returns Stellar mass in log10 Msun, all arguments should be passed in simmilar cosmology (Planck 15 unless otherwise stated)
    DM and z is longer than 1 are assumed pairwise if N == M, otherwise Array N is calculated for all elements (M) of z. If N==M but pairwise is not desired pass Pairwise == False
    Args:
        DM: Dark Matter in log10 Msun. Can be (1,), (N,), or (N, M)).
        z: Redshift. Can be (1,) or (M,) 
        Parameters: Python Dictonary Containing Subdictonary 'AbnMtch':
                        Containing Booleans: 'z_Evo', 'Moster', 'Override_0', 'Override_z', 'G18'
                        Containing Parameters: 'Scatter'
                        Containing Dictonary: 'OverRide':
                            Containing Parameters: 'M10', 'SHMnorm10', 'beta10', 'gamma10', 'M11', 'SHMnorm11', 'beta11', 'gamma11' 
        ScatterOn: Bool to switch scatter on/off
        Scatter: Scatter set low should be set in parameter section or sent via Scatter in dictonary.
        Pairwise: Bool. If true and N==M N and M will not be calculated Pairwise
    Returns:
        Stellar mass array in log10 Msun. Shape will be (1,), (N,), or (N, M) depending on the sape of inputs.
    Raises: 
        N/A
    """
    np.random.seed(int(str(time()).split('.')[1])+os.getpid())
    Paramaters = Paramaters['AbnMtch']
    if Paramaters['z_Evo']:
        if Paramaters['Moster']:
            zparameter = np.divide(z, z+1)
        elif Paramaters['Override_0'] or Paramaters['Override_z'] or Paramaters['G18'] or Paramaters['G18_notSE']:
            zparameter = np.divide(z-0.1, z+1)
        else:
            zparameter = np.divide(z-0.1, z+1)
    else:
        zparameter = 0

    if ScatterOn:
        Scatter = Paramaters['Scatter']
    
    if Paramaters['Override_0'] or Paramaters['Override_z']:
        Override = Paramaters['Override']

    # Go to RP17 abundance matching
    if Paramaters['RP17']:
        return SHMR_RP17(z, DM)
    # parameters from moster 2013
    if(Paramaters['Moster']):
        M10, SHMnorm10, beta10, gamma10, Scatter = 11.590, 0.0351, 1.376, 0.608, 0.15
        M11, SHMnorm11, beta11, gamma11 = 1.195, -0.0247, -0.826, 0.329
    if(Paramaters['Moster10']):
        M10, SHMnorm10, beta10, gamma10, Scatter = 11.884, 0.28320, 1.057, 0.556, 0.15
        M11, SHMnorm11, beta11, gamma11 = 1.195, -0.0247, -0.826, 0.329
    #paremeters from centrals Paper1
    if(Paramaters['G18']):
        M10, SHMnorm10, beta10, gamma10, Scatter = 11.95, 0.032, 1.61, 0.54, 0.11
        M11, SHMnorm11, beta11, gamma11 = 0.4, -0.02, -0.6, -0.1
    #paremeters from centrals Paper1 with a slight (-0.15) correction away from sersicexp
    if(Paramaters['G18_notSE']):
        M10, SHMnorm10, beta10, gamma10, Scatter = 11.95, 0.032, 1.61, 0.62, 0.11 #12.00, 0.022, 1.56, 0.55, 0.15
        M11, SHMnorm11, beta11, gamma11 = 0.4, -0.02, -0.6, 0.0 #0.4, 0.0, -0.5, 0.1
    if(Paramaters['G19_SE']):
        M10, SHMnorm10, beta10, gamma10, Scatter = 11.925, 0.032,1.639,0.532,0.15 #12.00, 0.022, 1.56, 0.55, 0.15
        M11, SHMnorm11, beta11, gamma11 = 0.576,-0.014,-0.693,0.03 #0.4, 0.0, -0.5, 0.1
    if(Paramaters['G19_cMod']):
        M10, SHMnorm10, beta10, gamma10, Scatter = 11.91,0.029,2.09,0.64,0.15 #12.0,0.032,1.74,0.66,0.15 #12.00, 0.022, 1.56, 0.55, 0.15
        M11, SHMnorm11, beta11, gamma11 = 0.644, -0.019, -1.422,  -0.043  #0.518,-0.018,-1.031,-0.084
    #parameters to recreate the illistrius M*Mh
    if(Paramaters['Illustris']):
        M10, SHMnorm10, beta10, gamma10, Scatter = 11.8,0.018,1.5,0.31,0.15 
        M11, SHMnorm11, beta11, gamma11 = 0.0,-0.01,0,-0.12
    #allows user to sent in their own abundance matching parameters either fixed at redshift 0/0.1 or evolving
    
    if(Paramaters['Override_0']):
        M10, SHMnorm10, beta10, gamma10 = Override['M10'], Override['SHMnorm10'], Override['beta10'], Override['gamma10']
        M11, SHMnorm11, beta11, gamma11 = 0.4, -0.02, -0.6, -0.1 #1.195, -0.0247, -0.826, 0.329
    if(Paramaters['Override_z']):
        M10, SHMnorm10, beta10, gamma10 = Override['M10'], Override['SHMnorm10'], Override['beta10'], Override['gamma10']
        M11, SHMnorm11, beta11, gamma11 = Override['M11'], Override['SHMnorm11'], Override['beta11'], Override['gamma11']
    #For Pairfraction Testing
    if Paramaters['PFT']:
        M10, SHMnorm10, beta10, gamma10, Scatter = 11.925, 0.032,1.639,0.532,0.15 #12.00, 0.022, 1.56, 0.55, 0.15
        M11, SHMnorm11, beta11, gamma11 = 0.576,-0.014,-0.693,0.03 #0.4, 0.0, -0.5, 0.1
        if(Paramaters['M_PFT1']):
            M10 = M10-0.25
        if(Paramaters['M_PFT2']):
            M11 = M11 + 0.1
        if(Paramaters['M_PFT3']):
            M11 = M11 - 0.1
        if(Paramaters['N_PFT1']):
            SHMnorm10 = SHMnorm10 + 0.004
        if(Paramaters['N_PFT2']):
            SHMnorm11 = SHMnorm11 + 0.007
        if(Paramaters['N_PFT3']):
            SHMnorm11 = SHMnorm11 - 0.007
        if(Paramaters['b_PFT1']):
            beta10 = beta10 - 0.3
        if(Paramaters['b_PFT2']):
            beta11 = beta11 + 0.3
        if(Paramaters['b_PFT3']):
            beta11 = beta11 - 0.3
        if(Paramaters['g_PFT1']):
            gamma10 = gamma10 + 0.06
        if(Paramaters['g_PFT2']):
            gamma11 = gamma11 + 0.2
        if(Paramaters['g_PFT3']):
            gamma11 = gamma11 - 0.2
        if(Paramaters['g_PFT4']):
            gamma10 = gamma10 - 0.1
    if Paramaters['HMevo']:
        M10, SHMnorm10, beta10, gamma10, Scatter = 11.91,0.029,2.09,0.64,0.15
        M11, SHMnorm11, beta11 = 0.518,-0.018,-1.031 
        gamma11 = Paramaters["HMevo_param"]
    #putting the parameters together for inclusion in the Moster 2010 equation
    M = M10 + M11*zparameter
    N = SHMnorm10 + SHMnorm11*zparameter
    b = beta10 + beta11*zparameter
    g = gamma10 + gamma11*zparameter

    # Moster 2010 eq2
    if ((np.shape(DM) == np.shape(z)) or np.shape(z) == (1,) or np.shape(z) == ()) and Pairwise:
        SM =  np.power(10, DM) * (2*N*np.power( (np.power(np.power(10,DM-M), -b) + np.power(np.power(10,DM-M), g)), -1))
    else:
        if (np.shape(DM)[0] != np.shape(z)[0]):
            M = np.full((np.size(DM), np.size(z)), M).T
            N = np.full((np.size(DM), np.size(z)), N).T
            b = np.full((np.size(DM), np.size(z)), b).T
            g = np.full((np.size(DM), np.size(z)), g).T
            DM = np.full((np.size(z), np.size(DM)), DM)
        SM =  np.power(10, DM) * (2*N*np.power( (np.power(np.power(10,DM-M), -b) + np.power(np.power(10,DM-M), g)), -1))
    #Adding Scatter
    if(ScatterOn):
        Scatter_Arr = np.random.normal(scale = Scatter, size = np.shape(SM))
        return( np.log10(SM) + Scatter_Arr)
    else:
        return( np.log10(SM))