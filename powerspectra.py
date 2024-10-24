"""
Script repurposed from Abhishek Maniyar's DopplerCIB github. 

Author: Tanveer Karim
Last Updated: 16 Oct 2024
"""

import numpy as np

# integrates using simpson method 
from scipy.integrate import simpson, trapezoid
from scipy.interpolate import interp1d

# import local modules
import consts
import precalc as pc
import gal
import cib
import halo as h

# cosmology constants
Hz = consts.Hz_list
chi = consts.chi_list

# halo constants 
Mh = consts.Mh_Msol
dm = consts.log10Mh[1] - consts.log10Mh[0]
hmfz = consts.hmfz
hmfzT = pc.hmfzT
biasmz = pc.halo_biases

hmfzTXbias = hmfzT * biasmz

# expand 
hmfzT = hmfzT[np.newaxis,np.newaxis,:,:]
biasmz = biasmz[np.newaxis,np.newaxis,:,:]
Pk_lin = consts.Pk_array_over_ell[np.newaxis,:,:]

# survey parameters
#dict_gal = consts.dict_gal['ELG']

# precalc values
dlnpk_dlnk = pc.dlnpk_dlnk
rad200 = pc.rad200
concentration = pc.concentration
concentration_amp = pc.concentration_amp
w1w2 = pc.w_cibxgal

def pcl(theta, M, B):
    """
    Returns pseudo-C_ell.
    
    Args:
        theta : model parameters 
        M : coupling matrix (obtained from Skylens)
        B : binning operator (obtained from Skylens)
    Returns:
        pcl_combined : single vector [gg, gcib_357, gcib_545, gcib_857]
    """
    
    # calculate unbinned pcl
    c_gcib = cibgalcross_cell_tot(theta) #(nu, ell)
    pcl_gcib = c_gcib @ M #(nu, ell') # FIXME: need to calculate M
    pcl_gg = "ok" #FIXME 
    
    # FIXME: bin pcl
    
    pcl_combined = np.concatenate(pcl_gg_binned, pcl_gcib_binned) #FIXME: make sure gcib order is correct
    return pcl_combined

def cibgalcross_cell_tot(theta, cib_model,
                         plot=False): 
    """
    Returns C_{g, CIB} accounting for all halo terms.
    """
    
    hmalpha = theta[0]
    shotnoise = theta[1]
    gal_params = theta[2:10] # Ncen (4): gamma, log10Mc, sigmaM, Ac
                           # Nsat (4): As, M0, M1, alpha
    prof_params = theta[10:13] # fexp, tau, lambda_NFW
    cib_params = theta[13:] # SFR (6): etamax (only for M23) or L0 (only for Y23), mu_peak0, mu_peakp, sigma_M0, tau, zc
                       # SED (3): beta, T0, alpha (only for Y23)
    uprof = h.uprof_mixed(prof_params, 
                         rad200, 
                         concentration, 
                         concentration_amp) # (k, Mh, z)
    galterm = gal.galterm(gal_params, uprof, 
                          gal_type = 'ELG')
    cibterm = cib.cibterm(cib_params, uprof, cib_model)
    
    # radial kernal and prefactors 
    geo = consts.Hz_over_c_times_chi2.value
    prefact = geo * w1w2
    
    # expand dims to pass 
    galterm = galterm[np.newaxis,:,:,:] #(nu,k,Mh,z)
    cibterm = cibterm[3:]
        
    # calculate Pk of both halo terms
    oneh = cibgalcross_pk_1h(galterm, cibterm)
    twoh = cibgalcross_pk_2h(galterm, cibterm)
    
    pk_oneh_plus_2h = prefact * (oneh**(1/hmalpha) + twoh**(1/hmalpha))**hmalpha
    pk_oneh_plus_2h[:,:,0] = 0 # z = 0 is 0, otherwise integral gets NaN
    c_ell_1h_plus_2h = simpson(pk_oneh_plus_2h, x = consts.Plin['z'], axis=2)
    tot = c_ell_1h_plus_2h+shotnoise
    
    if plot: # for plotting purposes
        return tot, prefact*oneh, prefact*twoh
    else:
        return tot 

def cibgalcross_pk_1h(galterm, cibterm):
        """
        Returns 1-halo term of CIB X g power spectrum.
        
        From A13 of 2204.05299.
        Form is: 
            Pk_1h = int HMF * g-term * CIB-term * dlogMh
            
        Args:
            galterm : Nc + Ns * unfw (nu, k, Mh, z)
            cibterm : jc + js * unfw (nu, k, Mh, z)
        Returns:
            pk_1h : (nu, k, z)
        """
        
        integrand = hmfzT * galterm * cibterm 
        pk_1h = simpson(y=integrand, dx=dm, axis = 2)

        return pk_1h 

def cibgalcross_pk_2h(galterm, cibterm, plot = False):
        """
        Check the calculation first using two different integrals for both CIB
        and galaxies, then check with a single integral for both after
        multiplying. If both are same, then we can save one integral.
        
        Pk_2h (nu, k, z) = b_g * b_CIB * P_lin/(nbar * jbar)
        
        b_g = int HMF(Mh, z) * b(Mh, z) * gal_term (k, Mh, z) dlogMh
        b_CIB = int HMF(Mh, z) * b(Mh, z) * cib_term (nu, k, Mh, z) dlogMh
        
        Returns:
            pk_2h : (nu, k, z)
        """
        # pk = self.uni.Pk_array(self.ell, self.z)
        
        # galaxy bias term: int HMF(Mh, z) * b(Mh, z) * gal_term (k, Mh, z) dlogMh
        
        integrand = hmfzTXbias * galterm # (nu,k,Mh,z)
        integral_g = simpson(y=integrand, dx=dm, axis=2) #(nu,k,z)
        
        # CIB bias term: int HMF(Mh, z) * b(Mh, z) * cib_term (nu, k, Mh, z) dlogMh
        integrand = hmfzTXbias * cibterm
        integral_cib = simpson(y=integrand, dx=dm, axis=2) # (nu,k,z)
        pk_2h = integral_g * integral_cib * Pk_lin
        
        if plot:
            return pk_2h, integral_g, integral_cib
        else:
            return pk_2h

# def cibgalcross_cell_shot():
    
#     """
#     Returns galaxy X CIB shot-noise. 
#     """
    
#     freq = np.array([100., 143., 217., 353., 545., 857.])
#     #nfreq = len(self.nu0) #nu_list
#     nfreq = len(freq)
#     nl = len(consts.ell) #
#     shotcib = np.zeros((nfreq, nl))
#     # 100, 143, 217, 353, 545, 857 Planck
#     # values for 100 and 143 i.e. 3 and 7 are fake
    
#     # read sar file #FIXME: what is this?
#     strfig = "allcomponents_lognormal_sigevol_1p5zcutoff_nolens_onlyautoshotpar_no3000_gaussian600n857n1200_planck_spire_hmflog10.txt"
#     cibres = "data/one_halo_bestfit_"+strfig
#     sar = np.loadtxt(cibres)[4:8, 0]  # 217, 353, 545, 857
    
#     # """
#     #freq = np.array([100., 143., 217., 353., 545., 857.])
#     sa = np.array([1.3*0.116689509208305475, 1.3*0.8714424869942087, 14., 357., 2349., 7407.])
#     sa[2:] = sar
#     #res = interp1d(freq, sa, kind='linear', bounds_error=False, fill_value="extrapolate")
#     #shotval = res(self.nu0)
#     shotval = sa
#     """
#     freq = self.nu0
#     shotval = sar
#     # """
#     for i in range(nfreq):
#         shotcib[i, :] = shotval[i]
#     # if max(self.nu0) > max(freq):
#     #     print ("shot noise values for frequencies higher than 857 GHz extrapolated using the values for Planck")

#     r_l = 1.0 # FIXME: line 333 of run_driver ?
#     shotgal = dict_gal['shot_noise']*np.ones_like(consts.ell)
#     crossshot = r_l*np.sqrt(shotcib*shotgal)
#     return crossshot

# shotnoise = cibgalcross_cell_shot() # calculate once to pass to cibgalcross_cell_tot