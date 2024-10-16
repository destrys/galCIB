"""
Script repurposed from Abhishek Maniyar's DopplerCIB github. 

Author: Tanveer Karim
Last Updated: 12 Oct 2024
"""

import numpy as np

# integrates using simpson method 
from scipy.integrate import simpson
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
hmfz = consts.hmfz
biasmz = pc.halo_biases

dict_gal = consts.dict_gal['ELG']
chi = dict_gal['chi']
dchi_dz = dict_gal['dchi_dz']

# precalc values
dlnpk_dlnk = pc.dlnpk_dlnk
rad200 = pc.rad200
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

def cibgalcross_cell_tot(theta, cib_model): 
    """
    Returns C_{g, CIB} accounting for all halo terms.
    """
    
    hmalpha = theta[0]
    gal_params = theta[1:9] # Ncen (4): gamma, log10Mc, sigmaM, Ac
                           # Nsat (4): As, M0, M1, alpha
    prof_params = theta[9:12] # fexp, tau, lambda_NFW
    cib_params = [12:] # SFR (6): etamax (only for M23), mu_peak0, mu_peakp, sigma_M0, tau, zc
                       # SED (3): beta, T0, alpha (only for Y23)
    
    unfw = h.uprof_mixed(prof_params, 
                         rad200, dlnpk_dlnk)
    galterm = gal.galterm(gal_params, unfw, 
                          gal_type = 'ELG')
    cibterm = cib.cibterm(cib_params, unfw, cib_model)
    
    # radial kernal and prefactors 
    geo = Hz/chi**2
    prefact = geo * w1w2
    
    # calculate Pk of both halo terms
    oneh = cibgalcross_pk_1h(galterm, cibterm)
    twoh = cibgalcross_pk_2h(galterm, cibterm)
    
    pk_oneh_plus_2h = prefact * (oneh**(1/hmalpha) + twoh**(1/hmalpha))**hmalpha
    c_ell_1h_plus_2h = simpson(pk_oneh_plus_2h, x = consts.Plin['z'])
    
    # shot noise term
    shot = cibgalcross_cell_shot()
    
    tot = c_ell_1h_plus_2h+shotnoise
    
    return tot
    
def cibgalcross_cell_shot():
    
    """
    Returns galaxy X CIB shot-noise. 
    """
    
    freq = np.array([100., 143., 217., 353., 545., 857.])
    #nfreq = len(self.nu0) #nu_list
    nfreq = len(freq)
    nl = len(consts.ell) #
    shotcib = np.zeros((nfreq, nl))
    # 100, 143, 217, 353, 545, 857 Planck
    # values for 100 and 143 i.e. 3 and 7 are fake
    
    # read sar file #FIXME: what is this?
    strfig = "allcomponents_lognormal_sigevol_1p5zcutoff_nolens_onlyautoshotpar_no3000_gaussian600n857n1200_planck_spire_hmflog10.txt"
    cibres = "data/one_halo_bestfit_"+strfig
    sar = np.loadtxt(cibres)[4:8, 0]  # 217, 353, 545, 857
    
    # """
    #freq = np.array([100., 143., 217., 353., 545., 857.])
    sa = np.array([1.3*0.116689509208305475, 1.3*0.8714424869942087, 14., 357., 2349., 7407.])
    sa[2:] = sar
    #res = interp1d(freq, sa, kind='linear', bounds_error=False, fill_value="extrapolate")
    #shotval = res(self.nu0)
    shotval = sa
    """
    freq = self.nu0
    shotval = sar
    # """
    for i in range(nfreq):
        shotcib[i, :] = shotval[i]
    # if max(self.nu0) > max(freq):
    #     print ("shot noise values for frequencies higher than 857 GHz extrapolated using the values for Planck")

    r_l = 1.0 # FIXME: line 333 of run_driver ?
    shotgal = dict_gal['shot_noise']*np.ones_like(consts.ell)
    crossshot = r_l*np.sqrt(shotcib*shotgal)
    return crossshot

shotnoise = cibgalcross_cell_shot() # calculate once to pass to cibgalcross_cell_tot

def cibgalcross_pk_1h(galterm, cibterm):
        """
        Returns 1-halo term of CIB X g power spectrum.
        
        From A13 of 2204.05299.
        Form is: 
            Pk_1h = int HMF * g-term * CIB-term * dlogMh
            
        Args:
            galterm : Nc + Ns * unfw (k, Mh, z)
            cibterm : jc + js * unfw (nu, k, Mh, z)
        Returns:
            pk_1h : (nu, k, z)
        """
        
        integrand = hmfz[np.newaxis, np.newaxis,:,:] * np.expand_dims(galterm, axis=0) * cibterm
        pk_1h = simpson(y=integrand, x=np.log10(Mh), axis = 2)

        return pk_1h 

def cibgalcross_pk_2h(galterm, cibterm):
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
        # print (self.hmfmz.shape, self.biasmz.shape)
        
        # galaxy bias term: int HMF(Mh, z) * b(Mh, z) * gal_term (k, Mh, z) dlogMh
        integrand = hmfz[np.newaxis,:,:] * biasmz[np.newaxis,:,:] * galterm # (k,Mh,z)
        integral_g = simpson(y=integrand, x=consts.log10Mh, axis=1) #(k, z)
        
        # CIB bias term: int HMF(Mh, z) * b(Mh, z) * cib_term (nu, k, Mh, z) dlogMh
        integrand = hmfz[np.newaxis,np.newaxis,:,:] * biasmz[np.newaxis,np.newaxis,:,:] * cibterm
        integral_cib = simpson(y=integrand, x=consts.log10Mh, axis=2) # (nu,k,z)
        
        pk_2h = np.expand_dims(integral_g, axis=0) * integral_cib * Plin #FIXME: read in plin from consts
        
        return pk_2h
     

    
    """
    Returns C_ell of 2-halo terms 
    """
    
    chi = dict_ELG['chi']
    z = dict_ELG['z']
    
    dz = z # integrating over redshift 
    integrand_prefact = 1/chi**2
    integrand = integrand_prefact * W_cib * W_gal 
    integrand = integrand * P_cibXgal
    
    C_ell_2h = simpson(y = integrand, x = dz)
    
    return C_ell_2h