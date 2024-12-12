"""
Script repurposed from Abhishek Maniyar's DopplerCIB github. 

Author: Tanveer Karim
Last Updated: 12 Dec 2024 (Fixed CIB-CIB bugs; now matches DopplerCIB)
Updated: 18 Nov 2024 (Added CIB-CIB function)
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
#geo = consts.Hz_over_c_times_chi2.value
geo = (consts.dchi_dz/consts.chi2).value

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
wcibwgal = pc.w_cibxgal
wcibwcib = pc.w_cibxcib

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
    prefact = geo * wcibwgal
    
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
        
###--C_CIB,CIB--###

def cibcrosscib_cell_2h(cibterm_nu, cibterm_nu_prime):
    """
    Returns P_{CIB X CIB'} 2-halo term.
    
    Note that the jnu terms get cancelled out by the ones in W_CIB
    
    Cell = int dz/c * H(z)/chi^2(z) * W_nu * W_nu' * Plin * integral_nu * integral_nu'
    integral_nu = integral_nu' = int dlog10Mh * I-term * b(Mh, z) * HMF
    
    Args:
        
    Returns: 
        C_ell : of shape (k)
    """
    
    prefact = geo * wcibwcib
    
    # integrals
    integrand_nu = cibterm_nu * hmfzTXbias 
    integral_nu = simpson(y=integrand_nu, dx=dm, axis=1) #(k,Mh,z)
    
    integrand_nu_prime = cibterm_nu_prime * hmfzTXbias 
    integral_nu_prime = simpson(y=integrand_nu_prime, dx=dm, axis=1) #(k,Mh,z)
    
    #(k,z)
    cell_integrand = prefact * Pk_lin * integral_nu * integral_nu_prime
    cell_integrand[:,:,0] = 0 # set z = 0 to 0 to not encounter nan
    C_ell = simpson(cell_integrand, x = consts.Plin['z'], axis=2)
    
    return C_ell
    
def cibcrosscib_cell_1h(djc_nu, djc_nu_prime, djsub_nu, djsub_nu_prime,
                        uprof):
    """
    Returns P_{CIB X CIB'} 1-halo term.
    
    Note that the jnu terms get cancelled out by the ones in W_CIB
    
    Cell = int dz/chi^2 * dchi/dz * W_nu * W_nu' * int(t1+t2+t3)*HMF*dlogMh
    t1 = djc_nu * djsub_nu' * unfw
    t2 = djc_nu' * djsub_nu * unfw
    t3 = djsub_nu * djsub_nu' * unfw^2 
    
    Args:
        djc_nu, djc_nu_prime : central emissivity (Mh,z)
        djsub_nu, djsub_nu_prime : sat emissivity (Mh,z)
        uprof : Fourier halo profile (ell,Mh,z)
    Returns: 
        C_ell : of shape (ell = k)
    """
    
    prefact = geo * wcibwcib
    # extend dimensions to match with uprof
    djc_nu_re = djc_nu[np.newaxis,:,:]
    djc_nu_prime_re = djc_nu_prime[np.newaxis,:,:]
    djsub_nu_re = djsub_nu[np.newaxis,:,:]
    djsub_nu_prime_re = djsub_nu_prime[np.newaxis,:,:]
    
    integrand1 = djc_nu_re * djsub_nu_prime_re * uprof
    integrand2 = djc_nu_prime_re * djsub_nu_re * uprof 
    integrand3 = djsub_nu_re * djsub_nu_prime_re * uprof**2 
    integrand_tot = (integrand1 + integrand2 + integrand3) * hmfzT[0]#[:,:,1:]
    integral = simpson(y=integrand_tot, dx=dm, axis=1)
    
    if consts.Plin['z'][0] == 0:
        C_ell_1h = simpson(y=integral*prefact[1:],
                           x=consts.Plin['z'][1:],
                           axis = 1)
    
    return C_ell_1h