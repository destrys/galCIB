"""
Helper function and constants for analysis
"""

from astropy.cosmology import Planck18 as planck
from astropy import constants as apconst
from astropy.io import fits 
import astropy.units as u
from scipy import constants as spconst
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd 
import pickle

# analysis variables
NSIDE = 2048 #FIXME: current galaxy windows are at 1024 
LMAX = 2000
LMIN = 100
ell = np.arange(LMIN, LMAX)

# global variables 
speed_of_light = apconst.c # in ms^-1
k_B = spconst.k # Boltzmann constant J K^-1
hp = spconst.h # Planck's constant J Hz^-1
hp_over_kB = hp/k_B # h/kB for faster calculation of h/k * nu/T unit: K Hz^-1
hp_times_2_over_c2 = 2*apconst.h/speed_of_light**2 # 2h/c^2 for faster calculation unit: J m^-2 s^-1

KC = 1.0e-10  # Kennicutt constant for Chabrier IMF in units of Msol * yr^-1 * Lsol^-1
L_sun = 3.828e26 # From Abhishek's code 
fsub = 0.134 # from Abhi's code, further note below.
#         fraction of the mass of the halo that is in form of
#         sub-halos. We have to take this into account while calculating the
#         star formation rate of the central halos. It should be calulated by
#         accounting for this fraction of the subhalo mass in the halo mass
#         central halo mass in this case is (1-f_sub)*mh where mh is the total
#         mass of the halo.
#         for a given halo mass, f_sub is calculated by taking the first moment
#         of the sub-halo mf and and integrating it over all the subhalo masses
#         and dividing it by the total halo mass.

##--COSMOLOGY--##
OmegaM0 = planck.Om0
Ode0 = planck.Ode0
H0 = planck.H0

# linear power spectrum
#with open('data/plin_unit_h.p', 'rb') as handle:
with open('data/plin_unit_Mpc.p', 'rb') as handle:
    Plin = pickle.load(handle)

# halo mass function from colossus
with open('data/hmfz_h.p', 'rb') as handle:
    hmfz_dict = pickle.load(handle)
Mh_Msol = hmfz_dict['M_Msol_h']/planck.h # units of Msol
log10Mh = np.log10(Mh_Msol)
hmfz = hmfz_dict['hmfz_log10M'] * planck.h**3 # units of (Mpc)^-3

# useful cosmology arrays to precalculate
chi_list = planck.comoving_distance(Plin['z']) # comoving distances
Hz_list = planck.H(Plin['z']) # H(z)
Omegab_to_OmegaM_over_z = planck.Ob(Plin['z'])/planck.Om(Plin['z'])
rho_crit = (planck.critical_density(Plin['z'])).to(u.Msun/u.Mpc**3).value # units of Msol/Mpc^3
mean_density0 = (OmegaM0*planck.critical_density0).to(u.Msun/u.Mpc**3).value # Returns mean density at z = 0, units of Msol/Mpc^3
Hz_over_c_times_chi2 = Hz_list/(speed_of_light * chi_list**2)
Hz_over_c_times_chi2 = Hz_over_c_times_chi2.decompose(bases=[u.Mpc]) # convert to units of 1/Mpc^3
Hz_over_c_times_chi2[0] = 0 # since no bins there


# based on ell and z ranges, calculate new k and pk grids
def get_Pk_array(ell=ell, z=Plin['z']):
    nl = len(ell)
    nz = len(z)
    #nreds = len(self.zpk)
    Pk_int = np.zeros((nl, nz)) # (ell, z)
    k_grid_over_ell = np.zeros_like(Pk_int)
    #Pk_int = np.zeros((nl, nz)) # Pk interpolated

    for i in range(nz):
        k_grid_over_ell[:,i] = ell/chi_list[i]
        Pk_int[:,i] = np.interp(k_grid_over_ell[:,i], 
                              Plin['k'], 
                              Plin['pk'][i,:])

    # for i in range(nl):
    #     Pk_int[i, :] = np.interp(z, self.zpk, pk1[i, :])

    return Pk_int, k_grid_over_ell

Pk_array_over_ell, k_grid_over_ell = get_Pk_array()

##--SUBHALO MASS FUNCTION--##
# pre-calculate central halo mass Mhc = Mh * (1 - fsub)
Mhc_Msol = Mh_Msol * (1 - fsub)

# define subhalo mass function grid 
log10ms_min = 6 # Msun according to pg 11 of 2310.10848 
num_points = 120 # number of subhalo masses sampled for given Mh
ms_Msol = np.logspace(log10ms_min, np.log10(Mhc_Msol), num_points) # units of Msol

# ratio of ms to Mhc, needed in SFRsub calculation.
ms_to_Mhc_Msol = ms_Msol/np.expand_dims(Mhc_Msol, axis = 0) 

# subhalo mass function
# based on 10 of 0909.1325.
#FIXME: is this the state of the art? 
def subhmf(m, M):
    """Vectorized f function that takes m and M arrays."""
    
    mass_ratio = m/M[np.newaxis, :]
    res = 0.13 * (mass_ratio)**(-0.7)
    expterm = -9.9 * (mass_ratio)**2.5
    
    res = res * np.exp(expterm) * np.log(10) # convert dN/dlnM_sub to dN/dlog_10M_sub
    
    return res 

subhalomf = subhmf(ms_Msol, Mhc_Msol)

###---GALAXY DICTIONARIES---### 
# store all relevant galaxy information
dict_gal = {}

# dict with ELG properties based on Karim et al. 2024
dict_gal['ELG'] = {}

with open('data/gal/dndz_extended.p', 'rb') as handle:
    dndz_ELG = pickle.load(handle)
    
# read in original data
dict_gal['ELG']['z'] = dndz_ELG['zrange']
dict_gal['ELG']['pz'] = dndz_ELG['dndz'].mean(axis=0) # mean of all photo-z realizations

# extend original data to match full redshift range
dict_gal['ELG']['pz'] = np.interp(Plin['z'], dict_gal['ELG']['z'], 
                                  dict_gal['ELG']['pz'],
                                  left=0, right=0) # do not extrapolate values
dict_gal['ELG']['z'] = Plin['z']

dict_gal['ELG']['mag_bias_alpha'] = 2.225

# shot noise 
dict_gal['ELG']['shot_noise'] = 4.618165243131944e-08 # from Karim et al. 2024 (assuming form delta_g,est = Wg * delta_g)

# dict with ELG HOD properties based on 
# Table 7 of Rocher et al. 2023
# dict_gal['ELG']['HOD'] = {}
# dict_gal['ELG']['HOD']['Ac'] = 0.1
# dict_gal['ELG']['HOD']['log10Mc'] = 11.64
# dict_gal['ELG']['HOD']['sigmaM'] = 0.30
# dict_gal['ELG']['HOD']['gamma'] = 5.47
# dict_gal['ELG']['HOD']['As'] = 0.41
# dict_gal['ELG']['HOD']['alpha'] = 0.81
# dict_gal['ELG']['HOD']['M0'] = 10**11.20
# # based on Eq 3.9 of Rocher et al. 2023
# dict_gal['ELG']['HOD']['M1'] = 10**13.84 * dict_gal['ELG']['HOD']['As']**(1/dict_gal['ELG']['HOD']['alpha'])

# dict with IR HOD properties based on pg.6 of 2310.10848
dict_gal['IR'] = {}
dict_gal['IR']['HOD'] = {}
dict_gal['IR']['HOD']['sigma_lnM'] = 0.4 #transition smoothing scale #FIXME: fixing this feels ad hoc?

##--SFR CALCULATIONS--##

# UNITS of Msol NOT Msol/h
def BAR(M, z):
    """
    Returns baryon accretion rate for models M21 and Y23.
    
    Represents of the total amount of mass growth,
    what fraction is baryon.
    
    Returns:
        bar : of shape (M, z)
    """
    
    def MGR(M, z):
        """
        Returns Mass Growth Rate. 
        
        From 2.37 of 2310.10848 (originally 2 of 1001.2304).
        
        Args:
            M : halo mass UNITS OF Msol NO little h
            z : redshift
        """
        
        # Reshape M and z to enable broadcasting
        M = np.expand_dims(M, axis = -1)  # Shape ((shape of M), 1)
        z = z[np.newaxis, :]  # Shape (1, len(z))
        
        res = 46.1 * (M/1e12)**1.1 * (1 + 1.11 * z) * np.sqrt(OmegaM0 *(1+z)**3 + Ode0)
        
        return res
    
    bar = MGR(M, z) * Omegab_to_OmegaM_over_z
    
    return bar

# pre-calculate BAR of central and sub haloes based on fsub 
bar_c = BAR(Mhc_Msol, Plin['z']) # shape (Mh, z)
bar_sub = BAR(ms_Msol, Plin['z']) #shape (ms, Mh, z)

## pre-calculate S_eff variables (parametrized version)
# Planck frequencies for CIB are: (100, 143, 217, 353, 545, 857) GHz frequencies
ghz = 1e9
nu_list = np.array([100, 143, 217, 353, 545, 857]) * ghz   # convert GHz to Hz 

# M23 Seff modeling 
snuaddr = 'data/filtered_snu_planck.fits'
hdulist = fits.open(snuaddr)
redshifts_M23 = hdulist[1].data
snu_eff_M23 = hdulist[0].data[:-1, :]  # in Jy/Lsun  # -1 because we are
# not considering the 3000 GHz channel which comes from IRAS
hdulist.close()

# interpolate over redshift
snu_eff_interp_func_M23 = interp1d(redshifts_M23, snu_eff_M23,
                                   kind='linear',
                                   bounds_error=False, 
                                   fill_value=0.)
snu_eff_z = snu_eff_interp_func_M23(Plin['z'])

# For parametric model, generate grid: nu_prime = (1 + z) * nu
# broad cast properly to get nu_prime_list of shape (nu, z)
nu_grid = np.linspace(1e2,1e3,10000)*ghz # sample 10k points from 100 to 1000 Ghz

#FIXME: does this need to be over Planck or can we match ELG spacing? 
nu_primes = nu_grid[:, np.newaxis] * (1 + redshifts_M23[np.newaxis, :]) # match redshift grid as Planck for convolution
chi_cib = planck.comoving_distance(redshifts_M23).value # in Mpc

# Lagrangian radius of a dark matter halo
#FIXME: Mh is in units of little h
mass_to_radius = (3*(Mhc_Msol)/(4*np.pi*mean_density0))**(1/3) # units of Mpc, shape (Mh,)

def get_k_R():
    """
    we need a c-M relation to calculate the fourier transform of the NFW
    profile. We use the relation from https://arxiv.org/pdf/1407.4730.pdf
    where they use the slope of the power spectrum with respect to the
    wavenumber in their formalism. This power spectrum slope has to be
    evaluated at a certain value of k such that k = kappa*2*pi/rad
    This kappa value comes out to be 0.69 according to their calculations.
    """
    rad = mass_to_radius # units of Mpc
    kappa = 0.69
    
    res = kappa * 2 * np.pi / rad
    
    return res

k_R = get_k_R() # shape (Mh,) # units of 1/Mpc