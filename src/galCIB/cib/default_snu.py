"""
Contains default S_nu models. 
M21 : Non-parametric model from Planck.
Y23 : Parametric model inspired by S12.
"""

import numpy as np 
from .registry import register_snu_model 
from scipy.special import lambertw

from scipy.constants import h, k, c  # Planck and Boltzmann in SI
kB_over_h = k/h # for nu0z 
h_over_kB = h/k 

def snu_Y23_factory(nu_prime, z):
    """
    Factory returns a callable: S_nu(theta_SED, nu_prime)
    
    Args:
        nu_prime : nu' = nu*(1+z) grid over which to evaluate SED
    """

    planck_prefactor = 2*h*nu_prime**3/c**2

    def snu_Y23(theta_SED):
        """
        Returns the modified SED for gray-body function 
        normalized to 1 at pivot freq.
        
        From 2.27 of 2310.10848.
        
        theta(nu') = nu'^beta * B_nu' (T) for n < nu0
                = nu'^(-gamma) for n >= nu0
        
        Args:
            params: (beta, T0, alpha)
                beta: controls "grayness" of blackbody function
                T0, alpha: dust parameters
        Returns:
            theta_normed : of shape (nu, z)
        """
    
        L0, beta_dust, T0, alpha_dust, gamma_dust = theta_SED
        Td = _compute_Tdust(T0, alpha_dust, z)
        nu0z = _compute_nu0_z(beta_dust, Td, gamma_dust)
        
        # calculate SED 
        flag = nu_prime < nu0z[np.newaxis, :]
        
                
        theta = np.where(flag, 
                        _compute_prenu0(beta_dust, Td, nu_prime, planck_prefactor), 
                        _compute_postnu0(nu_prime, gamma_dust))

        # normalize SED such that theta(nu0) = 1
        theta_normed = np.where(flag, 
            theta/_compute_prenu0(beta_dust, Td, nu0z, 2*h*nu0z**3/c**2),
            theta/_compute_postnu0(nu0z, gamma_dust))
                
        return theta_normed*L0
    
    return snu_Y23

def _compute_prenu0(beta, Td, nu, planck_prefactor):
    """
    Returns the gray-body part of the SED.
    """
    
    res = nu**beta * _compute_B_nu(nu, Td, planck_prefactor)
    return res 

def _compute_postnu0(nu, gamma_dust):
    """
    Returns the exponential decay part of the SED.
    """
    
    res = nu**(-gamma_dust)
    return res

def _compute_Tdust(T0, alpha_dust, z):
    """
    Returns dust temperature as a func. of z.
    """
    return T0 * (1 + z)**alpha_dust

def _compute_nu0_z(beta, Td, gamma):
    """
    Returns the pivot frequency as a function of redshift.
    
    For modified blackboy approximation, 
    we have dln[v^beta * B_nu(Td)]/dlnnu = -gamma 
    for nu=nu0 from 2.27 of 2310.10848.
    
    In order to find nu0 which is redshift dependent, we need to 
    use the Lambert W function. of the form x = a + be^(c*x). 
    Solution is given by: x = a - W(-bce^(ac))/c, check 
    https://docs.scipy.org/doc/scipy-1.13.0/reference/generated/scipy.special.lambertw.html
    for reference. 
    
    Here, x = nu0, a = K/h_KT, b = -a, c = -h_KT,
    where K = (gamma + beta + 3) and h_KT = h/(KT)
    
    Args:
        beta: controls "grayness" of blackbody function
        Td: dust temperature as a function of z
    Returns:
        nu0z: pivot frequencies as a function of redshift. Shape (z,)
    """
    
    K = (gamma + beta + 3)
    lambert_term = lambertw(-K*np.exp(-K))
    full_term = K + lambert_term
    nu0z = np.real(kB_over_h * Td * full_term )
    
    return nu0z

def _compute_B_nu(nu, Td, prefact):
    """
    Returns Planck's blackbody function.
    
    Args:
        nu : frequency array in Hz of shape (nu, z)
        Td : dust temperature  of shape (z,)
        
    Returns:
        res : of shape (nu, z)
    """
    # Pre-factor depending only on nu, shape (nu, z)
    #prefact = hp_times_2_over_c2 * nu**3

    # Ensure Td is broadcast correctly along the redshift dimension
    Td_re = Td[np.newaxis, :]  # Shape (1, z)
    
    # Calculate the exponential term
    x = h_over_kB * nu/Td_re # Shape (nu, z)

    # Compute the Planck function
    res = prefact / np.expm1(x)  # Shape (nu, z)
    
    return res

def register_default_snu_models():
    register_snu_model("Y23", snu_Y23_factory)       

# def snu_M21(nu0, z, path_to_data_file):
#     """
#     Non-parametric SED model based on M21. 

#     Args:
#         nu : array-like
#             Frequencies (CIB channels).
#         z : array-like
#             Redshift grid.

#     Returns:
#         ndarray of shape (len(nu), len(z)) representing SED values.
#     """
    
#     fname = path_to_data_file
    
#     snuaddr = f'{fname}/filtered_snu_planck.fits'
#     hdulist = fits.open(snuaddr)
#     redshifts_M21 = hdulist[1].data # read in Planck redshifts
#     hdulist.close()

#     # wavelengths in microns
#     wavelengths = np.loadtxt(f'{fname}/TXT_TABLES_2015/EffectiveSED_B15_z0.012.txt')[:,[0]]

#     # convert wavelengths to frequency 
#     c_light = 299792.458 # km/s
#     freq = 299792.458/wavelengths
    
#     # c_light is in Km/s, wavelength is in microns and we would like to
#     # have frequency in GHz. So have to multiply by the following
#     # numerical factor which comes out to be 1
#     # numerical_fac = 1e3*1e6/1e9
#     numerical_fac = 1.
#     freqhz = freq*1e3*1e6
#     freq *= numerical_fac
#     freq_rest = freqhz*(1+redshifts_M21)

#     # read in all the Planck SEDs as a function of redshift
#     import glob
#     list_of_files = sorted(glob.glob(f'{fname}/TXT_TABLES_2015/./*.txt'))
#     a = list_of_files[95]
#     b = list_of_files[96]
    
#     # column adjustment issue fix
#     for i in range(95, 208):
#         list_of_files[i] = list_of_files[i+2]
#     list_of_files[208] = a
#     list_of_files[209] = b

#     Mpc_to_m = 3.086e22  # Mpc to m
#     L_sun = 3.828e26
#     w_jy = 1e26  # Watt to Jy

#     def L_IR(snu_eff, freq_rest, redshifts):
#         # freq_rest *= ghz  # GHz to Hz
#         fmax = 3.7474057250000e13  # 8 micros in Hz
#         fmin = 2.99792458000e11  # 1000 microns in Hz
#         no = 10000
#         fint = np.linspace(np.log10(fmin), np.log10(fmax), no)
#         L_IR_eff = np.zeros((len(redshifts)))
#         dfeq = np.array([0.]*no, dtype=float)
        
#         for i in range(len(redshifts)):
#             L_feq = snu_eff[:, i]*4*np.pi*(Mpc_to_m*planck.luminosity_distance(redshifts[i]).value)**2/(w_jy*(1+redshifts[i]))
#             Lint = np.interp(fint, np.log10(np.sort(freq_rest[:, i])),
#                                 L_feq[::-1])
#             dfeq = 10**(fint)
#             L_IR_eff[i] = np.trapz(Lint, dfeq)
#         return L_IR_eff

#     n = np.size(wavelengths)
#     snu_unfiltered = np.zeros([n, len(redshifts_M21)])

#     for i in range(len(list_of_files)):
#         snu_unfiltered[:, i] = np.loadtxt(list_of_files[i])[:, 1]
#     L_IR15 = L_IR(snu_unfiltered, freq_rest, redshifts_M21)

#     for i in range(len(list_of_files)):
#         snu_unfiltered[:, i] = snu_unfiltered[:, i]*L_sun/L_IR15[i]

#     # Currently unfiltered snus are ordered in increasing wavelengths,
#     # we re-arrange them in increasing frequencies i.e. invert it
#     freq = freq[::-1]
#     snu_unfiltered = snu_unfiltered[::-1]

#     from scipy.interpolate import RectBivariateSpline
#     unfiltered_snu = RectBivariateSpline(freq, redshifts_M21,
#                                         snu_unfiltered)
#     snu_eff_z = unfiltered_snu(nu0, z)
    
#     return snu_eff_z