"""
Contains default S_nu models. 
M21 : Non-parametric model from Planck.
Y23 : Parametric model inspired by S12.
"""

import numpy as np 
from astropy.io import fits
import glob

from .registry import register_snu_model 
from scipy.special import lambertw
from scipy.interpolate import RectBivariateSpline

from scipy.constants import h, k, c  # Planck and Boltzmann in SI
kB_over_h = k/h # for nu0z 
h_over_kB = h/k 

MPC_TO_M = 3.085677581e22  # meters
WATT_TO_JY = 1e26          # Jy
L_SUN = 3.828e26           # watts

###--PARAMETRIC MODEL BASED ON Y23--###

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

###--NON-PARAMETRIC SED MODEL BASED ON M21--###    

def _fix_planck_file_list(file_list):
    """Fix column alignment issue as in the Planck SED files."""
    a, b = file_list[95], file_list[96]
    for i in range(95, 208):
        file_list[i] = file_list[i + 2]
    file_list[208] = a
    file_list[209] = b
    return file_list

def _load_raw_planck_seds(data_dir):
    # Load redshifts from Planck FITS file
    redshift_path = f"{data_dir}/filtered_snu_planck.fits"
    with fits.open(redshift_path) as hdulist:
        redshifts = hdulist[1].data

    # Load wavelength array (assumed to be in microns)
    wave_micron = np.loadtxt(
        f"{data_dir}TXT_TABLES_2015/EffectiveSED_B15_z0.012.txt"
    )[:, [0]]
    
    # Convert to frequency in Hz
    freq = c * 1e6 / wave_micron#[:, 0]  # m/s × 1e6 to convert μm to m

    # Load SED data files
    file_list = sorted(glob.glob(f"{data_dir}/TXT_TABLES_2015/*.txt"))
    file_list = _fix_planck_file_list(file_list) # fix column issue
    seds = np.stack([np.loadtxt(f)[:, 1] for f in file_list], axis=1)  # (Nfreq, Nz)

    return freq[::-1], redshifts, seds[::-1, :]  # Reversing to match increasing freq


def _compute_L_IR(seds, freqs, z_planck, cosmo):
    """
    Compute total IR luminosity from SED using interpolation and integration.
    
    Args:
        seds : SEDs measured by Planck in Jy/Lsun #FIXME: double check
        freq : Frequency grid in Hz 
        z_planck : Redshifts at which Planck measured seds 
        cosmo : Cosmology object
    Returns:
        L_IR_vals : Interpolated IR luminosity (z,)
    """
    
    fmin, fmax = 2.998e11, 3.747e13  # Hz: 1000 μm to 8 μm
    fint = np.logspace(np.log10(fmin), np.log10(fmax), 10000)

    L_IR_vals = np.zeros(len(z_planck))
    for i, z in enumerate(z_planck):
        
        # convert dL from units of Mpc/h to metre
        dL = (cosmo.cosmo.luminosity_distance(z)/cosmo.h) * MPC_TO_M
        
        L_nu = seds[:, i] * 4 * np.pi * dL**2 / ((1 + z) * WATT_TO_JY)
        
        # Interpolate from Planck grid to our grid
        L_interp = np.interp(np.log10(fint), np.log10(freqs), L_nu[::-1])
        L_IR_vals[i] = np.trapz(L_interp, fint)

    return L_IR_vals

def snu_M21_factory(data_dir="../data/"):
    """
    Factory function that returns a callable S_nu(theta, z)
    using precomputed non-parametric SEDs from Planck (M21).
    
    Returns:
        snu_M21(theta_snu) callable
    """
    freq, z_grid, raw_seds = _load_raw_planck_seds(data_dir)
    L_IR_vals = _compute_L_IR(raw_seds, freq, z_grid)

    # Normalize SEDs to total IR luminosity
    seds_normed = raw_seds * L_SUN / L_IR_vals[np.newaxis, :]

    # Interpolator in frequency [Hz] and redshift
    interpolator = RectBivariateSpline(freq, z_grid, seds_normed)

    def snu_M21(theta_unused, nu, z):
        """
        Evaluates the SED model for given nu and z.
        
        Args:
            theta_unused: placeholder for interface compatibility
            nu : frequency array (Hz)
            z  : redshift array
        Returns:
            SED of shape (len(nu), len(z))
        """
        return interpolator(nu, z)

    return snu_M21

def register_default_snu_models():
    register_snu_model("Y23", snu_Y23_factory)
    register_snu_model("M21", snu_M21_factory)