"""
Contains default S_nu models. 
M21 : Non-parametric model from Planck.
Y23 : Parametric model inspired by S12.
"""

import numpy as np 
from astropy.io import fits

def snu_M21(nu0, z, path_to_data_file):
    """
    Non-parametric SED model based on M21. 

    Args:
        nu : array-like
            Frequencies (CIB channels).
        z : array-like
            Redshift grid.

    Returns:
        ndarray of shape (len(nu), len(z)) representing SED values.
    """
    
    fname = path_to_data_file
    
    snuaddr = f'{fname}/filtered_snu_planck.fits'
    hdulist = fits.open(snuaddr)
    redshifts_M21 = hdulist[1].data # read in Planck redshifts
    hdulist.close()

    # wavelengths in microns
    wavelengths = np.loadtxt(f'{fname}/TXT_TABLES_2015/EffectiveSED_B15_z0.012.txt')[:,[0]]

    # convert wavelengths to frequency 
    c_light = 299792.458 # km/s
    freq = 299792.458/wavelengths
    
    # c_light is in Km/s, wavelength is in microns and we would like to
    # have frequency in GHz. So have to multiply by the following
    # numerical factor which comes out to be 1
    # numerical_fac = 1e3*1e6/1e9
    numerical_fac = 1.
    freqhz = freq*1e3*1e6
    freq *= numerical_fac
    freq_rest = freqhz*(1+redshifts_M21)

    # read in all the Planck SEDs as a function of redshift
    import glob
    list_of_files = sorted(glob.glob(f'{fname}/TXT_TABLES_2015/./*.txt'))
    a = list_of_files[95]
    b = list_of_files[96]
    
    # column adjustment issue fix
    for i in range(95, 208):
        list_of_files[i] = list_of_files[i+2]
    list_of_files[208] = a
    list_of_files[209] = b

    Mpc_to_m = 3.086e22  # Mpc to m
    L_sun = 3.828e26
    w_jy = 1e26  # Watt to Jy

    def L_IR(snu_eff, freq_rest, redshifts):
        # freq_rest *= ghz  # GHz to Hz
        fmax = 3.7474057250000e13  # 8 micros in Hz
        fmin = 2.99792458000e11  # 1000 microns in Hz
        no = 10000
        fint = np.linspace(np.log10(fmin), np.log10(fmax), no)
        L_IR_eff = np.zeros((len(redshifts)))
        dfeq = np.array([0.]*no, dtype=float)
        
        for i in range(len(redshifts)):
            L_feq = snu_eff[:, i]*4*np.pi*(Mpc_to_m*planck.luminosity_distance(redshifts[i]).value)**2/(w_jy*(1+redshifts[i]))
            Lint = np.interp(fint, np.log10(np.sort(freq_rest[:, i])),
                                L_feq[::-1])
            dfeq = 10**(fint)
            L_IR_eff[i] = np.trapz(Lint, dfeq)
        return L_IR_eff

    n = np.size(wavelengths)
    snu_unfiltered = np.zeros([n, len(redshifts_M21)])

    for i in range(len(list_of_files)):
        snu_unfiltered[:, i] = np.loadtxt(list_of_files[i])[:, 1]
    L_IR15 = L_IR(snu_unfiltered, freq_rest, redshifts_M21)

    for i in range(len(list_of_files)):
        snu_unfiltered[:, i] = snu_unfiltered[:, i]*L_sun/L_IR15[i]

    # Currently unfiltered snus are ordered in increasing wavelengths,
    # we re-arrange them in increasing frequencies i.e. invert it
    freq = freq[::-1]
    snu_unfiltered = snu_unfiltered[::-1]

    from scipy.interpolate import RectBivariateSpline
    unfiltered_snu = RectBivariateSpline(freq, redshifts_M21,
                                        snu_unfiltered)
    snu_eff_z = unfiltered_snu(nu0, z)
    
    return snu_eff_z