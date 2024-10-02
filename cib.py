"""
This module contains functions useful for CIB halo modelling.
"""

import numpy as np
import consts

# integrates using simpson method 
from scipy.integrate import simpson

# bivariate interpolation over a rectangular mesh
from scipy.interpolate import RectBivariateSpline

# Lambert W function solver
from scipy.special import lambertw

# physical constants
c_light = consts.speed_of_light
c_light_kms = c_light/1000
KC = consts.KC
L_sun = consts.L_sun
hp = consts.hp
kB = consts.k_B
hp_over_kB = consts.hp_over_kB
hp_times_2_over_c2 = consts.hp_times_2_over_c2

# cosmology constants
Om0 = consts.OmegaM0
Ode0 = consts.Ode0
bar_c = consts.bar_c
bar_sub = consts.bar_sub

# halo constants
Mh = consts.Mh
Mhc = consts.Mhc # central galaxies halo mass based on fsub
ms = consts.ms # subhalo mass grid based on fsub
ms_to_Mhc = consts.ms_to_Mhc # subhalo mass grid as a fraction with Mhc
hmfz = consts.hmfz # halo mass function
subhalomf = consts.subhalomf # subhalo mass function

# galaxy constants
dict_gal = consts.dict_gal['ELG']
chi = dict_gal['chi'] # comoving distance
z = dict_gal['z']

# SED constants
dgamma = 1.7
nu_primes = consts.nu_primes

## parametric dust model functions 
def Tdust(T0, alpha):
    """
    Returns dust temp as a function of z.
    From 1309.0382.
    
    Returns:
        res: of shape (z,)
    """

    res = T0*(1. + z)**alpha
    return res

def B_nu(nu, Td):
    """
    Returns Planck's blackbody function.
    
    Args:
        nu : frequency array in Hz of shape (nu, z)
        Td : dust temperature  of shape (z,)
        
    Returns:
        res : of shape (nu, z)
    """
    # Pre-factor depending only on nu, shape (nu, z)
    prefact = hp_times_2_over_c2 * nu**3

    # Ensure Td is broadcast correctly along the redshift dimension
    Td_re = Td[np.newaxis, :]  # Shape (1, z)
    
    # Calculate the exponential term
    x = hp_over_kB * nu/Td_re # Shape (nu, z)

    # Compute the Planck function
    res = prefact / (np.exp(x) - 1)  # Shape (nu, z)
    
    return res

def nu0_z(beta, Td):
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
    
    #FIXME: is this actually smoothly connecting?
    
    h_kT = hp_over_kB/Td # shape (z,)
    K = (dgamma + beta + 3)
    
    a = K/h_kT
    b = -1 * a
    c = -1 * h_kT
    
    x = a - lambertw(-b*c*np.exp(a*c))/c
    nu0z = np.real(x) # only take the real value
    
    return nu0z

# DONE
def Theta(params):
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

    """
    
    beta, T0, alpha = params
    theta = np.zeros_like(nu_primes) # shape (nu, z)
    Td = Tdust(T0, alpha)
    nu0z = nu0_z(beta, Td)
    
    def prenu0(beta, Td, nu):
        """
        Returns the gray-body part of the SED.
        """
        
        res = nu**beta * B_nu(nu, Td)
        
        return res 
        
    def postnu0(gamma, nu):
        """
        Returns the exponential decay part of the SED.
        """
        
        res = nu**(-gamma)
        
        return res
    
    # calculate SED 
    flag = nu_primes < nu0z[np.newaxis, :]
    theta = np.where(flag, 
                     prenu0(beta, Td, nu_primes), 
                     postnu0(dgamma, nu_primes))

    # normalize SED such that theta(nu0) = 1
    theta_normed = np.where(flag, 
         theta/prenu0(beta, Td, nu0z[np.newaxis, :]),
         theta/postnu0(dgamma, nu0z[np.newaxis, :]))
    
    return theta_normed

def Seff(params, model):
    """
    Returns the effective spectral energy distribution (SED) 
    which is the fraction of IR radiation at the
    rest-frame frequency (1 + z)nu. 
    
    It is defined as the mean flux density per total solar luminosity.
    
    Args:
        params: (beta, T0, alpha)
            beta: controls "grayness" of blackbody function
            T0, alpha: dust parameters
    Returns:
        seff : of shape (nu, z)
    """
    
    if (model == 'S12') | (model == 'Y23'):
        beta, T0, alpha = params
        seff = Theta(params)
        
    elif model == 'M23':
        seff = 'ok' # FIXME: process data
    else:
        print("Not correct model.")
        
    return seff  

## end of parametric dust model functions

def window_cib(nu, z):
    """
    Returns CIB radial kernel.
    
    Returns:
        w_cib : of shape (nu, z)
    """
    
    a = 1/(1+z)
    w_cib = a * jbar(nu) #FIXME
    
    return w_cib

def jbar(nu):
    """
    Returns the mean emissivity of the CIB halos.
    
    Args:
        nu : measurement frequency
        Mh : halo mass
    
    Returns:
        res : of shape (nu, z)
    """
    
    ##FIXME: maybe calculate phiCIB here and pass it down to relevant functions
    
    #djc/s returns of the shape (nu, Mh, z)
    djdlogmh = djc_dlogMh(nu) + djsub_dlogMh(nu) #FIXME
    dm = np.log10(Mh[1]/Mh[0])
    integrand = djdlogmh*hmfz
    
    # integrate along log mass axis 
    res = simpson(integrand, dx=dm, axis = 1)
    return res

def djc_dlogMh(params, nu, z, model):
    """
    Returns the emissivity of central galaxies per log halo mass. 
    
    from A6 of 2204.05299
    djc_dlogMh (Mh, z) = chi^2 * (1+z) * SFRc/K * S^eff_nu (z)
    
    Args:
        params : SFR model parameters, depends on model
        nu : measurement frequency 
        z : redshift 
        model : 'S12', 'M21' or 'Y23'
    Returns:
        jc : matrix of shape (nu, Mh, z)
    """
    
    # fraction of the mass of the halo that is in form of
    # sub-halos. We have to take this into account while calculating the
    # star formation rate of the central halos. It should be calculated by
    # accounting for this fraction of the subhalo mass in the halo mass
    # central halo mass in this case is (1-f_sub)*mh where mh is the total
    # mass of the halo.
    # for a given halo mass, f_sub is calculated by taking the first moment
    # of the sub-halo mf and and integrating it over all the subhalo masses
    # and dividing it by the total halo mass.
    
    #snu = self.snu #FIXME
    
    prefact = chi**2 * (1 + z)
    prefact = prefact/KC
    
    #FIXME: SFRc function will change as a function of models to be tested
    # prefact[z], SFRc[M, z], Seff[nu, z]
    #FIXME: precalculate Mh * (1 - fsub) for speedup
    jc = np.zeros(((len(nu_list), len(Mhc), len(z)))) #FIXME correct variable names
    
    sfrc = SFRc(params, model)
    seff = Seff(nu, z, model)   
    jc = prefact * sfrc * seff
    
    return jc

def djsub_dlogMh(params, model):
    """
    Returns the emissivity of satellite galaxies per log halo mass. 
    
    from A7 of 2204.05299
    djsub_dlogMh (Mh, z) = chi^2 * (1+z) S^eff_nu (z) * int (dN_dlogm_sub (m_sub|Mh) * SFRsub/K * dlogm_sub)
    
    Args:
        nu : measurement frequency 
        z : redshift 
        model : 'S12', 'M21' or 'Y23'
    Returns:
        jc : matrix of shape (nu, Mh, z)
    """
    
    prefact = chi**2 * (1 + z)
    prefact = prefact/KC # shape (z,)
    
    # integral in A7 of 2204.05299
    sfrsub = SFRsub(params, model)
    integrand = sfrsub * np.expand_dims(subhalomf, axis = -1)
    integral = simpson(y=integrand,
                       x = np.expand_dims(np.log10(ms), axis = -1),
                       axis = 0) # integrate along the ms axis, shape (Mh, z)
    
    # effective SED
    seff = "ok" #FIXME
    
    # jsub of shape (nu, Mh, z)
    jsub = prefact * integral * seff
    
    return jsub
    
def SFRsub(params, model):
    """
    Returns SFR of subhalos. 
    """
    if (model == 'S12'):
        # SFR_s (Mh, z) propto Sigma_s (M,z) Phi (z) 
        # from 2.34 of 2310.10848
        
        sigma_M0, mu_peak0, mu_peakp, delta = params
        
        phiCIB = phi_CIB(delta) #FIXME: what does this Phi represent?
        # Reshape phi(z) to broadcast across rows of Sigma
        phiCIB = phiCIB[np.newaxis, :]  # Make phi a row vector of shape (1, len(z))
        sigma = Sigma(sigma_M0, mu_peak0, mu_peakp)
        sfrc =  mean_N_IR_c * sigma * phiCIB 
        
    elif (model == 'M21'):
        # from 2.41 of 2310.10848
        # SFRs (m|M) = min(SFR(m), m/M * SFR(M))
        
        etamax, mu_peak0, mu_peakp, sigma_M0, tau, zc = params
        
        option1 = SFR(etamax, mu_peak0, mu_peakp, 
                      sigma_M0, tau, zc,
                      is_sub = True)
        
        option2 = ms_to_Mhc[:,:,np.newaxis] * SFR(etamax, mu_peak0, 
                                                  mu_peakp, sigma_M0, 
                                                  tau, zc, 
                                                  is_sub = False)[np.newaxis,:,:] # proper broadcasting to get shape (ms, Mh, z)
        
        sfrs = np.minimum(option1, option2)
        
        return sfrs
    
    elif (model == 'Y23'):
        mu_peak0, mu_peakp, sigma_M0, tau, zc = params
        
        option1 = SFR(etamax=1, mu_peak0=mu_peak0,
                      mu_peakp=mu_peakp, sigma_M0=sigma_M0,
                      tau=tau, zc=zc,
                      is_sub = True)
        
        option2 = ms_to_Mhc[:,:,np.newaxis] * SFR(etamax=1, 
                                                  mu_peak0=mu_peak0,
                                                  mu_peakp=mu_peakp,
                                                  sigma_M0=sigma_M0,
                                                  tau=tau, zc=zc,
                                                  is_sub = False)[np.newaxis,:,:] # proper broadcasting to get shape (ms, Mh, z)
        
        sfrs = np.minimum(option1, option2)
        
        return sfrs
    
## star-formation history functions        
def SFRc(params, model):
    """
    Returns star formation rate of central galaxies as a function of halo parameters and model.
    
    Args:
        params : model parameters
        Mhc : halo mass of central galaxies
        model : model name 
    """
    
    if model == 'S12':
        # SFR_c (Mh, z) propto Sigma_c (M,z) Phi (z) 
        # from 2.34 of 2310.10848
        
        sigma_M0, mu_peak0, mu_peakp, delta = params
        
        phiCIB = phi_CIB(delta) #FIXME: what does this Phi represent?
        # Reshape phi(z) to broadcast across rows of Sigma
        phiCIB = phiCIB[np.newaxis, :]  # Make phi a row vector of shape (1, len(z))
        sigma = Sigma(sigma_M0, mu_peak0, mu_peakp)
        sfrc =  mean_N_IR_c * sigma * phiCIB 
    
    elif model == 'M21':
        #SFR_c (Mh, z) = eta (Mh, z) * BAR (Mh, z)
        etamax, mu_peak0, mu_peakp, sigma_M0, tau, zc = params
        
        sfr = SFR(etamax, mu_peak0, mu_peakp, 
                  sigma_M0, tau, zc)
        sfrc = sfr * mean_N_IR_c #FIXME: what is this?

    elif model == 'Y23':
        mu_peak0, mu_peakp, sigma_M0, tau, zc = params
        
        # Model cannot constrain etamax so set to 1.
        # Normalization is absorbed by L0 param in SED.
        sfr = SFR(etamax = 1, mu_peak0=mu_peak0,
                  mu_peakp=mu_peak0, sigma_M0=sigma_M0,
                  tau=tau, zc=zc)
        sfrc = sfr * mean_N_IR_c #FIXME: what is this?
        
    else:
        print("Not correct model.")
        
    return sfrc 

# DONE
def SFR(etamax, mu_peak0, mu_peakp, 
        sigma_M0, tau, zc, is_sub = False):
    """
    Returns star formation rate for models
    M21 and Y23.
    
    Args:
        is_sub : flag for whether SFR is for subhaloes
    """
    
    eta_val = eta(etamax, mu_peak0, mu_peakp, 
                  sigma_M0, tau, zc, is_sub = is_sub) 
    
    if is_sub:
        sfr = eta_val * bar_sub 
    else:
        sfr = eta_val * bar_c

    return sfr

# DONE
def eta(etamax, mu_peak0, mu_peakp, sigma_M0,
        tau, zc, is_sub):
    """
    Returns star formation efficiency parameter.
    
    From 2.38 of 2310.10848.
    eta (M, z) = eta_max * exp(-0.5((ln M - ln Mpeak(z))/sigmaM(z))^2)
    M = Mh and z = z so these are fixed. 
    sigmaM represents range of halo masses over which star formation is efficient. 
    
    Args:
        etamax : highest star formation efficiency
        mu_peak0 : peak of halo mass contributing to IR emissivity at z = 0
        mu_peakp : rate of change of halo mass contributing to IR emissity at higher z 
        sigma_M0 : halo mass range contributing to IR emissivity below peak mass
        zc : the redshift below which the mass window for star formation starts to evolve
        tau : rate of zc evolution 
        is_sub : flag for whether this applies to subhaloes
    
    Returns:
        eta_val : of shape (Mh, z)
    """
    
    # Reshape M and z for broadcasting
    if is_sub:
        M_re = np.expand_dims(ms, axis = -1) # Shape (ms, Mh, 1)
    else:
        M_re = np.expand_dims(Mhc, axis = -1) # Shape (Mh, 1)
    z_re = z[np.newaxis, :]  # Shape (1, len(z))
    
    Mpeak = mu_peak0 + mu_peakp * z_re/(1+z_re) # M_peak may change with z 
    
    # parametrization based on 2.39 of 2310.10848.
    sigmaM = np.where(M_re < Mpeak, sigma_M0, 
                      sigma_M0 * (1 - tau/zc * 
                                  np.maximum(0, zc - z_re)))  # Shape (len(M), len(z))
    
    eta_val = etamax * np.exp(-0.5 * ((np.log(M_re) - np.log(Mpeak))/sigmaM)**2)
    
    return eta_val
         
# DONE
def phi_CIB(delta):
    """
    Returns redshift kernel of CIB contribution. 
    
    from 2.26 of 2310.10848 (originally 22 of 1109.1522).
    
    Phi(z) = (1 + z)^delta
    
    Args:
        delta : power index defining redshift evolution contribution.
    """
    
    phi = (1 + z)**delta
    
    return phi

def Sigmasub(sigma_M0, mu_peak0, mu_peakp):
    
    """
    Returns Luminosity-Mass relationship of satellite galaxies. 
    From 2.33 of 2310.10848.
    
    Sigma_s(M) = integrate from M_min to M 
    integrand = d ln m dN_sub/dln m (m | M) Sigma(M)
    
    Args:
        sigma_M0 : halo mass range contributing to IR emissivity 
        mu_peak0 : peak of halo mass contributing to IR emissivity at z = 0
        mu_peakp : rate of change of halo mass contributing to IR emissity at higher z  
    
    Returns : 
        res : of shape (Mh, z)
    """
    
    # Represents minimum halo mass that can host subhalos
    Mmin = 1e6 #Msun according to pg 11 of 2310.10848. 
    
    # Discretize the log-space between Mmin and Mmax for all Mh values
    def log_m_range_vectorized(M, num_points=100):
        """Create a 2D log-spaced array of m values for each M."""
        
        M_log_min = np.log(Mmin) 
        M_log_vals = np.log(M)
        log_m_vals = M_log_min + (np.linspace(0, 1, num_points) * (M_log_vals[:, np.newaxis] - M_log_min))  # Shape (len(M), num_points)
        m_vals = np.exp(log_m_vals)  # Convert back to m values
        return m_vals  # Shape (len(M), num_points)

    # Generate the 2D m grid for each Mh
    m_vals = log_m_range_vectorized(Mh)  # Shape (Mh, m) here m is of length num_points

    # Compute subhalo function (m, M)
    subhalo_func = subhmf(m_vals, Mh)  # Shape (Mh, m)
    
    # Compute Sigma(m, z) for all m and z
    Sigma_vals =  Sigma(sigma_M0, mu_peak0, mu_peakp)
    
    # Integrate over ln m
    ln_m_vals = np.log(m_vals)
    
    #FIXME: can take out Sigma since it does not depend on m? 
    integral = simpson(subhalo_func, x=ln_m_vals, axis=1)  # Shape (Mh,)
    
    # Combine results: Sigmas(M, z) = Sigma(M, z) * integral
    res = Sigma_vals * integral[:, np.newaxis]  # Shape (Mh, z)
    
    return res

# DONE 
def Sigma(sigma_M0, mu_peak0, mu_peakp):
    """
    Returns the Luminosity-Mass relation.
    From 2.30 of 2310.10848 (originally 23 of 1109.1522)
    
    Sigma(M) = M * 1/sqrt(2*pi*sigma_M,0^2) * exp[-0.5(ln M - lnM_peak)^2/sigma_M,0^2]
    
    Args:
        sigma_M0 : halo mass range contributing to IR emissivity 
        mu_peak0 : peak of halo mass contributing to IR emissivity at z = 0
        mu_peakp : rate of change of halo mass contributing to IR emissity at higher z 
    
    Returns: 
        res : of shape (Mh, z)
    """
    
    
    M_peak = mu_peak0 + mu_peakp * z/(1+z) # M_peak may change with z 

    # broadcast properly since Sigma is of shape (Mh, z)
    M = Mhc[:, np.newaxis] # Make M a column vector of shape (len(Mh), 1)
    M_peak_z = M_peak[np.newaxis, :]  # Make M_peak(z) a row vector of shape (1, len(z))
    
    prefact = M/np.sqrt(2 * np.pi * sigma_M0**2)
    
    expterm = -0.5 * ((np.log(M) - np.log(M_peak_z))/sigma_M0)**2
    
    res = prefact * np.exp(expterm)
    
    return res


# #FIXME
# def Seff_planck():
#     """
#     Returns the effective SEDs for the CIB for 
#     Planck (100, 143, 217, 353, 545, 857) and
#     IRAS (3000) GHz frequencies.
    
#     Args: 
    
#     Returns:
#         unfiltered_snu : spline approximant function that takes (nu, z)
#     """
    
#     def L_IR(snu_eff, freq_rest, redshifts): #FIXME: explanation
#         """
#         Returns ??
#         """
        
#         fmax = 3.7474057250000e13  # 8 microns in Hz
#         fmin = 2.99792458000e11  # 1000 microns in Hz
#         no = 10000
#         fint = np.linspace(np.log10(fmin), np.log10(fmax), no)
#         L_IR_eff = np.zeros((len(redshifts)))
#         dfeq = np.array([0.]*no, dtype=float)
#         for i in range(len(redshifts)):
#             L_feq = snu_eff[:, i]*4*np.pi*(Mpc_to_m*cosmo.luminosity_distance(redshifts[i]).value)**2/(w_jy*(1+redshifts[i]))
#             Lint = np.interp(fint, np.log10(np.sort(freq_rest[:, i])),
#                                 L_feq[::-1])
#             dfeq = 10**(fint)
#             L_IR_eff[i] = np.trapz(Lint, dfeq)
#         return L_IR_eff
    
#     list_of_files = sorted(glob.glob('./data/TXT_TABLES_2015/./*.txt'))
#     a = list_of_files[95] #FIXME: why 95 and 96
#     b = list_of_files[96]
    
#     for i in range(95, 208): #FIXME: why this range 
#         list_of_files[i] = list_of_files[i+2]
#     list_of_files[208] = a
#     list_of_files[209] = b

#     # wavelengths are in microns
#     wavelengths = np.loadtxt('./data/TXT_TABLES_2015/EffectiveSED_B15_z0.012.txt')[:,[0]]
    
#     # Need freq in GHz so multiply by the following numerical factor 
#     freq = c_light_kms/wavelengths
#     numerical_fac = 1.
#     freqhz = freq*1e3*1e6
#     freq *= numerical_fac
#     #freq_rest = freqhz*(1+redshifts) #FIXME what are these redshifts? same as z?
#     freq_rest = freqhz*(1 + z)

#     n = len(wavelengths) # number of wavelength bins 

#     snu_unfiltered = np.zeros([n, len(z)])
#     for i in range(len(list_of_files)): #FIXME: redshift of data does not map to galaxy redshift?
#         snu_unfiltered[:, i] = np.loadtxt(list_of_files[i])[:, 1]
#     L_IR15 = L_IR(snu_unfiltered, freq_rest, z) #FIXME: define L_IR

#     for i in range(len(list_of_files)):
#         snu_unfiltered[:, i] = snu_unfiltered[:, i]*L_sun/L_IR15[i]
#         #FIXME: define L_sun

#     # Currently unfiltered snus are ordered in increasing wavelengths,
#     # we re-arrange them in increasing frequencies i.e. invert it
#     freq = freq[::-1]
#     snu_unfiltered = snu_unfiltered[::-1]
#     # snu_unfiltered = snu_eff
#     # freq = np.array([100., 143., 217., 353., 545., 857., 3000.])
#     unfiltered_snu = RectBivariateSpline(freq, z, snu_unfiltered)

#     return unfiltered_snu

# #FIXME
# def Seff_parametric(nu, z, model):
#     """
#     Returns the effective IR SED of CIB galaxies.
    
#     Args:
#         nu : frequency
#         z : redshift
#         model : 'S12', 'M21' or 'Y23'
        
#     Returns:
#         SED : shape (nu, z), values corresponding to ((1+z)*nu, z)
#     """
    
#     if ((model == 'S12') | (model == 'Y3')):
#         # SED is proportional but proportionality constant
#         # absorbed by L0 normalization
#         SED = theta((1 + z)*nu, z)/(chi**2 * (1+z))
#     elif model == 'M21':
#         print("M21")
#     else:
#         print("Seff model is not properly specified.")
    
#     return SED