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

c_light = consts.speed_of_light
c_light_kms = c_light/1000
Mh = consts.Mh
hmfz = consts.hmfz
dict_gal = consts.dict_gal['ELG']
chi = dict_gal['chi']
z = dict_gal['z']
KC = consts.KC
L_sun = consts.L_sun
Ob_to_Om = dict_gal['Omegab_to_OmegaM_over_z']
hp = consts.hp
kB = consts.k_B
fsub = consts.fsub

#dust parameters
dalpha = 0.36
dT0 = 24.4 # Planck CIB 2013
dbeta = 1.75 
dgamma = 1.7

## parametric dust model functions 
def Tdust(z):
    """
    Returns dust temp as a function of z.
    From 1309.0382.
    
    Returns:
        res: of shape (z,)
    """

    res = dT0*(1. + z)**dalpha
    return res

def B_nu(nu, Td):
    """
    Returns Planck's blackbody function.
    
    Args:
        nu : frequency array in Hz
        Td : dust temperature 
        
    Returns:
        res : of shape (nu, z)
    """
    
    prefact = 2.*hp*nu**3/c_light**2
    x = np.outer(hp * nu, 1/(kB * Td)) #broadcast to shape (nu, z)
    res = prefact[:,np.newaxis]/(np.exp(x) - 1)
    
    return res

def mod_blackbody(Bnu, nu):
    
    """
    Returns gray-body function. 
    
    Args:
        Bnu : Planck function of shape (nu, z)
        nu : freq. array matching Bnu axis 0 binning of nu
    Returns:
        res : modified Planck func of shape (nu, z)
    """
    
    res = Bnu
    res = res * (nu**dbeta)[:,np.newaxis] # broadcast for proper shape
    
    return res

def nu0_z(Td):
    """
    Returns the pivot frequency as a function of redshift.
    
    For modified blackboy approximation, we have dlntheta/dlnnu = -gamma for nu=nu0.
    Here theta is the modified blackbody spectrum. In order to find nu0
    which isredshift dependent, we need to take a derivative and solve
    for this numerically. In the end it comes out in the form
    (x-(3+beta+gamma))e(x-(3+beta+gamma)) = -(3+beta+gamma)e(-(3+beta+gamma))
    The solution is x-(3+beta+gamma) = W(-(3+beta+gamma)e(-(3+beta+gamma)))
    here W is Lambert's W fnction which is implemented in scipy.
    x = hnu/KT
    
    Args:
        Td : dust temperature of shape (z,)
    """
    
    #FIXME: is this actually smoothly connecting?
    y = -(3 + dbeta + dgamma) * np.exp(-(3 + dbeta + dgamma))
    xx = lambertw(y)
    x = xx + (3 + dbeta + dgamma)
    nu0z = np.real(x * (kB * Td)/hp)
    return nu0z


def Theta(mod_Bnu, Bnu_at_nu0, nu, nu0):
    """
    Returns the modified SED for gray-body function normalized to 1 at pivot freq.
    
    Args:
        mod_Bnu : gray-body function 
        Bnu_at_nu0 : gray-body value at pivot freq. 
    """

    #theta(nu') = nu'**beta * Bnu(Td) nu < nu0
    #theta(nu') = nu'**(-gamma) nu>= nu0
    
    # calculating only for nu < nu0
    # normalised SED such that theta(nu0) = 1
    
    theta = mod_Bnu/Bnu_at_nu0 
    #theta[nu >= nu0] = 
    #FIXME: why not the nu >= nu0 SED? 
    return theta

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
    
    #djc/s returns of the shape (nu, Mh, z)
    djdlogmh = djc_dlogMh(nu) + djsub_dlogMh(nu) #FIXME
    dm = np.log10(Mh[1]/Mh[0])
    integrand = djdlogmh*hmfz
    
    # integrate along log mass axis 
    res = simpson(integrand, dx=dm, axis = 1)
    return res

def djc_dlogMh(nu, z, model, fsub = 0.134):
    """
    Returns the emissivity of central galaxies per log halo mass. 
    
    from A6 of 2204.05299
    djc_dlogMh (Mh, z) = chi^2 * (1+z) * SFRc/K * S^eff_nu (z)
    
    Args:
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
    jc = np.zeros(((len(nu_list), len(Mh), len(z)))) #FIXME correct variable names
    jc = prefact * SFRc(Mh * (1-fsub)) * Seff(nu, z, model)   
    
    #---- look at the code below to understand order of operation 
    # a = np.zeros((len(snu[:, 0]), len(self.mh), len(self.z)))
    # rest = self.sfr(self.mh*(1-fsub))*(1 + self.z) *\
    #     self.cosmo.comoving_distance(self.z).value**2/KC
    # # print (rest[50, 10:13])
    # for f in range(len(snu[:, 0])):
    #     a[f, :, :] = rest*snu[f, :]
    # return a #jc for us 
    
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
    prefact = prefact/KC
    
    sfrsub = SFRsub(params, model)
     
def SFRc(params, model):
    """
    Returns star formation rate of central galaxies as a function of halo parameters and model.
    
    Args:
        params : model parameters
        model : model name 
    """
    
    if model == 'S12':
        # SFRC_c (Mh, z) propto Sigma_c (M,z) Phi (z) 
        # from 2.34 of 2310.10848
        
        sigma_M0, mu_peak0, mu_peakp, delta = params
        
        phiCIB = phi_CIB(delta) #FIXME: what does this Phi represent?
        # Reshape phi(z) to broadcast across rows of Sigma
        phiCIB = phiCIB[np.newaxis, :]  # Make phi a row vector of shape (1, len(z))

        sigma_c = Sigmac(sigma_M0, mu_peak0, mu_peakp) * phiCIB 
        #sigma_s = Sigmas(sigma_M0, mu_peak0, mu_peakp) * phiCIB
        
    if model == 'M21':
        sfrmhdot = self.sfr_mhdot(mhalo)
        mhdot = Mdot(mhalo)

        res = Ob_to_Om
    #return mhdot * f_b * sfrmhdot
    return res 

def SFRsub(params, model):
    """
    Returns star formation rate of subhalos as a function of halo parameters and model.
    """
    
    if model == 'S12':
        # SFRC_s (Mh, z) propto Sigma_s (M,z) Phi (z) 
        # from 2.34 of 2310.10848
        
        sigma_M0, mu_peak0, mu_peakp, delta = params
        
        phiCIB = phi_CIB(delta) #FIXME: what does this Phi represent?
        # Reshape phi(z) to broadcast across rows of Sigma
        phiCIB = phiCIB[np.newaxis, :]  # Make phi a row vector of shape (1, len(z))

        sigma_sub = Sigmasub(sigma_M0, mu_peak0, mu_peakp) * phiCIB 
           
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

def Sigmac(sigma_M0, mu_peak0, mu_peakp):
    
    """
    Returns Luminosity-Mass relationship of central galaxies. 
    From 2.32 of 2310.10848.
    
    Sigma_c(M) = <N^IR_c (M)> Sigma(M).
    
    Args:
        sigma_M0 : halo mass range contributing to IR emissivity 
        mu_peak0 : peak of halo mass contributing to IR emissivity at z = 0
        mu_peakp : rate of change of halo mass contributing to IR emissity at higher z 
        
    Returns:
        res : of shape (Mh, z)
    """
    
    #FIXME: is mean_N_IR_c same as the galaxy HOD sample? is it a function of z or just M?
    res = mean_N_IR_c * Sigma(sigma_M0, mu_peak0, mu_peakp) 
    
    return res

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
    
    # based on 12 of 0909.1325.
    #FIXME: is this the state of the art? 
    def subhmf(m, M):
        """Vectorized f function that takes m and M arrays."""
        res = 0.30 * (m/M[:, np.newaxis])**(-0.7)
        expterm = -9.9 * (m/M[:, np.newaxis])**2.5
        return res * np.exp(expterm) ## FIXME: Abhi's subhmf code has additional * log(10)? 

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
    M = Mh[:, np.newaxis] # Make M a column vector of shape (len(Mh), 1)
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