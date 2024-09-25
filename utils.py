"""
Script repurposed from Abhishek Maniyar's DopplerCIB github. 

Author: Tanveer Karim
Last Updated: 4 September 2024
"""

from scipy.integrate import simpson
#from astropy import constants as const
import consts

OmegaM = consts.OmegaM
c = consts.speed_of_light
H0 = consts.H0
dict_ELG = consts.dict_gal['ELG']
Plin = consts.Plin


def nfwfourier_u(self):
        rs = self.r_star()
        c = self.nu_to_c200c()
        a = self.ampl_nfw(c)
        mu = np.outer(self.kk, rs)
        Si1, Ci1 = self.sine_cosine_int(mu + mu * c)
        Si2, Ci2 = self.sine_cosine_int(mu)
        unfw = a*(cos(mu)*(Ci1-Ci2) + sin(mu)*(Si1-Si2)-sin(mu*c) / (mu+mu*c))
        return unfw.transpose()  # dim(len(m), len(k))

def B(nu, T): #DONE
    """
    Returns Plancks blackbody function as a function
    of freq nu and temperature T.
    """
    
    const_term = 2 * planck_const_h/speed_of_light**2
    exp_term = 1/(np.exp((planck_const_h * nu)/(boltzmann_kb * T))-1)
    return const_term * nu**3 * exp_term
    
def Theta(nu, z, beta, Td, gamma):
    """
    Returns the SED model of CIB-emitting haloes
    
    Arguments:
        nu : frequency
        z : redshift
        beta : emissivity index, related to physical nature of dust
        Td : effective dust temperature
        gamma : power index 
    """
    
    res = np.zeros_like(nu)
    flag = (nu < nu0) # freq below pivot freq nu0
     
    res[flag] = nu**beta * B(nu, Td)
    res[~flag] = nu**(-1 * gamma)
    
    return res 

def Sigma(M, sigma_LM, Meff):
    """
    Returns the Luminosity-Mass relationship 
    """
    
    first_term = M/np.sqrt(2 * np.pi * sigma_LM**2)
    exp_term = (np.log10(M) - np.log10(Meff))/sigma_LM ##FIXME: 2023 paper and Serra 2014 differ slightly
    
    return first_term * np.exp(-0.5 * exp_term**2)
    

def shang12(z, nu, L0, M, sigma_LM, Meff, beta, Td, gamma):
    """
    Returns the specific IR luminosity at frequency nu
    using the Shang et al. 2012 (1109.1522) model.
    """
    
    Sigma_M = Sigma(M, sigma_LM, Meff)
    Theta_nu = Theta ((1+z) * nu, beta, Td, gamma)
    res = L0 * (1 + z)**(3.6) * Sigma_M * Theta_nu
    return res

def L_nuprime_z(M, z, model_name):
    """"
    Returns the specific IR luminosity at frequency nu.
    """
    
    # call specific model and return value 
    if model_name == 'S12':
        return shang12(z, nu, L0, M, 
                       sigma_LM, Meff, beta, 
                       Td, gamma)
    return 0 #FIXME
    

def j_nuprime_z(nu, z, dM):
    """
    Returns the j_(1+z)nu (z), comoving emissivity at a
    given frequency nu and redshift z.
    
    Arguments:
        dM : spacing of the halo mass function
    
    Returns:
        res : comoving emissivity value
    """
    
    #integrand = dndM * L_nuprime_z(M, z)/4*np.pi
    # res = simpson(y = integrand, x = dM)
    #return res
    return 0 #FIXME


def W_cib(): #FIXME
    """
    Returns redshift kernel of CIB field
    """
    outp = 1/(1 + z)
    return j_nuprime_z(nu, z, dM) * outp

def W_gal(b_gal, dict_gal = dict_ELG): #DONE
    """
    Returns redshift kernel of galaxy field
    """
    
    z = dict_gal['z']
    pz = dict_gal['pz']
    chi = dict_gal['chi']
    Hz = dict_gal['Hz']
    mag_bias_alpha = dict_gal['mag_bias_alpha']
    
    gal_bias_term = b_gal * pz
    
    mag_bias_prefact = 3 * OmegaM/(2 * c)
    mag_bias_prefact = (mag_bias_prefact * H0**2/Hz * (1 +z) * chi).decompose() # to reduce to the same unit
    
    integrated_values = np.zeros_like(z)
    for i in range(len(z)): # loop over to get integrand values 
      zspecific_indx = i
      
      # only consider bins above the specific index 
      flag = (z >= z[zspecific_indx])
      ratio = chi[zspecific_indx]/chi[flag]
      
      # assuming constant alpha over z 
      integrand_term = (1 - ratio) * (mag_bias_alpha - 1) * pz[flag]
      integrated_values[i] = simpson(y = integrand_term, x = z[flag])
  
    mag_bias_term = mag_bias_prefact * integrated_values
    return gal_bias_term + mag_bias_term
    
    
def cibterm(djc_dlogMh, djs_dlogMh, unfw): #FIXME: needs testing
    """
    Returns the first bracket in A13 of 2204.05299.
    This corresponds to the CIB term in calculating Pk. 
    
    Args:
        djc_dlogMh : Specific emissivity of central galaxies (Mh, z)
        djs_dlogMh : Specific emissivity of sat galaxies (Mh, z)
        unfw : Fourier transform the NFW profile inside the halo. 
               Shape is num of modes X num of Halo mass func bins X num of redshifts

    Returns:
        res : shape (k, Mh, z)
    """
    
    #reshape to include k dimension to multiply with unfw
    djs_dlogMh = djs_dlogMh[np.newaxis, :, :]
    djc_dlogMh = djc_dlogMh[np.newaxis, :, :]
    
    res = djs_dlogMh * unfw
    res = res + djc_dlogMh
    
    return res
     

def P_cibXgal_1h():
    """
    Returns the 1-halo power spectrum 
    """
    
    integrand = HMFz * cibterm * galterm #HMFz = halo mass func (z)
    res = simpson(integrand, x = np.log10(Mh))
    res = res/(nbar_gal * jbar)
    
    return res 

def P_cibXgal_2h(): #func of k,z,nu?, 
    prefact = Plin/(nbar_gal() * jbar(nu)) #FIXME: what is nbar_gal and j_nu? 
    
    
def cibXgal_cell_2h(W_cib, W_gal, P_cibXgal, dict_gal = dict_ELG):
    
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