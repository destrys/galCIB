"""
Script repurposed from Abhishek Maniyar's DopplerCIB github. 

Author: Tanveer Karim
Last Updated: 4 September 2024
"""

from scipy.integrate import simpson
#from astropy import constants as const
import consts

z = consts.zrange.values # redshift bin of galaxies 
pz = consts.pz.values # dndz values 

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


def W_cib():
    """
    Returns redshift kernel of CIB field
    """
    outp = 1/(1 + z)
    return j_nuprime_z(nu, z, dM) * outp
    
def W_gal(b_gal, pz = pz, z = z, 
          mag_bias_alpha = 2.225):
    """
    Returns redshift kernel of galaxy field
    """
    
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
    
    
def cibXgal_cell_2h(W_cib, W_gal, P_cibXgal):
    
    """
    Returns C_ell of 2-halo terms 
    """
    
    dchi = z # integrating over dchi 
    integrand = 1/consts.chi**2
    integrand *= W_cib * W_gal 
    integrand *= P_cibXgal
    
    C_ell_2h = simpson(y = integrand, x = dchi)
    
    return C_ell_2h
       

# def cibXgal_pk_2h(Pmm):
#     """
#     Returns the theory cross Pk of CIB and galaxy
    
#     Arguments
#     ---------
#         Pmm : linear matter power spectrum
#     """
    
#     bias_gal
#     bias_CIB
    
#     P_cibXgal = bias_gal * bias_CIB * Pmm 

# def cibXgal_cell_2h():
#     """"
#     Returns the theory cross C_ell of CIB and galaxy
#     """
    
#     ucen = 
#     unfw = 
#     power = 
#     w1w2 = 
#     geo = 
#     fact = 
#     rest = 
    
    