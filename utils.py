"""
Script repurposed from Abhishek Maniyar's DopplerCIB github. 

Author: Tanveer Karim
Last Updated: 4 September 2024
"""

from scipy.integrate import simpson

def B(nu, T):
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

def shang12(L0, z):
    """
    Returns the specific IR luminosity at frequency nu
    using the Shang et al. 2012 (1109.1522) model.
    """
    
    res = L0 * (1 + z)**(3.6) * Sigma(M) * Theta ((1+z) * nu)
    return res
    return 0 #FIXME

def L_nuprime_z(M, z, model_name):
    """"
    Returns the specific IR luminosity at frequency nu.
    """
    
    # call specific model and return value 
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


def cib_halo():
    
    

def cibXgal_pk_2h(Pmm):
    """
    Returns the theory cross Pk of CIB and galaxy
    
    Arguments
    ---------
        Pmm : linear matter power spectrum
    """
    
    bias_gal
    bias_CIB
    
    P_cibXgal = bias_gal * bias_CIB * Pmm 

def cibXgal_cell_2h():
    """"
    Returns the theory cross C_ell of CIB and galaxy
    """
    
    ucen = 
    unfw = 
    power = 
    w1w2 = 
    geo = 
    fact = 
    rest = 
    
    