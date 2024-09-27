"""
Script repurposed from Abhishek Maniyar's DopplerCIB github. 

Author: Tanveer Karim
Last Updated: 27 September 2024
"""

# integrates using simpson method 
from scipy.integrate import simpson

# import local modules
import consts
import precalculation as pc

dict_gal = consts.dict_gal['ELG']
chi = dict_gal['chi']
dchi_dz = dict_gal['dchi_dz']

def cibgalcross_cell_tot(theta): 
    """
    Returns C_{g, CIB} accounting for all halo terms
    """
    
    #FIXME: do we need to split up 1h and 2h theta params
    oneh = cibgalcross_cell_1h(theta) #FIXME
    twoh = cibgalcross_cell_2h(theta) #FIXME
    shot = cibgalcross_cell_shot(theta) #FIXME
    
    tot = oneh+twoh+shot
    return tot

def cibgalcross_cell_1h():
    """
    Returns C_{g, CIB} 1-halo term. 
    """  
    
    ucen = self.unfw
    unfw = self.unfw
    w1w2 = self.window_cibxgal()
    # print (ucen[50, 10, 10:14])
    # print (unfw[50, 10, 10:14])
    # print (w1w2.shape)
    # print (w1w2[4, 10:14])
    geo = self.uni.dchi_dz(self.z)/self.uni.chi(self.z)**2
    fact = geo*w1w2
    res = fact[:, None, :]*self.cibgalcross_pk_1h(ucen, unfw)
    return intg.simps(res, x=self.z, axis=2, even='avg')

def cibgalcross_cell_2h(theta):
    """
    Returns C_{g, CIB} 2-halo term. 
    
    # Based on A15 from 2204.05299
    C_{g,CIB} = integral [1/chi^2 * dchi/dz * W_CIB (nu, z) * W_gal (z) * P^{g,CIB}_2h (k = l/chi, z)] dz
    
    Returns:
        res: shape (nu, z) FIXME??
    """  

    # prefact = 1/chi^2 * dchi/dz
    prefact = 1/chi**2 * dchi_dz
    
    # w1w2 = W_CIB * W_gal
    w1w2 = window_cibxgal() #FIXME

    # P^{g,CIB}_2h
    ucen = self.unfw #FIXME: need this from Rocher
    unfw = self.unfw #FIXME: need this from Rocher
    power = Pk_array(self.ell, self.z) #FIXME: in cosmo_related.py
    pk_2h = cibgalcross_pk_2h(ucen, unfw, power) #FIXME
    
    
    res = fact[:, None, :]*pk_2h #FIXME
    
    #FIXME: res seems to have shape ()??
    res = simpson(res, x=self.z, axis=2)
    return res

def window_cibxgal():
    """
    Returns the multiplication of the radial window functions.
    """
    
    # W_gal = dz/dchi * pz
    w_gal = pc.radial_window_gal/dchi_dz
    
    w_cib = pc.radial_window_cib

    return w_cib*w_gal #shape (z,)
    


def cibgalcross_cell_shot(theta):
    #FIXME: need nu0, ell, 
    
    nfreq = len(self.nu0)
    nl = len(self.ell)
    shotcib = np.zeros((nfreq, nl))
    # 100, 143, 217, 353, 545, 857 Planck
    # values for 100 and 143 i.e. 3 and 7 are fake
    sar = self.dv.shotpl
    # """
    freq = np.array([100., 143., 217., 353., 545., 857.])
    sa = np.array([1.3*0.116689509208305475, 1.3*0.8714424869942087, 14., 357., 2349., 7407.])
    sa[2:] = sar
    res = interp1d(freq, sa, kind='linear', bounds_error=False, fill_value="extrapolate")
    shotval = res(self.nu0)
    """
    freq = self.nu0
    shotval = sar
    # """
    for i in range(nfreq):
        # shot[i, i, :] = sa[i]
        shotcib[i, :] = shotval[i]
        # print (self.nu0[i], shotval[i])
    if max(self.nu0) > max(freq):
        print ("shot noise values for frequencies higher than 857 GHz extrapolated using the values for Planck")

    r_l = self.r_l
    # shotcib = self.shot_cib()
    shotgal = np.repeat(self.clshot_gal(), len(self.ell))
    crossshot = r_l*np.sqrt(shotcib*shotgal)
    return crossshot


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