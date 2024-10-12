"""
Script repurposed from Abhishek Maniyar's DopplerCIB github. 

Author: Tanveer Karim
Last Updated: 12 Oct 2024
"""

import numpy as np

# integrates using simpson method 
from scipy.integrate import simpson

# import local modules
import consts
import precalc as pc
import gal

# halo constants 
Mh = consts.Mh
hmfz = consts.hmfz
biasmz = pc.halo_biases

dict_gal = consts.dict_gal['ELG']
chi = dict_gal['chi']
dchi_dz = dict_gal['dchi_dz']

def pseudo_Cell(theta, M, B):
    """
    Returns pseudo-C_ell.
    
    Args:
        theta : model parameters 
        M : coupling matrix (obtained from Skylens)
        B : binning operator (obtained from Skylens)
    Returns:
        pcl_combined : single vector [gg, gcib_357, gcib_545, gcib_857]
    """
    
    # calculate unbinned pcl
    pcl_gcib = cibgalcross_cell_tot(theta) @ M
    pcl_gg = "ok" #FIXME 
    
    # bin pcl
    
    pcl_combined = np.concatenate(pcl_gg_binned, pcl_gcib_binned) #FIXME: make sure gcib order is correct
    return pcl_combined

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
    prefact = 1/chi**2 * dchi_dz # (z,)
    
    # w1w2 = W_CIB * W_gal
    w1w2 = window_cibxgal(b_gal) # (z,)

    # P^{g,CIB}_2h
    ucen = self.unfw #FIXME: need this from Rocher
    unfw = self.unfw #FIXME: need this from Rocher
    power = Pk_array(self.ell, self.z) #FIXME: in cosmo_related.py
    pk_2h = cibgalcross_pk_2h(ucen, unfw, power) #FIXME
    
    
    res = fact[:, None, :]*pk_2h #FIXME
    
    #FIXME: res seems to have shape ()??
    res = simpson(res, x=self.z, axis=2)
    return res

def get_W_gal(b_gal):
    """
    Returns galaxy radial kernel accounting
    for bias, dndz and magnification bias. 
    """
    
    pz = dict_gal['pz']
    
    gal_bias_term = b_gal * pz
    mag_bias_term = pc.mag_bias_gal
    
    w_gal = gal_bias_term + mag_bias_term
    
    return w_gal # (z,)
    
def window_cibxgal(b_gal):
    """
    Returns the multiplication of the radial window functions.
    """
    
    # W_gal = dz/dchi * pz
    w_gal = get_W_gal(b_gal)/dchi_dz #FIXME: dchi_dz should be IR galaxies 

    w_cib = pc.W_cib

    return w_gal * w_cib #shape (z,) #FIXME: match size through interpolation?
    
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
    
def cibterm(djc_dlogMh, djs_dlogMh, unfw): #FIXME: needs testing
    """
    Returns the first bracket in A13 of 2204.05299.
    This corresponds to the CIB term in calculating Pk. 
    
    Args:
        djc_dlogMh : Specific emissivity of central galaxies (nu, Mh, z)
        djs_dlogMh : Specific emissivity of sat galaxies (nu, Mh, z)
        unfw : Fourier transform the NFW profile inside the halo. (k, Mh, z)

    Returns:
        res : shape (k, nu, Mh, z)
    """
    
    #reshape to include k dimension to multiply with unfw
    djc_dlogMh = np.expand_dims(djc_dlogMh, axis = 0)
    djs_dlogMh = np.expand_dims(djs_dlogMh, axis = 0)
    
    # reshape to include nu dimension in unfw
    res = djs_dlogMh * np.expand_dims(unfw, axis = 1)
    res = res + djc_dlogMh
    
    return res


def cibgalcross_pk_2h(ucen, unfw, pk):
        """
        Check the calculation first using two different integrals for both CIB
        and galaxies, then check with a single integral for both after
        multiplying. If both are same, then we can save one integral.
        
        Pk_2h (nu, k, z) = b_g * b_CIB * P_lin/(nbar * jbar)
        
        b_g = int HMF(Mh, z) * b(Mh, z) * gal_term (k, Mh, z) dlogMh
        b_CIB = int HMF(Mh, z) * b(Mh, z) * cib_term (nu, k, Mh, z) dlogMh
        """
        # pk = self.uni.Pk_array(self.ell, self.z)
        # print (self.hmfmz.shape, self.biasmz.shape)
        
        # galaxy bias term: int HMF(Mh, z) * b(Mh, z) * gal_term (k, Mh, z) dlogMh
        galterm = gal.galterm_Pk(params, rho_crit, rad, dlnpk_dlnk, gal_type = 'ELG')
        integrand = hmfz * biasmz * galterm # FIXME: pass galterm properly
        integral_g = simpson(y=integrand, x=Mh) #FIXME: define axis properly 
        
        # CIB bias term: int HMF(Mh, z) * b(Mh, z) * cib_term (nu, k, Mh, z) dlogMh
        integrand = hmfz * biasmz * cibterm #FIXME: pass cibterm properly
        integral_cib = simpson(y=integrand, x=Mh)
        
        res1 = hmfz*self.biasmz*self.cibterm(unfw)  # snu_eff, unfw)
        dm = np.log10(self.mh[1] / self.mh[0])
        intg_mh1 = intg.simps(res1, dx=dm, axis=2, even='avg')
        
        res2 = self.hmfmz*self.biasmz*self.galterm(ucen, unfw)  # ucen, unfw)
        intg_mh2 = intg.simps(res2, dx=dm, axis=1, even='avg')
        res3 = intg_mh1*intg_mh2*pk
        return res3/self.nbargal()/self.jbar()[:, None, :]
     

    
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