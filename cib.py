"""
This module contains functions useful for CIB halo modelling.
"""

import consts
from scipy.integrate import simpson

Mh = consts.Mh
hmfz = consts.hmfz
dict_gal = consts.dict_gal['ELG']
KC = consts.KC

def jbar(nu):
    """
    Returns the mean emissivity of the CIB halos.
    
    Args:
        nu : measurement frequency
        Mh : halo mass
    """
    
    djdlogmh = djc_dlnMh() + djsub_dlnMh() #FIXME: define these functions
    dm = np.log10(Mh[1]/self.Mh[0])
    integrand = djdlogmh*hmfz
    
    res = simpson(integrand, dx=dm, axis=1, even='avg')
    return res

def djc_dlnMh(fsub = 0.134):
    fsub = 0.134 #FIXME: check that this is correct for ELGs
    """fraction of the mass of the halo that is in form of
    sub-halos. We have to take this into account while calculating the
    star formation rate of the central halos. It should be calculated by
    accounting for this fraction of the subhalo mass in the halo mass
    central halo mass in this case is (1-f_sub)*mh where mh is the total
    mass of the halo.
    for a given halo mass, f_sub is calculated by taking the first moment
    of the sub-halo mf and and integrating it over all the subhalo masses
    and dividing it by the total halo mass.
    
    Args:
    
    Returns:
        jc : matrix of shape (nu, Mh, z)
    """
    
    snu = self.snu #FIXME: where is this coming from?
    chi = dict_gal['chi']
    z = dict_gal['z']
    
    prefact = chi**2 * (1 + z)
    prefact = prefact/KC
    
    #FIXME: SFRc function will change as a function of models to be tested
    jc = prefact * SFRc(Mh, z) * S_nu_eff(z)   
    
    #---- look at the code below to understand order of operation 
    a = np.zeros((len(snu[:, 0]), len(self.mh), len(self.z)))
    rest = self.sfr(self.mh*(1-fsub))*(1 + self.z) *\
        self.cosmo.comoving_distance(self.z).value**2/KC
    # print (rest[50, 10:13])
    for f in range(len(snu[:, 0])):
        a[f, :, :] = rest*snu[f, :]
    return a #jc for us 

def djsub_dlnMh(fsub = 0.134): 
    """
    for subhalos, the SFR is calculated in two ways and the minimum of the
    two is assumed.
    """
    
    fsub = 0.134 #FIXME: take this out in the const file 
    # a = np.zeros((len(self.snu_eff[:, 0]), len(self.mh), len(self.z)))
    
    snu = self.snu #FIXME: where is this coming from?
    chi = dict_gal['chi']
    z = dict_gal['z']

    prefact = chi**2 * (1 + z)
    prefact = prefact/KC
    
    #FIXME: SFRc function will change as a function of models to be tested
    dlogmsub = #FIXME 
    integrand = dNdlogmsub(msub, Mh) * SFRsub(Mh, z)
    js = prefact * S_nu_eff(z) * simpson(integrand, dx = dlogmsub)
        
    
    #FIXME: understand the following     
        snu = self.snu
        a = np.zeros((len(snu[:, 0]), len(self.mh), len(self.z)))
        # sfrmh = self.sfr(mh)
        for i in range(len(self.mh)):
            ms = self.msub(self.mh[i]*(1-fsub))
            dlnmsub = np.log10(ms[1] / ms[0])
            sfrI = self.sfr(ms)  # dim(len(ms), len(z))
            sfrII = self.sfr(self.mh[i]*(1-fsub))*ms[:, None]/(self.mh[i]*(1-fsub))
            # sfrII = sfrmh[i] * ms / mh[i]
            sfrsub = np.zeros((len(ms), len(self.z)))
            for j in range(len(ms)):
                sfrsub[j, :] = np.minimum(sfrI[j, :], sfrII[j, :])
            integral = self.subhmf(self.mh[i], ms)[:, None]*sfrsub / KC
            intgn = intg.simps(integral, dx=dlnmsub, axis=0)
            a[:, i, :] = snu*(1 + self.z)*intgn *\
                self.cosmo.comoving_distance(self.z).value**2
        return a