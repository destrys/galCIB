"""
Contains 1-halo and 2-halo term P(k) functions.
"""

import numpy as np
from scipy.integrate import simpson
from .utils import ensure_nm_nz_shape
class PkBuilder:
    def __init__(self, cosmology, hod_model, prof_model,
                 theta_cen=None, theta_sat=None, theta_prof=None):
        self.cosmo = cosmology
        self.hod = hod_model
        self.prof_model = prof_model
        self.u = prof_model.u
        self.k = cosmology.k
        self.z = cosmology.z
        self.log10Mh = cosmology.log10Mh
        self.dlog10Mh = self.log10Mh[1] - self.log10Mh[0] # equal spacing
        #FIXME: let user pass unequal spacing as an option
        self.hmf = cosmology.hmf_grid
        self.hmfxbias = self.hmf * prof_model.hbias
        
        self.theta_cen = theta_cen
        self.theta_sat = theta_sat
        self.theta_prof = theta_prof
        
        # Cache galaxy integral term
        self._cache_galaxy_integral()
        
        # Calculate Pgg (2h)
        self.compute_Pgg_2h()
        
        # Calculate Pgg (1h)
        self.compute_Pgg_1h()

    def _cache_integrals(self):
        """
        Cache I_gal(k,z) = ∫ dlogM HMF(M,z) * bias(M,z) * [Ncen(M,z) + Nsat(M,z) * u(k,M,z)]
        Galaxy term from A11 of 2204.05299
        
        Cache I_CIB(nu,k,z) = ∫ dlogM HMF(M,z) * bias(M,z) * [djc(nu,M,z) + djs(nu,M,z) * u(k,M,z)]
        CIB term from A9 of 2204.05299
        """
        
        self.prof_model.update_theta(self.theta_prof)
        u=self.prof_model.u
        
        ##--Galaxy--##
        self.ncen = ensure_nm_nz_shape(self.hod.ncen(self.theta_cen),
                                       len(self.cosmo.Mh),
                                       len(self.cosmo.z))  # shape (Nm, Nz)
        self.nsat = ensure_nm_nz_shape(self.hod.nsat(self.theta_sat),
                                       len(self.cosmo.Mh),
                                       len(self.cosmo.z))  # shape (Nm, Nz)
        
        integrand = self.hmfxbias * (self.ncen + self.nsat * u)  # shape (Nk, Nm, Nz)
        Ig = simpson(integrand,dx=self.dlog10Mh,axis=1)
        
        self.nbar = self._compute_nbar()
        self.Igal = Ig/self.nbar
        
        ##--CIB--##
        self.djc = self.cib.djc(self.theta_sfr,
                                self.theta_snu,
                                self.theta_hod)
        # self.djs = self.cib.djs(self.theta_sfr,
        #                         self.theta_snu,
        #                         self.theta_hod)
        
        integrand = self.hmfxbias * (self.djc + self.nsat * u) # shape (Nnu,Nk,Nm,Nz)
        self.Icib = simpson(integrand,dx=self.dlog10Mh,axis=-2)
        
    def _compute_nbar(self):
        # A12 of 2204.05299

        nbar = simpson(self.hmf * (self.ncen + self.nsat), 
                       dx=self.dlog10Mh,
                       axis=0)
        
        return nbar
    
    def compute_Pgg_2h(self):
        """
        Compute P_gg^2h(k,z) = P_lin(k,z) * I_gal^2
        """
        
        self.Pgg_2h = self.cosmo.pk_grid * self.Igal**2
        
    def compute_Pgg_1h(self):
        """
        Returns 1-halo term 
        """
        
        prefact = 2/self.nbar**2
        numerator = self.ncen*self.nsat*self.u + (self.nsat*self.u)**2
        
        integrand = self.hmf * numerator
        res = simpson(integrand,axis=1,dx=self.dlog10Mh)
        
        self.Pgg_1h = res * prefact 
        
    def compute_Pgg(self, hmalpha):
        self.Pgg_tot = (self.Pgg_1h**hmalpha + self.Pgg_2h**hmalpha)**1/hmalpha
        
        
    def _cache_CIB_integral(self):
        """
        
        """

        self.prof_model.update_theta(self.theta_prof)
        u=self.prof_model.u
        
        #print(f'shape = {self.hod.ncen(self.theta_cen).shape}')
        
        self.ncen = ensure_nm_nz_shape(self.hod.ncen(self.theta_cen),
                                        len(self.cosmo.Mh),
                                        len(self.cosmo.z))  # shape (Nm, Nz)
        self.nsat = ensure_nm_nz_shape(self.hod.nsat(self.theta_sat),
                                        len(self.cosmo.Mh),
                                        len(self.cosmo.z))  # shape (Nm, Nz)
        
        integrand = self.hmfxbias * (self.ncen + self.nsat * u)  # shape (Nm, Nz)
        Ig = simpson(integrand,dx=self.dlog10Mh,axis=1)
        
        self.nbar = self._compute_nbar()
        self.Igal = Ig/self.nbar
    
    def compute_PII_2h(self):
        """
        Compute P_(CIB-CIB)^2h (nu,k,z) = Plin(k,z) * I_CIB^2
        
        I_CIB = 
        """
        
    def update_theta(self, theta_cen, theta_sat, theta_prof):
        """
        Update model parameters and recompute the Pk.
        """
        self.theta_cen = theta_cen
        self.theta_sat = theta_sat
        self.theta_prof = theta_prof
        
        self._cache_galaxy_integral()
        self.Pgg_2h = self.compute_Pgg_2h()
