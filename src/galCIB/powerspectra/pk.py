"""
Contains 1-halo and 2-halo term P(k) functions.
"""

import numpy as np
from scipy.integrate import simpson
from .utils import ensure_nm_nz_shape, compute_Pgg_1h, compute_Pgg_2h, compute_PgI_1h, compute_PgI_2h, compute_PII_1h, compute_PII_2h, compute_Puv_tot

class PkBuilder:
    def __init__(self, cosmology, hod_model, cib_model, prof_model,
                 theta_cen=None, theta_sat=None, 
                 theta_prof=None,
                 theta_sfr=None, theta_snu=None,
                 theta_IR_hod=None):
        self.cosmo = cosmology
        self.hod = hod_model
        self.cib = cib_model
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
        self.theta_sfr = theta_sfr
        self.theta_snu = theta_snu
        self.theta_IR_hod = theta_IR_hod

    def _cache_galaxy_integral(self):
        """
        Cache [Ncen(M,z) + Nsat(M,z) * u(k,M,z)]
        Galaxy term from A11 of 2204.05299
        """
        
        u=self.u
        
        self.ncen = ensure_nm_nz_shape(self.hod.ncen(self.theta_cen),
                                       len(self.cosmo.Mh),
                                       len(self.cosmo.z))  # shape (Nm, Nz)
        self.nsat = ensure_nm_nz_shape(self.hod.nsat(self.theta_sat),
                                       len(self.cosmo.Mh),
                                       len(self.cosmo.z))  # shape (Nm, Nz)
        
        self.nsat_u = self.nsat*u # useful in multiple places in Pk
        self.ncen_plus_nsat_u = self.ncen + self.nsat_u 
        
        # pre-compute 2h mass integral term for speedup
        integrand = self.hmfxbias * self.ncen_plus_nsat_u # (Nk, NMh, Nz)
        self.Ig = simpson(integrand, dx=self.dlog10Mh, axis=1) # (Nk, Nz)
        
        
    def _cache_cib_integral(self):
        """
        Cache [djc(nu,M,z) + djs(nu,M,z) * u(k,M,z)]
        CIB term from A9 of 2204.05299
        """
        
        u=self.u
        
        # update CIB model
        self.cib.update(self.theta_sfr, 
                        self.theta_snu,
                        self.theta_IR_hod)
        
        self.djc = self.cib.get_djc()[:,None,:,:] # (Nnu, 1, NMh, Nz)
        self.djsub = self.cib.get_djsub()[:,None,:,:]
        
        self.djsub_u = self.djsub * u[None,:,:,:] # useful in multiple places in Pk
        self.djc_plus_djsub_u = self.djc + self.djsub_u 
        
        # pre-compute 2h mass integral term for speedup
        integrand = self.djc_plus_djsub_u* self.hmfxbias # (Nnu, Nk, NMh, Nz)
        self.Icib = simpson(integrand, dx=self.dlog10Mh, axis=2) # (Nnu, Nk, Nz)
        
    def _cache_u_profile(self):
        """
        Cache new profile
        """
        self.prof_model.update_theta(self.theta_prof)
        
    def _compute_nbar(self):
        # A12 of 2204.05299

        self.nbar = simpson(self.hmf * (self.ncen + self.nsat), 
                       dx=self.dlog10Mh,
                       axis=0)
        self.nbar2 = self.nbar**2 # useful for multiple Pk 
            
    def _update_theta(self, theta_cen, theta_sat, theta_prof, 
                      theta_sfr, theta_snu, theta_IR_hod):
        """
        Update model parameters and recompute the Pk.
        """
        
        # new theta 
        self.theta_cen = theta_cen
        self.theta_sat = theta_sat
        self.theta_prof = theta_prof
        self.theta_sfr = theta_sfr 
        self.theta_snu = theta_snu 
        self.theta_IR_hod = theta_IR_hod 
        
        # update cache 
        self._cache_u_profile()
        self._cache_galaxy_integral() 
        self._cache_cib_integral()
        
        # nbar 
        self._compute_nbar()
        
    def compute_pk(self, theta_cen=None, theta_sat=None, theta_prof=None,
                   theta_sfr=None, theta_snu=None, theta_IR_hod=None,
                   hmalpha=1, return_full_matrix_II=False):
        """
        Return Pgg, PII, PgI.
        
        Args:
            theta_cen, theta_sat = galaxy parameters
            theta_prof = radial profile parameters
            theta_sfr = SFR parameters
            theta_snu = Snu parameters 
            theta_IR_hod = IR galaxy Ncen parameters 
            hmalpha = 1h to 2h transition relaxation parameter 
            return_full_matrix_II = Whether to return full nu x nu' matrix or only unique one (upper-triangle)
            
        """
        
        # update relevant cached values. 
        self._update_theta(theta_cen, theta_sat, theta_prof,
                           theta_sfr, theta_snu, theta_IR_hod)
        
        pk_gg_2h = compute_Pgg_2h(self)
        pk_gg_1h = compute_Pgg_1h(self)
        
        if return_full_matrix_II is False: 
            pk_II_2h, self.twoh_pairs = compute_PII_2h(self, return_full_matrix=return_full_matrix_II)
            pk_II_1h, self.oneh_pairs = compute_PII_1h(self, return_full_matrix=return_full_matrix_II)
        else:
            pk_II_2h = compute_PII_2h(self, return_full_matrix=return_full_matrix_II)
            pk_II_1h = compute_PII_1h(self, return_full_matrix=return_full_matrix_II)
        
        pk_gI_2h = compute_PgI_2h(self)
        pk_gI_1h = compute_PgI_1h(self)
        
        pk_gg_tot = compute_Puv_tot(pk_gg_1h, pk_gg_2h, hmalpha)
        pk_II_tot = compute_Puv_tot(pk_II_1h, pk_II_2h, hmalpha)
        pk_gI_tot = compute_Puv_tot(pk_gI_1h, pk_gI_2h, hmalpha)
        
        return pk_gg_tot, pk_II_tot, pk_gI_tot 