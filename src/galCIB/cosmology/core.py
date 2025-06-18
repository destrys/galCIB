"""
This module defines the fixed cosmology used in the analysis.
"""


import numpy as np

from colossus.cosmology import cosmology as cc
from colossus.lss import mass_function


class Cosmology:
    def __init__(self, zs, ks, Mh, 
                 colossus_cosmo_name = 'planck18',
                 use_little_h = False):
        
        """
        Returns Cosmology Class with user-defined choices.
        
        Args:
            zs : redshift range
            ks : k range 
            Mh : Halo mass range
            colossus_cosmo_name : str referring to built-in colossus cosmology
            use_little_h : bool, True if using little h convention
        """
        
        self.cosmo_name = colossus_cosmo_name
        self.z = zs
        self.k = ks
        self.Mh = Mh
        self.use_little_h = use_little_h
        
        self.pk_grid, self.hmf_grid, self.cosmo = self._load_colossus()
        
        # initialize frequently used quantities
        self._cache_quantities()
        
    
    def _load_colossus(self):
        """
        Returns P_mm (k), Halo Mass Function and Halo Bias
        from colossus. 
        """
        
        cosmo = cc.setCosmology(self.cosmo_name)
        
        pk_grid = np.zeros((len(self.k),len(self.z)))
        hmf_grid = np.zeros((len(self.Mh),len(self.z)))
        
        if self.use_little_h is False:
            k_range = self.k * cosmo.h
            Mh_range = self.Mh/cosmo.h
        else: 
            k_range = self.k
            Mh_range = self.Mh
        
        for i in range(len(self.z)):
            tmp_pk = cosmo.matterPowerSpectrum(k_range, 
                                               self.z[i],
                                               model = 'camb')
            
            # Convert dn/dlnM to dn/dlog_10M, using ln(10) factor. 
            tmp_hmf = mass_function.massFunction(x=Mh_range, z=self.z[i],
                                                     mdef='200m', model='tinker08',
                                                     q_in='M', q_out='dndlnM') * np.log(10)
            
            # whether to convert HMF into h-less unit
            if self.use_little_h is False:
                pk_grid[:,i] = tmp_pk/cosmo.h**3 
                hmf_grid[:,i] = tmp_hmf*cosmo.h**3
            else:
                pk_grid[:,i] = tmp_pk
                hmf_grid[:,i] = tmp_hmf
                
        return pk_grid, hmf_grid, cosmo
    
    def _cache_quantities(self):
        """
        Caches most frequently used quantities downstream. 
        
        Calculates the following:
        1. comoving distance, chi(z)
        2. H(z)
        3. Omegab/OmegaM (z)
        4. rho_crit (z)
        5. mean density at z = 0
        6. H(z)/(c * chi^2(z))
        7. chi^2(z)
        8. dchi/dz (z)
        
        """
        
        # comoving distance in units of Mpc
        self.chi = self._calculate_chi()
            
        # H(z) 
        self.Hz = self.cosmo.Hz(self.z) # units of km/s/Mpc
        
        # H0 
        self.H0 = self.cosmo.H0 # units of km/s/Mpc
        
        # Omega_M(z = 0)
        self.Om0 = self.cosmo.Om0 
        
        # speed of light in km/s 
        
        self.c = 299792.458  # km/s
         
    def _calculate_chi(self):
        
        chi = np.zeros_like(self.z)
        for i in range(len(self.z)):
            chi[i] = self.cosmo.comovingDistance(0,
                                                 self.z[i])
            
        if self.use_little_h is False:
            chi = chi/self.cosmo.h 
            
        return chi 
             
    def get_pk_grid(self):
        """
        Returns the precomputed P(k,z) grid.
        """   
        
        return self.pk_grid
    
    def get_hmf_grid(self):
        """
        Returns precomputed HMF(z,Mh)
        """
        
        return self.hmf_grid
    
    def get_k_range(self, use_little_h):
        
        if use_little_h is False:
            res = self.k * self.cosmo.h
        else:
            res = self.k
            
        return res 
    
    def get_Mh_range(self, use_little_h):
        
        if use_little_h is False:
            res = self.Mh/self.cosmo.h
        else:
            res = self.Mh 
            
        return res
    
    def save_grids(self, filename):
        np.savez(filename, 
                    pk=self.pk_grid, 
                    z=self.z, k=self.k,
                    hmf=self.hmf_grid,
                    Mh=self.Mh)
        
    @classmethod
    def load_grids(cls, filename):
        data = np.load(filename)
        instance = cls.__new__(cls)  # create instance without __init__
        instance.pk_grid = data['pk']
        instance.z = data['z']
        instance.k = data['k']
        instance.hmf = data['hmf']
        instance.Mh = data['Mh']
        
        return instance