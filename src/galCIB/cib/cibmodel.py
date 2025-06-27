"""
This sets up the CIB model class that can take user-defined SFR and Snu
models. It has the M21 and Y23 models implemented as defaults. 
"""

#simport inspect 
from .utils import compute_BAR_grid
from .registry import get_sfr_model, _lazy_register_defaults

class CIBModel:
    """
    Container class for a CIB emissivity model.
    
    Attributes
    ----------
    name : str
        Unique identifier for the model.
    sfr_fn : callable
        Function that returns SFR(M, z) given M, z, and theta.
    snu_fn : callable
        Function that returns effective SED given M, z, and theta.
    cosmo : object 
        Providing cosmology-dependent quantities
    """
    
    def __init__(self, sfr_fn=None, snu_fn=None,
                 hod=None, cosmo=None):
        
        self.hod = hod      # galaxy HODModel for SFR_C 
        self.cosmo = cosmo
        self.z_ratio = self.hod.z_ratio # z/(1+z)
        
        # Precompute geometric prefactor, A6 and A7 of 2204.05299
        self.geom_prefact = cosmo.chi**2 * (1 + cosmo.z)
        
        # Precompute geometric prefactor/KC to speed up calculation
        KC = 1.0e-10  # Kennicutt constant for Chabrier IMF in units of Msol * yr^-1 * Lsol^-1
        self.geom_prefact_over_KC = self.geom_prefact/KC
        
        # Load default SFR model if none provided
        if sfr_fn is None:
            # Precompute BAR grid 
            self.BAR_grid = compute_BAR_grid(self.cosmo)
            
            _lazy_register_defaults()  # Ensures registry is populated
            sfr_factory = get_sfr_model("M21")
            self._sfr_fn = sfr_factory(self.BAR_grid, self.z_ratio)
        else:
            self._sfr_fn = sfr_fn
        
        

    def compute_sfr(self, theta_sfr):
        return self._sfr_fn(self.cosmo.Mh_grid[0],
                            self.cosmo.z,
                            theta_sfr) # Pass Mh_grid[0] because of shape (Nm,Nz)
    
    def snu(self, theta_snu):
        return self._snu_fn(theta_snu)

    # def djc(self, Mh, theta_sfr, theta_snu, theta_hod):
        
    #     """
    #     Returns the emissivity of central galaxies per log halo mass. 
        
    #     from A6 of 2204.05299
    #     djc_dlogMh (Mh, z) = chi^2 * (1+z) * SFRc/K * S^eff_nu (z)
        
    #     Args:
    #         theta_sfr : SFR parameters
    #         theta_snu : Snu parameters
    #     Returns:
    #         jc : matrix of shape (nu, Mh, z)
    #     """
        
    #     # fraction of the mass of the halo that is in form of
    #     # sub-halos. We have to take this into account while calculating the
    #     # star formation rate of the central halos. It should be calculated by
    #     # accounting for this fraction of the subhalo mass in the halo mass
    #     # central halo mass in this case is (1-f_sub)*mh where mh is the total
    #     # mass of the halo.
    #     # for a given halo mass, f_sub is calculated by taking the first moment
    #     # of the sub-halo mf and and integrating it over all the subhalo masses
    #     # and dividing it by the total halo mass.
            
    #     prefact = self.geom_prefact_over_KC
        
    #     #sfr = self.sfr_fn(Mh, theta_sfr, z_ratio=self.z_ratio)  # M_sol / year
    #     ncen = self.hod_model.ncen(self.cosmo.log10Mh, 
    #                                self.theta_hod, self.cosmo.z)
    #     sfr_c = ncen * sfr 
    #     snu = self.snu_fn(theta_snu)    # unitless
        
    #     central_emissivity = prefact * sfr_c * snu        
        
    #     return central_emissivity

    # def emissivity_satellite(self, M, z, theta_sfr, theta_snu):
    #     # analogous calculation for satellites
    #     pass
