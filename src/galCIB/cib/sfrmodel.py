#cib/sfrmodel.py

from .registry import get_sfr_model
from .utils import _compute_BAR_grid

class SFRModel:
    def __init__(self, name: str, hod, cosmo, fsub=None):
        self.z_ratio = hod.z_ratio
        self.cosmo = cosmo
        self.fsub = fsub 
        
        if self.fsub is not None: 
            """fraction of the mass of the halo that is in form of
            sub-halos. We have to take this into account while calculating the
            star formation rate of the central halos. It should be calulated by
            accounting for this fraction of the subhalo mass in the halo mass
            central halo mass in this case is (1-f_sub)*mh where mh is the total
            mass of the halo.
            for a given halo mass, f_sub is calculated by taking the first moment
            of the sub-halo mf and and integrating it over all the subhalo masses
            and dividing it by the total halo mass.
            """
            
            self.Mh = cosmo.Mh * (1-fsub)
        else:
            self.Mh = cosmo.Mh
            self.fsub = 0
            
        self.BAR_grid = _compute_BAR_grid(cosmo=self.cosmo,
                                          Mh=self.Mh)[0] # (1, NMh, Nz)
        
        self.model_factory = get_sfr_model(name)
        self.model_fn = self.model_factory(self.BAR_grid, self.z_ratio)
        
    def __call__(self, theta_sfr):
        """
        Standard call: computes SFR(Mh, z) using precomputed BAR.
        """
        
        return self.model_fn(self.Mh, self.cosmo.z, theta_sfr)
    
    def evaluate_from_BAR(self, BAR_grid, M, theta_sfr):
        """
        Compute SFR(M, z) given an external BAR grid (e.g. for subhaloes).
        """

        model_fn = self.model_factory(BAR_grid, self.z_ratio)  # reuse structure
        return model_fn(M, self.cosmo.z, theta_sfr)
    
    