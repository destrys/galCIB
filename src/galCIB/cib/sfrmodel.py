#cib/sfrmodel.py

from .registry import get_sfr_model
from .utils import _compute_BAR_grid

class SFRModel:
    def __init__(self, name: str, hod, cosmo):
        self.z_ratio = hod.z_ratio
        self.cosmo = cosmo
        self.BAR_grid = _compute_BAR_grid(cosmo) # (1, NMh, Nz)
        self.model_factory = get_sfr_model(name)
        self.model_fn = self.model_factory(self.BAR_grid, self.z_ratio)
        
    def __call__(self, theta_sfr):
        """
        Standard call: computes SFR(Mh, z) using precomputed BAR.
        """
        
        return self.model_fn(self.cosmo.Mh_grid[0],
                             self.cosmo.z, theta_sfr)
    
    def evaluate_from_BAR(self, BAR_grid, M, theta_sfr):
        """
        Compute SFR(M, z) given an external BAR grid (e.g. for subhaloes).
        """
        #print(f'BAR shape = {BAR_grid.shape}')
        #print(f'm shape = {M.shape}')
        model_fn = self.model_factory(BAR_grid, self.z_ratio)  # reuse structure
        return model_fn(M, self.cosmo.z, theta_sfr)
    
    