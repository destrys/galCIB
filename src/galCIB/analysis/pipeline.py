#from galCIB.galaxy import get_hod_model
from scipy.integrate import simpson
import numpy as np 

class AnalysisModel:
    def __init__(self, cosmology, survey, pk3d):
        """
        Initializes the model with fixed cosmology and survey properties.

        Args:
            cosmology: Cosmology object (with Mh, z, k, P_lin, etc.)
            survey: Survey object (with z, pz, compute_Wg, etc.)
            hod_name: Name of the default HOD model to use
            profile_type: e.g., "nfw" or "mixed"
        """
        
        self.cosmo = cosmology
        self.survey = survey
        self.pk = pk3d
        
        self.geom_factor = self.cosmo.geom_factor
        self.Wg = survey.Wg
        self.Wcib = survey.Wcib
        
        self.Nell = len(self.survey.ells)
        self.Nz = len(self.survey.z)
        self.Nnu = len(self.survey.nu_obs)
        self.Nnu_comb = self.Nnu * (self.Nnu + 1)//2
    
    def _kPk_interpolator(self, pk):
        """
        k and Pk need to be interpolated on the same grid 
        to calculate C_ell. The grid is:
        k = (ell+0.5)/chi(z)
        
        Returns:
            kz_grid : (Nell, Nz)
            Pk_grid : (Nell, Nz)
        """
        
        self.kz_grid = (self.survey.ells[:,np.newaxis]+0.5)/self.cosmo.chi[np.newaxis,:]
        
        # interpolate pk 
        
        pk_interp = np.zeros((self.Nell, self.Nz))
        
        for zidx in range(len(self.survey.z)):
            pk_interp[:,zidx] = np.interp(self.kz_grid[:,zidx],
                                          self.cosmo.k,
                                          pk[:,zidx])
            
        return pk_interp
        
    def compute_cl(self, pkz_xy, Wx, Wy):
        """
        Return C_ell gg using Limber. 
        
        Args:
            pkz_xy : interpolated P(k) of X and Y fields.
            Wx : Window of X
            Wy : Window of Y
        """
        
        integrand = self.geom_factor*Wx*Wy*pkz_xy
        cl = simpson(integrand,x=self.survey.z,axis=-1)
        
        return cl
    
    def update_cl(self, theta_cen=None, theta_sat=None,
                  theta_prof=None,
                  theta_sfr=None, theta_snu=None, theta_IR_hod=None,
                  hmalpha=1):
        """
        Recalculates C_ell based on parameters. 
        """
        
        pgg, pII, pgI = self.pk.compute_pk(theta_cen=theta_cen,
                                        theta_sat=theta_sat,
                                        theta_prof=theta_prof,
                                        theta_sfr=theta_sfr,
                                        theta_snu=theta_snu,
                                        theta_IR_hod=theta_IR_hod,
                                        hmalpha=1)
        
        # interpolate on the ell-to-k grid
        pgg_int = self._kPk_interpolator(pgg)
        
        pgI_int = np.zeros((self.Nnu, self.Nell, self.Nz)) # Nnu, Nk, Nz
        for i in range(self.Nnu):
            pgI_int[i] = self._kPk_interpolator(pgI[i])
        
        pII_int = np.zeros((self.Nnu_comb, self.Nell, self.Nz))
        for i in range(self.Nnu_comb):
            pII_int[i] = self._kPk_interpolator(pII[i])
            
        # compute C_ell 
        Cgg = self.compute_cl(pgg_int, self.Wg, self.Wg)
        CgI = self.compute_cl(pgI_int, self.Wg, self.Wcib)
        CII = self.compute_cl(pII_int, self.Wcib, self.Wcib)
        
        return Cgg, CgI, CII 