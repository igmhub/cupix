import numpy as np
import copy
import camb
from lace.cosmo import camb_cosmo
from lace.cosmo import fit_linP
from cupix.likelihood import likelihood_parameter


class CAMBModel(object):
    """Interface between CAMB object and Theory"""

    def __init__(self, zs, cosmo=None, z_star=3.0, kp_kms=0.009):
        """Setup from CAMB object and list of redshifts"""

        # list of redshifts at which we evaluate linear power
        self.zs = zs

        # setup CAMB cosmology object
        if cosmo is None:
            self.cosmo = camb_cosmo.get_cosmology()
        else:
            self.cosmo = cosmo
        
        # cache CAMB results when computed
        self.cached_camb_results = None
        # cache wavenumbers and linear power (at zs) when computed
        self.cached_linP_Mpc = None
        # cache linear power parameters at (z_star, kp_kms)
        self.z_star = z_star
        self.kp_kms = kp_kms
        self.cached_linP_params = None

    def get_likelihood_parameters(self, cosmo_priors=None):
        """Return a list of likelihood parameters"""

        # should clarify role of min/max given that these are also
        # set in the likelihood

        params = []
        params.append(
            likelihood_parameter.LikelihoodParameter(
                name="ombh2",
                min_value=0.018,
                max_value=0.026,
                value=self.cosmo.ombh2,
            )
        )
        params.append(
            likelihood_parameter.LikelihoodParameter(
                name="omch2",
                min_value=0.10,
                max_value=0.14,
                value=self.cosmo.omch2,
            )
        )
        if cosmo_priors is not None:
            min_val = cosmo_priors["As"][0]
            max_val = cosmo_priors["As"][1]
        else:
            min_val = 0.90e-09
            max_val = 3.60e-09
        params.append(
            likelihood_parameter.LikelihoodParameter(
                name="As",
                min_value=min_val,
                max_value=max_val,
                value=self.cosmo.InitPower.As,
            )
        )

        if cosmo_priors is not None:
            min_val = cosmo_priors["ns"][0]
            max_val = cosmo_priors["ns"][1]
        else:
            min_val = 0.85
            max_val = 1.10
        params.append(
            likelihood_parameter.LikelihoodParameter(
                name="ns",
                min_value=min_val,
                max_value=max_val,
                value=self.cosmo.InitPower.ns,
            )
        )
        params.append(
            likelihood_parameter.LikelihoodParameter(
                name="mnu",
                min_value=0.0,
                max_value=1.0,
                value=camb_cosmo.get_mnu(self.cosmo),
            )
        )

        if cosmo_priors is not None:
            min_val = cosmo_priors["nrun"][0]
            max_val = cosmo_priors["nrun"][1]
        else:
            min_val = -0.05
            max_val = 0.05
        params.append(
            likelihood_parameter.LikelihoodParameter(
                name="nrun",
                min_value=min_val,
                max_value=max_val,
                value=self.cosmo.InitPower.nrun,
            )
        )
        params.append(
            likelihood_parameter.LikelihoodParameter(
                name="H0", min_value=50, max_value=100, value=self.cosmo.H0
            )
        )

        return params

    def get_camb_results(self):
        """Check if we have called CAMB.get_results yet, to save time.
        It returns a CAMB.results object."""

        if self.cached_camb_results is None:
            self.cached_camb_results = camb_cosmo.get_camb_results(
                self.cosmo, zs=self.zs, fast_camb=True
            )

        return self.cached_camb_results

    def get_linP_Mpc(self):
        """Check if we have already computed linP_Mpc, to save time.
        It returns (k_Mpc, zs, linP_Mpc)."""

        if self.cached_linP_Mpc is None:
            camb_results = self.get_camb_results()
            self.cached_linP_Mpc = camb_cosmo.get_linP_Mpc(
                pars=self.cosmo, zs=self.zs, camb_results=camb_results
            )

        return self.cached_linP_Mpc

    def get_linP_params(self):
        """Linear power parameters at (z_star,kp_kms) for this cosmology"""

        if self.cached_linP_params is None:
            self.cached_linP_params = fit_linP.parameterize_cosmology_kms(
                self.cosmo, self.get_camb_results(), self.z_star, self.kp_kms
            )

        return self.cached_linP_params

    def get_linP_Mpc_params(self, kp_Mpc):
        """Get linear power parameters to call emulator, at each z.
        Amplitude, slope and running around pivot point kp_Mpc."""

        ## Get the P(k) at each z
        k_Mpc, z, pk_Mpc = self.get_linP_Mpc()

        # specify wavenumber range to fit
        kmin_Mpc = 0.5 * kp_Mpc
        kmax_Mpc = 2.0 * kp_Mpc

        linP_params = []
        ## Fit the emulator call params
        for pk_z in pk_Mpc:
            linP_Mpc = fit_linP.fit_polynomial(
                kmin_Mpc / kp_Mpc,
                kmax_Mpc / kp_Mpc,
                k_Mpc / kp_Mpc,
                pk_z,
                deg=2,
            )
            # translate the polynomial to our parameters
            ln_A_p = linP_Mpc[0]
            Delta2_p = np.exp(ln_A_p) * kp_Mpc**3 / (2 * np.pi**2)
            n_p = linP_Mpc[1]
            # note that the curvature is alpha/2
            alpha_p = 2.0 * linP_Mpc[2]
            linP_z = {"Delta2_p": Delta2_p, "n_p": n_p, "alpha_p": alpha_p}
            linP_params.append(linP_z)

        return linP_params
    
    def dkms_dMpc(self, z):
        """Return H(z)/(1+z) to convert Mpc to km/s"""

        # get CAMB results objects (might be cached already)
        camb_results = self.get_camb_results()
        H_z = camb_results.hubble_parameter(z)
        return H_z / (1 + z)
    
    def get_M_kms_of_zs(self):
        """Return M(z)=H(z)/(1+z) for each z"""
        M_kms_of_zs = []
        for z in self.zs:    
            M_kms_of_zs.append(camb_cosmo.dkms_dMpc(self.cosmo, z))

        return M_kms_of_zs
    
    def get_M_AA_of_zs(self):
        """Return conversion factor from inv cMpc to inv Angstroms for each z"""
        M_AA_of_zs = []
        for z in self.zs:
            M_AA_of_zs.append(camb_cosmo.dAA_dMpc(self.cosmo, z, 1215.67))

        return M_AA_of_zs
    
    def get_M_tdeg_of_zs(self):
        """Return conversion factor from Mpc to deg for each z"""
        M_tdeg_of_zs = []
        for z in self.zs:
            M_tdeg_of_zs.append(camb_cosmo.ddeg_dMpc(self.cosmo, z))

        return M_tdeg_of_zs

    def get_new_model(self, zs, like_params):
        """For an arbitrary list of like_params, return a new CAMBModel"""

        # store a dictionary with parameters set to input values
        camb_param_dict = {}

        # loop over list of likelihood parameters own by this object
        for mypar in self.get_likelihood_parameters():
            # loop over input parameters
            for inpar in like_params:
                if inpar.name == mypar.name:
                    camb_param_dict[inpar.name] = inpar.value
                    continue

        # set cosmology object (use fiducial for parameters not provided)
        new_cosmo = camb_cosmo.get_cosmology_from_dictionary(
            camb_param_dict, cosmo_fid=self.cosmo
        )

        return CAMBModel(zs=zs, cosmo=new_cosmo)
