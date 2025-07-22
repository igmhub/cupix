import numpy as np
import copy
import os
import lace
from scipy.interpolate import interp1d
from cupix.likelihood import likelihood_parameter


class MeanFluxModel(object):
    """Use a handful of parameters to model the mean transmitted flux fraction
    (or mean flux) as a function of redshift.
     For now, we use a polynomial to describe log(tau_eff) around z_tau.
    """

    def __init__(
        self,
        z_tau=3.0,
        ln_tau_coeff=None,
        free_param_names=None,
        fid_igm=None,
        order_extra=2,
        smoothing=False,
        priors=None,
    ):
        """Construct model as a rescaling around a fiducial mean flux"""

        if fid_igm is None:
            repo = os.path.dirname(lace.__path__[0]) + "/"
            fname = repo + "data/sim_suites/Australia20/IGM_histories.npy"
            try:
                igm_hist = np.load(fname, allow_pickle=True).item()
            except:
                raise ValueError(
                    fname
                    + " not found. You can produce it using the LaCE"
                    + r" script save_mpg_IGM.py"
                )
            else:
                fid_igm = igm_hist["mpg_central"]
        self.fid_igm = fid_igm
        self.priors = priors

        mask = fid_igm["tau_eff"] != 0
        if np.sum(mask) == 0:
            raise ValueError("No non-zero tau_eff in fiducial IGM")
        elif np.sum(mask) != fid_igm["tau_eff"].shape[0]:
            print(
                "The fiducial value of tau_eff is zero for z: ",
                fid_igm["z_tau"][mask == False],
            )

        # fit power law to fiducial data to reduce noise
        pfit = np.polyfit(
            fid_igm["z_tau"][mask],
            np.log(fid_igm["tau_eff"][mask]),
            order_extra,
        )
        p = np.poly1d(pfit)
        _ = fid_igm["z_tau"] != 0
        ind = np.argsort(fid_igm["z_tau"][_])
        self.fid_z = fid_igm["z_tau"][_][ind]
        self.fid_tau_eff = fid_igm["tau_eff"][_][ind]

        # use poly fit to interpolate when data is missing (needed for Nyx)
        mask2 = self.fid_tau_eff == 0
        self.fid_tau_eff[mask2] = np.exp(p(self.fid_z[mask2]))

        # extrapolate to z=1.9 and 5.5 (if needed)
        if np.min(self.fid_z) > 1.9:
            low = True
        else:
            low = False
        if np.max(self.fid_z) < 5.5:
            high = True
        else:
            high = False

        if low and high:
            z_to_inter = np.concatenate([[1.9], self.fid_z, [5.5]])
        elif low:
            z_to_inter = np.concatenate([[1.9], self.fid_z])
        elif high:
            z_to_inter = np.concatenate([self.fid_z, [5.5]])
        else:
            z_to_inter = self.fid_z
        igm_to_inter = np.zeros_like(z_to_inter)

        # extrapolate to low z
        if low:
            igm_to_inter[0] = np.exp(p(z_to_inter[0]))
        # extrapolate to high z
        if high:
            igm_to_inter[-1] = np.exp(p(z_to_inter[-1]))

        ind_all = np.arange(z_to_inter.shape[0])
        if low and high:
            ind_use = ind_all[1:-1]
        elif low:
            ind_use = ind_all[1:]
        elif high:
            ind_use = ind_all[:-1]
        else:
            ind_use = ind_all

        # apply smoothing to IGM history
        if smoothing:
            igm_to_inter[ind_use] = np.exp(p(z_to_inter[ind_use]))
        else:
            igm_to_inter[ind_use] = self.fid_tau_eff

        self.fid_tau_interp = interp1d(z_to_inter, igm_to_inter, kind="cubic")

        self.z_tau = z_tau
        if ln_tau_coeff:
            assert free_param_names is None
            self.ln_tau_coeff = ln_tau_coeff
        else:
            if free_param_names:
                # figure out number of mean flux free params
                n_mf = len([p for p in free_param_names if "ln_tau_" in p])
            else:
                n_mf = 2
            self.ln_tau_coeff = [0.0] * n_mf
        # store list of likelihood parameters (might be fixed or free)
        self.set_parameters()

    def get_Nparam(self):
        """Number of parameters in the model"""
        assert len(self.ln_tau_coeff) == len(self.params), "size mismatch"
        return len(self.ln_tau_coeff)

    def power_law_scaling(self, z, like_params=[]):
        """Power law rescaling around z_tau"""

        ln_tau_coeff = self.get_tau_coeffs(like_params=like_params)

        xz = np.log((1 + z) / (1 + self.z_tau))
        ln_poly = np.poly1d(ln_tau_coeff)
        ln_out = ln_poly(xz)
        return np.exp(ln_out)

    def get_tau_eff(self, z, like_params=[]):
        """Effective optical depth at the input redshift"""
        tau_eff = self.power_law_scaling(
            z, like_params=like_params
        ) * self.fid_tau_interp(z)
        return tau_eff

    def get_mean_flux(self, z, like_params=[]):
        """Mean transmitted flux fraction at the input redshift"""
        tau = self.get_tau_eff(z, like_params=like_params)
        return np.exp(-tau)

    def set_parameters(self):
        """Setup likelihood parameters in the mean flux model"""

        self.params = []
        Npar = len(self.ln_tau_coeff)
        for i in range(Npar):
            name = "ln_tau_" + str(i)
            pname = "tau_eff"
            if i == 0:
                if self.priors is not None:
                    xmin = self.priors[pname][-1][0]
                    xmax = self.priors[pname][-1][1]
                else:
                    xmin = -0.2
                    xmax = 0.2
            else:
                if self.priors is not None:
                    xmin = self.priors[pname][-2][0]
                    xmax = self.priors[pname][-2][1]
                else:
                    xmin = -0.5
                    xmax = 0.5
            # note non-trivial order in coefficients
            value = self.ln_tau_coeff[Npar - i - 1]
            par = likelihood_parameter.LikelihoodParameter(
                name=name, value=value, min_value=xmin, max_value=xmax
            )
            self.params.append(par)

        return

    def get_parameters(self):
        """Return likelihood parameters for the mean flux model"""
        return self.params

    def get_tau_coeffs(self, like_params=[]):
        """Return list of mean flux coefficients"""

        if like_params:
            ln_tau_coeff = self.ln_tau_coeff.copy()
            Npar = 0
            array_names = []
            array_values = []
            for par in like_params:
                if "ln_tau" in par.name:
                    Npar += 1
                    array_names.append(par.name)
                    array_values.append(par.value)
            array_names = np.array(array_names)
            array_values = np.array(array_values)

            if Npar != len(self.params):
                raise ValueError(
                    f"number of params {Npar} vs {self.params} mismatch in get_tau_coeffs"
                )

            for ip in range(Npar):
                _ = np.argwhere(self.params[ip].name == array_names)[:, 0]
                if len(_) != 1:
                    raise ValueError(
                        "could not update parameter" + self.params[ip].name
                    )
                else:
                    ln_tau_coeff[Npar - ip - 1] = array_values[_[0]]
        else:
            ln_tau_coeff = self.ln_tau_coeff

        return ln_tau_coeff
