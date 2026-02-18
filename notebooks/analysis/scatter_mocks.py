# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: forestflow
#     language: python
#     name: forestflow
# ---

# %%
import numpy as np
from cupix.likelihood.lya_theory import set_theory
from cupix.likelihood.forestflow_emu import FF_emulator
from cupix.likelihood.input_pipeline import Args
from cupix.likelihood.likelihood_parameter import LikelihoodParameter
from cupix.likelihood.likelihood import Likelihood
from lace.cosmo import camb_cosmo, fit_linP
import matplotlib.pyplot as plt
from cupix.likelihood.window_and_rebin import convolve_window, rebin_theta
from cupix.px_data.data_DESI_DR2 import DESI_DR2
import scipy
from cupix.likelihood.iminuit_minimizer import IminuitMinimizer
from astropy.io import fits
import cupix
import os
omnuh2 = 0.0006
mnu = omnuh2 * 93.14
H0 = 67.36
omch2 = 0.12
ombh2 = 0.02237
As = 2.1e-9
ns = 0.9649
nrun = 0.0
w = -1.0
omk = 0
fid_cosmo = {
    'H0': H0,
    'omch2': omch2,
    'ombh2': ombh2,
    'mnu': mnu,
    'omk': omk,
    'As': As,
    'ns': ns,
    'nrun': nrun,
    'w': w
}
theory_AA = None


# %%
def get_params(data):
    bias, beta, bias_err, beta_err = data['bias'], data['beta'], data['bias_err'], data['beta_err']
    return(bias,beta,bias_err,beta_err)
    
def get_chi2_prob(data):
    chi2, prob = data['chi2'], data['prob']
    return(chi2,prob)
def plot_ellipses(pname_x, pname_y, val_x, val_y, sig_x, sig_y, cov, r, nsig=2, true_vals=None, true_val_label="true value"):
        """Plot Gaussian contours for parameters (pname_x,pname_y)
        - nsig: number of sigma contours to plot
        - cube_values: if True, will use unit cube values."""

        from matplotlib.patches import Ellipse
        from numpy import linalg as LA
        print(r)

        # shape of ellipse from eigenvalue decomposition of covariance
        w, v = LA.eig(
            np.array(
                [
                    [sig_x**2, sig_x * sig_y * r],
                    [sig_x * sig_y * r, sig_y**2],
                ]
            )
        )

        # semi-major and semi-minor axis of ellipse
        a = np.sqrt(w[0])
        b = np.sqrt(w[1])
        print(a,b)
        # figure out inclination angle of ellipse
        alpha = np.arccos(v[0, 0])
        if v[1, 0] < 0:
            alpha = -alpha
        # compute angle in degrees (expected by matplotlib)
        alpha_deg = alpha * 180 / np.pi

        # make plot
        fig = plt.subplot(111)
        for isig in range(1, nsig + 1):
            ell = Ellipse(
                (val_x, val_y), 2 * isig * a, 2 * isig * b, angle=alpha_deg
            )
            ell.set_alpha(0.6 / isig)
            fig.add_artist(ell)
        if true_vals is not None:
            plt.axvline(true_vals[pname_x], color='grey', linestyle='--', label=true_val_label)
            plt.axhline(true_vals[pname_y], color='grey', linestyle='--')
                        
            plt.legend()
        plt.xlabel(pname_x)
        plt.ylabel(pname_y)
        if true_vals is None:
            plt.xlim(val_x - (nsig + 1) * sig_x, val_x + (nsig + 1) * sig_x)
            plt.ylim(val_y - (nsig + 1) * sig_y, val_y + (nsig + 1) * sig_y)
        else:
            minx = min(val_x - (nsig + 1) * sig_x, true_vals[pname_x]-.1*abs(true_vals[pname_x]))
            maxx = max(val_x + (nsig + 1) * sig_x, true_vals[pname_x]+.1*abs(true_vals[pname_x]))
            miny = min(val_y - (nsig + 1) * sig_y, true_vals[pname_y]-.1*abs(true_vals[pname_y]))
            maxy = max(val_y + (nsig + 1) * sig_y, true_vals[pname_y]+.1*abs(true_vals[pname_y]))
            plt.ylim([miny,maxy])
            plt.xlim([minx,maxx])


# %%
z = [2.2]
# Load Laura's CF fits
with fits.open(f"/global/cfs/cdirs/desicollab/science/lya/mock_analysis/develop/ifae-ql/qq_desi_y3/v1.0.5/analysis-0/jura-124/raw_bao_unblinding/fits/output_fitter-z-bins/bin_{z[0]}/lyaxlya.fits") as zbin_cf_file:
    zbin_cf_fit = zbin_cf_file[1].header
    cf_bias = zbin_cf_fit['bias_LYA']
    cf_beta = zbin_cf_fit['beta_LYA']
    cf_q1   = zbin_cf_fit['dnl_arinyo_q1']
    cf_kv   = zbin_cf_fit['dnl_arinyo_kv']
    cf_av   = zbin_cf_fit['dnl_arinyo_av']
    cf_bv   = zbin_cf_fit['dnl_arinyo_bv']
    cf_kp   = zbin_cf_fit['dnl_arinyo_kp']

# new Laura fit parameters

like_params = []

like_params.append(LikelihoodParameter(
    name='bias',
    min_value=-1.0,
    max_value=1.0,
    value=cf_bias,
    ini_value=-0.3,
    Gauss_priors_width=.5
    ))
like_params.append(LikelihoodParameter(
    name='beta',
    min_value=0.0,
    max_value=3.0,    
    value = cf_beta,
    ini_value = 2.3,
    Gauss_priors_width=1
    ))
like_params.append(LikelihoodParameter(
    name='q1',
    min_value=0.0,
    max_value=1.0,
    ini_value = 0.1112,
    value = cf_q1,
    Gauss_priors_width=.5
    ))
like_params.append(LikelihoodParameter(
    name='kvav',
    min_value=0.0,
    max_value=1.0,
    ini_value = 0.0001**0.2694,
    value = cf_kv**cf_av
    ))
like_params.append(LikelihoodParameter(
    name='av',
    min_value=0.0,
    max_value=2.0,
    ini_value = 0.2694,
    value = cf_av
    ))
like_params.append(LikelihoodParameter(
    name='bv',
    min_value=0.0,
    max_value=2.0,
    ini_value = 0.0002,
    value = cf_bv
    ))
like_params.append(LikelihoodParameter(
    name='kp',
    min_value=0.0,
    max_value=1.0,
    ini_value = 0.5740,
    value = cf_kp
    ))

# %%
stack = np.load(f"/global/common/software/desi/users/mlokken/cupix/data/fitter_results/iminuit_stack_tru_cont_binned_mocks_bias_beta_z{z[0]}.npz")
stack_bias,stack_beta,stack_bias_err,stack_beta_err = get_params(stack)
stack_chi2,stack_prob = get_chi2_prob(stack)
stack_cov = stack['cov']
stack_r   = stack['r']
nmocks = 50
biases = []
betas = []
chi2s = []
for n in range(nmocks):
    mockn = np.load(f"/pscratch/sd/m/mlokken/desi-lya/px/iminuit_mock{n}_tru_cont_stack_binned_bias_beta.npz")
    mockbias, mockbeta, mockbias_err,mockbeta_err = get_params(mockn)
    mock_chi2, mock_prob = get_chi2_prob(mockn)
    
    biases.append(mockbias)
    betas.append(mockbeta)
    chi2s.append(mock_chi2)

# %%
plt.rc('font', size=16) 
plot_ellipses('bias','beta',stack_bias,stack_beta,stack_bias_err,stack_beta_err, stack_cov, stack_r, true_vals={"bias":cf_bias, "beta":cf_beta}, true_val_label=r"$\xi$ best-fit")
plt.plot(stack_bias,stack_beta,'*')
plt.plot(biases, betas, 'o', markersize=2, color='k')
plt.ylabel(r"$\beta$")

# %%
chi2 = scipy.stats.chi2(160-2)

# %%
x = np.arange(100, 250)

# %%
plt.hist(chi2s, density=True, label='scatter of mocks')
plt.plot(x, scipy.stats.chi2.pdf(x, 160), label='expected')
plt.xlabel(r"$\chi^2$")
plt.legend()

# %%

# %%


