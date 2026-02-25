# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: cupix
#     language: python
#     name: cupix
# ---

# %% [markdown]
# # This notebook is mostly a wish list for how I (Andreu) would like to interact with cupix

# %% [markdown]
# ### Read a Px measurement

# %%
data_dir = '/path/at/nersc/with/lots/of/px/files/'
fname = data_dir + '/px_forecast.hdf5'
data = cupix.data.Px_data(fname)
# native binning (no rebinning)
Nz, Nt_a, Nk_M, Nk_m = data.U_ZaMn.shape
# rebinned values
Nz, Nt_A, Nk_M = data.Px_ZAM.shape

# %% [markdown]
# ### Plot a given Px measurement

# %%
# get the central value of each redshift bin, of length Nz
zs = data.z
# get a 1D array of central values of the measured k bins, of length Nk_M
k_M = data.k_M_mid
# although it's ok if the user needs to do this 
k_M = 0.5(data.k_M_edges[1:Nk_M] + data.k_M_edges[:Nk_M-1])

# %%
# get two 1D arrays with the edges of each theta bin, of length Nt_A each
theta_A_min = data.theta_min_A_arcmin
theta_A_max = data.theta_max_A_arcmin


# %%
# make a plot for a couple of theta bins, and one redshift bin
def plot_theta_bins(data, iz, it_M):
    label = '{} < theta < {}'.format(theta_A_min[it_M], theta_A_max[it_M])
    # 1D array with measured Px, length Nk_M
    Px = data.Px_ZAM[iz][it]
    # get also errorbars
    sig_Px = np.sqrt(np.diagonal(data.Cov_ZAM[iz][it]))
    plt.errorbars(k_M, Px, sig_Px, label=label)


# %% [markdown]
# ### For a given z bin, setup a Lya theory, without contaminants
#
# Alternatively, we could setup a LyaTheory without specifying a redshift, and then have a Theory_for_data object that does require you to provide a redshift and a binning. The problem here would be that then the Theory object would need to have a full model for the redshift evolution of all the relevant parameters. 
#
# For now, let's assume that we setup a new theory for each z bin.

# %%
# (fixed) cosmology to be used in the analysis
cosmo = camb_cosmo.get_cosmology(H0=68.0)
# setup theory for this redshift
iz = 1
z = zs[iz]

# %%
# Option 1: using the best guess at the Arinyo parameters (from P1D + ForestFlow)
theory_label = 'best_fit_arinyo_from_p1d'
theory_ari = cupix.likelihood.lya_theory.LyaTheory(theory_label, cosmo=cosmo, z=z)
bias = theory_ari.get_bias()
q1 = theory_ari.get_q1()
# or alternatively
beta = theory_ari.get_parameter_value(param_name='beta')
# this should crash
T0 = theory_ari.get_T0()

# %%
# Option 2: using the best guess at the IGM parameters (from P1D)
theory_label = 'best_fit_igm_from_p1d'
emu_label = 'ForestFlow_v2'
theory_igm = cupix.likelihood.lya_theory.LyaTheory(theory_label, cosmo=cosmo, z=z, emu_label = emu_label)
T0 = theory_igm.get_T0()
gamma = theory_igm.get_parameter_value(param_name='gamma')
# this would use ForestFlow to emulate these
bias = theory_igm.get_bias()
q1 = theory_igm.get_q1()
# this should now crash since beta is no longer a parameter name, it is a derived quantity
beta = theory_igm.get_parameter_value(param_name='beta')

# %% [markdown]
# ### Make theory predictions for Px

# %%
# the rest of the code shouldn't care about which theory you are using
theory = theory_ari
#theory = theory_igm

# %%
# numpy array of kpar values (in inverse AA)
kp_AA = np.linspace(0.01, 2.0, 100)
# numpy array of theta values (in arcmin)
theta_arc = np.linspace(0.1, 60.0, 100)
# get a 2D array prediction
Px_model = theory.get_Px_2d(theta_arc, kp_AA)

# %%
# plot the prediction for a couple of theta values
for it in [0, 5, 10]:
    label = 'theta = {}'.format(theta_arc[it])
    plt.plot(kp_AA, Px_model[it], label=label)

# %% [markdown]
# ### Now make predictions for different parameter values

# %%
# this could be a list of likelihood parameters, but easier to write it here as a dictionary
params = {'bias': bias, 'beta': beta, 'q1': q1}
# this would only modify the input parameters, and leave others (like k_p or av) unchanged
Px_model = theory.get_Px_2d(theta_arc, kp_AA, params=params) 

# %%
for bias in [-0.12, -0.13, -0.14]:
    # these would use the initial values for beta, q1, etc.
    Px_model = theory.get_Px_2d(theta_arc, kp_AA, params={'bias':bias}) 

# %% [markdown]
# ### Setup a likelihood object

# %%
# the likelihood is created for each redshift separately
like = cupix.likelihood.Likelihood(data=data, theory=theory, iz=iz)
# or even better, make it such that one can extract all the relevant information from the data for this iz (Px, Cov, window, etc)
one_z_data = data.get_one_z_data(iz)
like = cupix.likelihood.Likelihood(data=one_z_data, theory=theory)

# %% [markdown]
# ### Compute the convolved prediction for a rebinned theta bin

# %%
it_M = 5
# this should call the theory for all the relevant values of (k_m, theta_A) and convolve with window function
Px_model = like.get_convolved_Px(it_M=it_M)

# %%
# same, but for particular values of some of the parameters
Px_model = like.get_convolved_Px(it_M=it_M, params={'bias':bias}) 

# %% [markdown]
# ### Compute chi2 for different models

# %%
# chi2 for the default values in the theory
chi2 = like.get_chi2()

# %%
# chi2 scan when varying one parameter
for bias in [-0.12, -0.13, -0.14]:
    chi2 = like.get_chi2(params={'bias':bias})

# %%
# chi2 scan for two parameters
bias = np.linspace([-0.2, -0.05, Nbias])
beta = np.linspace([0.5, 2.5, Nbeta])
# if needed avoid loops... but I'm lazy
chi2 = np.zeros([Nbias, Nbeta])
for ibias in range(Nbias):
    for ibeta in range(Nbeta):
        chi2[ibias, ibeta] = like.get_chi2(params={'bias':bias[ibias], 'beta':beta[ibeta]})
# lazy here
plt.imshow(chi2)
