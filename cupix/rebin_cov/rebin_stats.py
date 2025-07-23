import numpy as np
import h5py
from cupix.rebin_cov.Px_meas import Px_meas
from collections import defaultdict


def compute_binned_stats_px(input_path,k_bins_ratio,k_max_ratio,theta_bins_ratio,theta_bin_opt,k_bin_opt,unbinned_px):
    px_dr2 = Px_meas()
    
    px_dr2.read_hdf5(input_path)
    
    key = px_dr2.px.keys()
    zbin_to_keys = defaultdict(list)
    for keys in px_dr2.px.keys():
        zbin = keys[0]  # Extract the z_bin  from the key tuple
        zbin_to_keys[zbin].append(keys)
    
    bin_info = {} 
    N_fft = px_dr2.N_fft
    pw_A = px_dr2.pw_A
    L_fft=N_fft*pw_A
    k_arr = px_dr2.k_arr
    
    # we will rebin the wavenumbers to make them more independent, and better measured
    k0_fft=2*np.pi/L_fft
    dk_bin=k0_fft*k_bins_ratio
    
    print('dk =',dk_bin)
    
    # stop roughly at 1/4 of the Nyquist frequency for now (equivalent to rebinning 4 pixels)
    k_Ny_fft=np.pi/pw_A
    k_max=k_Ny_fft/k_max_ratio
    
    print('k < ',k_max)
    
    k_edges=np.arange(0.01*dk_bin,k_max+dk_bin,dk_bin)
    
    Nk=k_edges.size-1
    
    print('Nk =',Nk)
    
    bin_info['Nk'] = Nk
    bin_info['k_max'] = k_max
    bin_info['k_edges'] = k_edges
    bin_info['N_fft'] = N_fft
    
    
    zbins = np.unique(px_dr2.z_bins)
    theta_bins = np.unique(px_dr2.theta_bins)
    
    # we need an average over all theta_bins belonging to the new theta_rebins, which will give the array (z,nhp,Nk) for each theta_rebins
    Ntheta_a = len(theta_bins)-1 # no. of theta bins
    downsize = theta_bins_ratio
    Ntheta_A = Ntheta_a//downsize
    print('number of theta bins:',Ntheta_A)
    
    # just redefine using np logspace 
    x0 = np.log10(theta_bins.min())
    x1 = np.log10(theta_bins.max())
    x_arr = np.logspace(x0,x1,Ntheta_A+1)
    
    theta_min_rebin = np.zeros(len(x_arr)-1)
    theta_max_rebin = np.zeros(len(x_arr)-1)
    
    for i in range(len(x_arr)-1):
        print(x_arr[i],x_arr[i+1])
        theta_min_rebin[i] = x_arr[i]
        theta_max_rebin[i] = x_arr[i+1]
    
    #print(theta_min_rebin)
    #print(theta_max_rebin)
    #print(len(theta_min_rebin))
    
    # unbinned px avg and cov
    if unbinned_px:
        px_dr2.compute_stat()
    
    # bin in theta
    if theta_bin_opt:
        px_dr2.compute_thetabinned_px(theta_min_rebin,theta_max_rebin)
    
    #now bin in k 
    if k_bin_opt:
        px_dr2.compute_binned_cov(bin_info,theta_binning=theta_bin_opt)
    

    return px_dr2
