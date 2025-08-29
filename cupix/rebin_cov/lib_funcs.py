# Functions related to binning and averaging that can be used as libraries across cupix

import numpy as np
import h5py

def bin_func_k(k_arr,k_fund,k_bins_ratio,k_max_ratio,bin_func_type):
    dk_bin=k_fund*k_bins_ratio
    print('Computing B^M_m...')
    print('dk =',dk_bin)

    # stop roughly at 1/4 of the Nyquist frequency for now (equivalent to rebinning 4 pixels)
    k_max= k_arr.max()/k_max_ratio #k_Nyq/k_max_ratio

    print('k < ',k_max)

    k_edges=np.arange(0.01*dk_bin,k_max+dk_bin,dk_bin)

    Nk=k_edges.size-1
    N_fft = len(k_arr)

    #define bin function
    B_M_m=np.zeros([Nk,N_fft]) # includes negative k values
    if bin_func_type == 'top_hat':
        for i in range(Nk):
            inbin=(abs(k_arr)>k_edges[i]) & (abs(k_arr)<k_edges[i+1])
            B_M_m[i,inbin]=1
    k_M =0.5*(k_edges[:-1] + k_edges[1:])        
    print('Done')

    return B_M_m,k_M

def bin_func_theta(theta_bin_min, theta_bin_max, theta_bins_ratio,bin_func_type):
    """
    Rebin theta in log-space by merging original a-bins into N_A coarse bins.
    Returns:
      B_A_a               (N_A, N_a)
      theta_min_rebin     (N_A,)
      theta_max_rebin     (N_A,)
    """
    theta_min = np.asarray(theta_bin_min, float)
    theta_max = np.asarray(theta_bin_max, float)
    N_a = len(theta_min)
    assert len(theta_max) == N_a, "theta_min/max length mismatch"

    # How many coarse bins?
    downsize = int(theta_bins_ratio)
    N_A = N_a // downsize  # you can choose something else if you want
    print('number of theta bins:', N_A)
    if N_A < 1:
        raise ValueError("N_A < 1; choose a smaller downsize/theta_bins_ratio.")

    # Use geometric centers for classification (since we are binning in log space)
    c_a = np.sqrt(theta_min * theta_max)
    lo, hi = c_a.min(), c_a.max()
    print(c_a)

    # Log-spaced coarse edges over the full range of centers
    edges = np.logspace(np.log10(lo), np.log10(hi), N_A + 1)

    
    # Assign each original bin 'a' to a coarse bin 'A'
    A_idx = np.digitize(c_a, edges, right=False) - 1   # in [0, N_A-1]
    
    underflows = np.sum(A_idx < 0)
    overflows  = np.sum(A_idx >= N_A)

    if underflows or overflows:
        print(f"[bin_func_theta] WARNING: {underflows} underflows, {overflows} overflows before clipping")
        A_idx = np.clip(A_idx, 0, N_A - 1) # take care of underflow and overflow

    B_A_a = np.zeros((N_A, N_a), float)
    if bin_func_type == 'top_hat':
        for a, A in enumerate(A_idx):
            B_A_a[A, a] += 1.0

    theta_min_rebin = edges[:-1]
    theta_max_rebin = edges[1:]
    
    return B_A_a, theta_min_rebin, theta_max_rebin


def rebin_k(F_zh_am,B_M_m):
    # numerator: (Z,H,a,M)
    num_k = np.einsum('zham,Mm->zhaM', F_zh_am, B_M_m, optimize=True)

    # denom: (M,) -> (1,1,1,M) for broadcasting
    den_k = np.einsum('Mm->M', B_M_m)[None, None, None, :]

    F_zh_aM = np.divide(num_k, den_k, out=np.zeros_like(num_k), where=den_k>0)
    return F_zh_aM

def rebin_theta(F_zh_aM,B_A_a):
    # numerator: (Z,H,A,M)
    num_k = np.einsum('zahM,Aa->zAhM', F_zh_aM, B_A_a, optimize=True)

    # denom: (M,) -> (1,1,1,M) for broadcasting
    den_k = np.einsum('Aa->A', B_A_a)[None, :, None, None]

    F_zh_AM = np.divide(num_k, den_k, out=np.zeros_like(num_k), where=den_k>0)

    return F_zh_AM

def calculate_estnorm(W, R, L):
    '''
    W (np.ndarray): vector length N, average FFT of the weights per healpix 
    R (np.ndarray): vector length N, resolution in Fourier space
    L (float): length of the spectra (in physical units, e.g. Angstroms or Mpc)
    Returns:
    estnorm (np.ndarray): vector length N, to be multiplied by every P1D mode of the measurement
    '''
    R2 = R.real**2 + R.imag**2
    denom = np.absolute(np.fft.ifft(np.fft.fft(W)* np.fft.fft(R2)))    
    estnorm = np.absolute(denom/L)
    return estnorm  

def calculate_V_zh_AM(W_zh_AM,R_zh_AM,L):
    R2 = R_zh_AM.real**2 + R_zh_AM.imag**2
    denom = np.absolute(np.fft.ifft(np.fft.fft(W_zh_AM,axis=-1)* np.fft.fft(R2)))    
    estnorm = np.absolute(denom/L)
    return estnorm  

def average_over_hp(F_zh_AM,W_zh_AM,R_zh_AM,L):
    W_z_AM = np.einsum('zAhM->zAM',W_zh_AM)
    R_z_AM = np.einsum('zAhM->zAM',R_zh_AM)
    
    V_z_AM = calculate_V_zh_AM(W_z_AM,R_z_AM,L)
    
    num_k = np.einsum('zAhM->zAM', F_zh_AM)

    P_z_AM = np.divide(num_k, V_z_AM, out=np.zeros_like(num_k), where=V_z_AM>0)

    return P_z_AM, W_z_AM, R_z_AM

def compute_covariance(F_zh_AM,W_zh_AM,R_zh_AM,L_fft):
    P_z_AM, _, _ = average_over_hp(F_zh_AM,W_zh_AM,R_zh_AM,L_fft)
    V_zh_AM = calculate_V_zh_AM(W_zh_AM,R_zh_AM,L_fft)
    V_z_AM = np.einsum('zAhM->zAM',V_zh_AM)
    P_zh_AM = np.divide(F_zh_AM,V_zh_AM,out=np.zeros_like(F_zh_AM),where=V_zh_AM>0)#F_zh_AM/V_zh_AM
    P_diff = P_zh_AM - P_z_AM[:,:,None,:]

    cov_num = np.einsum('zAhM,zAhN,zAhM,zAhN->zAMN',V_zh_AM,V_zh_AM,P_diff,P_diff) 
    denom =  V_z_AM[:, :, :, None] * V_z_AM[:, :, None, :]

    C_z_AMN = np.divide(cov_num,denom,out=np.zeros_like(cov_num),where=denom>0)

    return C_z_AMN

def calculate_window_matrix(W_zh_AM, R_zh_AM, L):
    '''
    W (np.ndarray): average of (w1) conj(w2) where w1 and w2 are FFT of original weights per skewer
    R (np.ndarray): vector length N, resolution in Fourier space
    L (float): physical length of skewers (e.g., in Angstroms)
    Returns:
    window_matrix (np.ndarray): window matrix to be convolved with pure theory
    estnorm (np.ndarray): vector length N, to be multiplied by every P1D mode of the measurement
    '''
    W_z_AM = np.einsum('zAhM->zAM',W_zh_AM)
    R_z_AM = np.einsum('zAhM->zAM',R_zh_AM)
    
    R2 = R_z_AM.real**2 + R_z_AM.imag**2
    denom = np.absolute(np.fft.ifft(np.fft.fft(W_z_AM,axis=-1)* np.fft.fft(R2)))
  
    Nz, Ntheta_A, Nk = W_z_AM.shape
    print(Nz, Ntheta_A,Nk) 
    window_matrix = np.zeros((Nz,Ntheta_A,Nk,Nk))
    for m in range(Nk):
        for n in range(Nk):
            #window_matrix[:,:,m,n] = W_z_AM[:,:,m-n]*R2[:,:,n] / denom[:,:,m]
            num_k = W_z_AM[:,:,m-n]*R2[:,:,n]
            window_matrix[:,:,m,n] = np.divide(num_k,denom[:,:,m], out=np.zeros_like(num_k), where=denom[:,:,m]>0)
    return window_matrix

def bin_window(U_z_amn,B_M_m,W_zh_am,R_zh_am,R_zh_aM,L):
    W_z_am = np.einsum('zahm->zam',W_zh_am)
    R_z_am = np.einsum('zahm->zam',R_zh_am)
    
    W_zh_aM = rebin_k(W_zh_am,B_M_m)
    W_z_aM = np.einsum('zahm->zam',W_zh_aM)
    R_z_aM = np.einsum('zahm->zam',R_zh_aM)
    
    V_z_am = calculate_V_zh_AM(W_z_am,R_z_am,L)
    V_z_aM = calculate_V_zh_AM(W_z_aM,R_z_aM,L)

    # Numerator: sum over m
    num = np.einsum('Mm,zam,zamn->zaMn', B_M_m, V_z_am, U_z_amn, optimize=True)

    # Safe divide, broadcasting over n
    U_z_aMn = np.divide(num, V_z_aM[..., None], out=np.zeros_like(num), where=V_z_aM[..., None] > 0)

    return U_z_aMn, V_z_aM, V_z_am


def save_to_hdf5(filename,P_z_AM,C_z_AMN,U_z_aMn,B_A_a,V_z_aM,V_z_am,k_m,k_M,theta_bin_min,theta_bin_max,theta_min_A,theta_max_A,N_fft,L_fft,zbin_centers):

    with h5py.File(filename, 'w') as f:  
        # all bins before and after rebinning
        f.create_dataset('k_m', data=k_m)
        f.create_dataset('k_M',data=k_M)
        f.create_dataset('theta_min_a',data=theta_bin_min)
        f.create_dataset('theta_max_a',data=theta_bin_max)
        f.create_dataset('theta_min_A',data=theta_min_A)
        f.create_dataset('theta_max_A',data=theta_max_A)
        f.create_dataset('z_centers',data=zbin_centers)

        f.attrs['N_fft'] = N_fft
        f.attrs['L_fft'] = L_fft

        f.create_dataset("P_z_AM", data=P_z_AM, compression="gzip", compression_opts=4)
        f.create_dataset("C_z_AMN", data=C_z_AMN, compression="gzip", compression_opts=4)
        f.create_dataset("U_z_aMn", data=U_z_aMn, compression="gzip", compression_opts=4)
        f.create_dataset("B_A_a", data=B_A_a, compression="gzip", compression_opts=4)
        f.create_dataset("V_z_aM", data=V_z_aM, compression="gzip", compression_opts=4)
        f.create_dataset("V_z_am", data=V_z_am, compression="gzip", compression_opts=4)


    return None











    return None



def global_avg(fm_data, wm_data,N_fft,pw_A,R_avg):
    """Computes the global average of Px measurements across all healpix pixels
    Args:
        fm_data: Px in each healpix in a given z and theta bin, having shape  (Nhp, N_k)
        wm_data: Weights on the Px measurement in each healpix, same shape as px

    Returns:
        The global average of Px
    """
    # Method 1 
    # F^a_m  =  Sum over all healpixels(F^a_m,hp)
    # W^a_m  =  Sum over all healpixels(W^a_m,hp)
    # Px^a_m =  F^a_m/V^a_m

    # get F^a_m
    F_a_m = np.nansum(fm_data,axis=0)

    # compute V^a_m from W^a_m
    L = N_fft*pw_A
    R = R_avg
    W_a_m = np.nansum(wm_data,axis=0)
    V_a_m = calculate_estnorm(W_a_m,R , L)       

    # Px
    Px_a_m = F_a_m/V_a_m
                
    return Px_a_m


def compute_cov(px, weights):
    """Computes the covariance matrix using the subsampling technique

    Args:
        px: array of floats
            Px measurement in each healpix in a given z and theta bin, having shape  (Nhp, N_k)
        weights: array of floats
            Weights on the Px measurement in each healpix, same shape as px

    Returns:
        The covariance matrix
    """
    print("Computing mean Px...")
    mean_px = (px * weights).sum(axis=0)
    sum_weights = weights.sum(axis=0)
    w = sum_weights > 0.
    mean_px[w] /= sum_weights[w]


    meanless_px_times_weight = weights * (px - mean_px)

    print("Computing subsampling cov...")

    covariance = meanless_px_times_weight.T.dot(meanless_px_times_weight)

    sum_weights_squared = sum_weights * sum_weights[:, None]

    w = sum_weights_squared > 0.

    # covariance estimator C^_ij
    covariance[w] /= sum_weights_squared[w]

    num = np.matmul((weights**2).T,weights) + np.matmul(weights.T, weights**2)
    correction_factor = 1+ np.matmul(weights.T, weights) - num/np.matmul(weights.T, weights)

    #print(np.matmul(weights.T, weights),num)
    #print("Correction factor:", correction_factor)

    # True covariance C_ij
    #covariance /= correction_factor
    #print("true Covariance:", covariance)
    return mean_px, covariance

"""
def calculate_window_matrix(W, R, L):
    '''
    W (np.ndarray): average of (w1) conj(w2) where w1 and w2 are FFT of original weights per skewer
    R (np.ndarray): vector length N, resolution in Fourier space
    L (float): physical length of skewers (e.g., in Angstroms)
    Returns:
    window_matrix (np.ndarray): window matrix to be convolved with pure theory
    estnorm (np.ndarray): vector length N, to be multiplied by every P1D mode of the measurement
    '''
    R2 = R.real**2 + R.imag**2
    denom = np.absolute(np.fft.ifft(np.fft.fft(W)* np.fft.fft(R2)))
    estnorm = np.absolute(L/denom)
    N = estnorm.size
    window_matrix = np.zeros((N,N))
    for m in range(N):
        for n in range(N):
            window_matrix[m,n] = W[m-n]*R2[n] / denom[m]
    return window_matrix, estnorm
"""

