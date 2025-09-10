# Functions related to binning and averaging that can be used as libraries across cupix

import numpy as np
import h5py

def bin_func_k(k_arr,k_fund,k_bins_ratio,max_k,k_max_ratio,bin_func_type):
    '''
    Generates the binning function 

    Inputs:
    k_arr (np.1darray) : array of k values (1D) from FFT
    k_fund (float): fundamental frequency of the FFT 
    k_bins_ratio (int): ratio of original k-bin width to desired k-bin width 
    k_max_ratio (int): ratio of maximum k to k_max of data to stop rebinning 
    bin_func_type (str): type of binning function ('top_hat' supported for now) 
    
    Returns:
    B_M_m (np.ndarray): binning function matrix (N_k_rebin, N_k)

    '''
    dk_bin=k_fund*k_bins_ratio
    print('Computing B^M_m...')
    print('dk =',dk_bin)

    # stop roughly at 1/4 of the Nyquist frequency for now (equivalent to rebinning 4 pixels)
    k_max= max_k/k_max_ratio

    print('k < ',k_max)

    k_edges=np.arange(0.01*dk_bin,k_max+dk_bin,dk_bin)
    
    Nk=k_edges.size-1
    N_fft = len(k_arr)

    #define bin function
    B_M_m=np.zeros([Nk,N_fft]) # includes negative k values
    if bin_func_type == 'top_hat':
        for i in range(Nk):
            inbin=(abs(k_arr)>=k_edges[i]) & (abs(k_arr)<k_edges[i+1]) # left closed, right open edges
            B_M_m[i,inbin]=1
    k_M =0.5*(k_edges[:-1] + k_edges[1:])        
    print('Done')
    print('shape of B_M_m is ', np.shape(B_M_m))
    

    #return B_M_m,k_M
    return B_M_m, k_edges

def bin_func_theta(theta_bin_min, theta_bin_max, theta_bins_ratio,bin_func_type):
    """
    Rebin theta in log-space by merging original a-bins into N_A coarse bins.
    Args:
        theta_bin_min (float) : (N_a,), array of min theta in each original bin
        theta_bin_max  (float) : (N_a,), must have same length as theta_bin_min
        theta_bins_ratio (float): ratio of original theta-bin width to desired theta-bin width
        bin_func_type (str): type of binning function ('top_hat' supported for now)
    Returns:
      B_A_a               (N_A, N_a)
      theta_min_rebin     (N_A,)
      theta_max_rebin     (N_A,)
    """
    theta_min = np.asarray(theta_bin_min, float)
    theta_max = np.asarray(theta_bin_max, float)
    N_a = len(theta_min)
    assert len(theta_max) == N_a, "theta_min/max length mismatch"
    print('number of fine theta bins:', N_a)
    # How many coarse bins?
    downsize = int(theta_bins_ratio)
    N_A = N_a // downsize  
    print('number of theta bins:', N_A)
    if N_A < 1:
        raise ValueError("N_A < 1; choose a smaller downsize/theta_bins_ratio.")

    # Use geometric centers for classification (since we are binning in log space)
    c_a = np.sqrt(theta_min * theta_max)  # (N_a,)
    
    epsilon = 1e-5 #* (c_a_max + c_a_min)/2  # small padding to avoid underflow/overflow
    #print('min and max of theta centers:', c_a.min(), c_a.max())
    #print('epsilon:', epsilon)
    lo, hi = c_a.min()-epsilon, c_a.max()+epsilon  # add a bit of padding to the high and low ends to avoid underflow/overflow
    
    theta_edges = np.concatenate([theta_min, [theta_max[-1]]])  # (N_a+1,)

    # Log-spaced coarse edges over the full range of centers
    edges_A = theta_edges[::downsize] #np.logspace(np.log10(lo), np.log10(hi), N_A + 1)
    # set an offset to avoid underflow/overflow
   
    edges_A[0] = edges_A[0]-epsilon
    edges_A[-1] =edges_A[-1]+epsilon
    print('bins:', c_a)
    print(edges_A)
    
    
    # Assign each original bin 'a' to a coarse bin 'A'
    A_idx = np.digitize(c_a, edges_A, right=False) - 1   # in [0, N_A-1]
    
    underflows = np.sum(A_idx < 0)  # digitize returns A_idx as -1 for theta values smaller than the first edge 
    overflows  = np.sum(A_idx >= N_A) # digitize returns A_idx as len(edges)-1 = N_A for theta values larger than the last edge

    if underflows or overflows:
        print(f"[bin_func_theta] WARNING: {underflows} underflows, {overflows} overflows before clipping")
        ask = input("The coarse bins do not cover the theta bin range, do you want to combine the overflow/underflow into the last/first bins? (y/n): ")
        if ask.lower() == 'y':
            A_idx = np.clip(A_idx, 0, N_A - 1) # if enabled, the underflow/overflow bins will be assigned to the first/last coarse bin

    B_A_a = np.zeros((N_A, N_a), float)
    if bin_func_type == 'top_hat':
        for a, A in enumerate(A_idx):
            B_A_a[A, a] += 1.0

    #theta_edges_rebin = theta_edges[::downsize]
    theta_min_rebin = edges_A[:-1]  # left edge of previous bin is right edge of next bin
    theta_max_rebin =  edges_A[1:] # right edge of previous bin is left edge of next bin
    print('shape of B_A_a is ', np.shape(B_A_a))

    return B_A_a, theta_min_rebin, theta_max_rebin


def rebin_k(F_zh_am,B_M_m, healpix):
    '''
    Rebins the k dimension given a binning function. Need to specify if the input has healpix dimension or not.

    '''
    if healpix==True:
        # numerator: (Z,H,a,M)
        num_k = np.einsum('zham,Mm->zhaM', F_zh_am, B_M_m, optimize=True)
        # denom: (M,) -> (1,1,1,M) for broadcasting
        den_k = np.einsum('Mm->M', B_M_m)[None, None, None, :]
    else:
        # numerator: (Z,a,M)
        num_k = np.einsum('zam,Mm->zaM', F_zh_am, B_M_m, optimize=True)
        
        # denom: (M,) -> (1,1,M) for broadcasting
        den_k = np.einsum('Mm->M', B_M_m)[None,None,:]
        

    F_zh_aM = np.divide(num_k, den_k, out=np.zeros_like(num_k), where=den_k>0)
    

    return F_zh_aM

def rebin_theta(F_zh_aM,B_A_a,healpix=None):
    '''
    Rebins the theta dimension given a binning function. Need to specify if the input has healpix dimension or not.
    '''

    if healpix:
        # numerator: (Z,H,A,M)
        num_k = np.einsum('zahM,Aa->zAhM', F_zh_aM, B_A_a, optimize=True)
        # denom: (M,) -> (1,1,1,M) for broadcasting
        den_k = np.einsum('Aa->A', B_A_a)[None, :, None, None]
    else:
        # numerator: (Z,A,M)
        num_k = np.einsum('zaM,Aa->zAM', F_zh_aM, B_A_a, optimize=True)
        # denom: (M,) -> (1,1,M) for broadcasting
        den_k = np.einsum('Aa->A', B_A_a)[None, :, None]

    

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

def calculate_V_zh_AM(W_zh_AM,R2,L):
    '''
    Calculates the normalisation from given W, R

    '''
    denom = np.absolute(np.fft.ifft(np.fft.fft(W_zh_AM,axis=-1)* np.fft.fft(R2)))    
    estnorm = np.absolute(denom/L)
    print(np.shape(estnorm),'denom')
    return estnorm  

def average_px(F_zh_AM,W_zh_am,R2_m,L,B_A_a,B_M_m):
    '''
    Average of Px over combined F, V of all healpix pixels

    '''

    print('Computing average...')
    W_z_am = np.einsum('zAhM->zAM',W_zh_am)

    V_z_am = calculate_V_zh_AM(W_z_am,R2_m,L)
    
    V_z_aM = rebin_k(V_z_am,B_M_m,healpix=False)
    print('done')
    V_z_AM = rebin_theta(V_z_aM,B_A_a,healpix=False)

    num_k = np.einsum('zAhM->zAM', F_zh_AM)

    P_z_AM = np.divide(num_k, V_z_AM, out=np.zeros_like(num_k), where=V_z_AM>0)
    
    print('Shape of average Px is',np.shape(P_z_AM))


    return P_z_AM, V_z_AM

def compute_covariance(F_zh_AM,V_zh_AM):
    '''
    Covariance of Px over all healpix pixels, returns healpix average and covariance matrix
    '''
    V_z_AM = np.einsum('zAhM->zAM',V_zh_AM)
    
    P_zh_AM = np.divide(F_zh_AM,V_zh_AM,out=np.zeros_like(F_zh_AM),where=V_zh_AM>0)#F_zh_AM/V_zh_AM
    #print('per healpix Px shape is', np.shape(P_zh_AM))

    P_z_AM = np.divide(np.einsum('zAhM,zAhM->zAM',V_zh_AM,P_zh_AM),V_z_AM,out=np.zeros_like(V_z_AM),where=V_z_AM>0)

    #P_avg = np.divide(avg_num,avg_denom,out=np.zeros_like(cov_num),where=denom>0)

    P_diff = P_zh_AM - P_z_AM[:,:,None,:]

    cov_num = np.einsum('zAhM,zAhN,zAhM,zAhN->zAMN',V_zh_AM,V_zh_AM,P_diff,P_diff) 
    denom =  V_z_AM[:, :, :, None] * V_z_AM[:, :, None, :]

    #print(np.shape(cov_num))
    #print(np.shape(denom))

    C_z_AMN = np.divide(cov_num,denom,out=np.zeros_like(cov_num),where=denom>0)
    #print(np.shape(C_z_AMN))
    print('Shape of covariance matrix is',np.shape(C_z_AMN))
    return C_z_AMN, P_z_AM

def calculate_window_matrix(W_z_AM, R2_m):
    '''
    W (np.ndarray): average of (w1) conj(w2) where w1 and w2 are FFT of original weights per skewer
    R (np.ndarray): vector length N, resolution in Fourier space
    L (float): physical length of skewers (e.g., in Angstroms)
    Returns:
    window_matrix (np.ndarray): window matrix to be convolved with pure theory
    estnorm (np.ndarray): vector length N, to be multiplied by every P1D mode of the measurement
    '''
   
    #R_z_AM = np.einsum('zAhM->zAM',R_zh_AM)
    
    #R2 = R_z_AM.real**2 + R_z_AM.imag**2
    denom = np.absolute(np.fft.ifft(np.fft.fft(W_z_AM,axis=-1)* np.fft.fft(R2_m[None,None,:])))
  
    Nz, Ntheta_A, Nk = W_z_AM.shape
    print(Nz, Ntheta_A,Nk) 
    window_matrix = np.zeros((Nz,Ntheta_A,Nk,Nk))
    for m in range(Nk):
        for n in range(Nk):
            #window_matrix[:,:,m,n] = W_z_AM[:,:,m-n]*R2[:,:,n] / denom[:,:,m]
            num_k = W_z_AM[:,:,m-n]*R2_m[None,None,n]
            window_matrix[:,:,m,n] = np.divide(num_k,denom[:,:,m], out=np.zeros_like(num_k), where=denom[:,:,m]>0)
    
    print('Shape of window matrix is',np.shape(window_matrix))
    
    return window_matrix

def bin_window(U_z_amn,B_M_m,W_z_am,R2_m,L):
    '''
    Bins window function in k dimension

    '''
    
    V_z_am = calculate_V_zh_AM(W_z_am,R2_m,L)

    V_z_aM = rebin_k(V_z_am,B_M_m,healpix=False) #calculate_V_zh_AM(W_z_aM,R2_M,L)

    # Numerator: sum over m
    num = np.einsum('Mm,zam,zamn->zaMn', B_M_m, V_z_am, U_z_amn, optimize=True)

    # Safe divide, broadcasting over n
    U_z_aMn = np.divide(num, V_z_aM[..., None]*np.einsum('Mm->M', B_M_m)[...,None], out=np.zeros_like(num), where=V_z_aM[..., None] > 0)
    print('Shape of window matrix after rebinning in k is ',np.shape(U_z_aMn))

    return U_z_aMn, V_z_aM, V_z_am

def get_sum_over_healpix(W_zh_am):
    return np.einsum('zahm->zam',W_zh_am)

def save_to_hdf5(filename,P_Z_AM,C_Z_AMN,U_Z_aMn,B_A_a,V_Z_aM,k_m,k_M_edges,theta_bin_min,theta_bin_max,theta_min_A,theta_max_A,N_fft,L_fft,zbin_centers):

    with h5py.File(filename, 'w') as f:  
        # Save metadata as attributes
        g = f.create_group('metadata')
        g.attrs['k_m'] = k_m
        g.attrs['k_M_edges'] = k_M_edges
        g.attrs['N_fft'] = N_fft
        g.attrs['L_fft'] = L_fft
        g.attrs['z_centers'] = zbin_centers
        g.attrs['theta_min_a'] = theta_bin_min
        g.attrs['theta_max_a'] = theta_bin_max
        g.attrs['theta_min_A'] = theta_min_A
        g.attrs['theta_max_A'] = theta_max_A
        g.create_dataset('B_A_a',data=B_A_a)

        
        
        # Save rebinned Px and covariance for each redshift and coarse theta bin
        gP = f.create_group('P_Z_AM')
        gC = f.create_group('C_Z_AMN')
        gU = f.create_group('U_Z_aMn')
        gV = f.create_group('V_Z_aM')

        Nz, N_A, NK = P_Z_AM.shape
        for i in range(Nz):
            z_str = f'z_{i}'
            for j in range(N_A):
                theta_str = f'theta_rebin_{j}'
                ds_name = f'{z_str}/{theta_str}'
                gP.create_dataset(ds_name, data=P_Z_AM[i,j,:], compression="gzip", compression_opts=4)
                gC.create_dataset(ds_name, data=C_Z_AMN[i,j,:,:], compression="gzip", compression_opts=4)

        Nz, N_a, NK, Nk = U_Z_aMn.shape 
        for i in range(Nz):
            z_str = f'z_{i}'
            for j in range(N_a):
                theta_str = f'theta_{j}'
                ds_name = f'{z_str}/{theta_str}'
                gU.create_dataset(ds_name, data=U_Z_aMn[i,j,:,:], compression="gzip", compression_opts=4)
                gV.create_dataset(ds_name, data=V_Z_aM[i,j,:], compression="gzip", compression_opts=4)

        
        

        #f.create_dataset("P_Z_AM", data=P_z_AM, compression="gzip", compression_opts=4)
        #f.create_dataset("C_Z_AMN", data=C_z_AMN, compression="gzip", compression_opts=4)
        #f.create_dataset("U_Z_aMn", data=U_z_aMn, compression="gzip", compression_opts=4)
        #f.create_dataset("V_Z_aM", data=V_z_aM, compression="gzip", compression_opts=4)
        #f.create_dataset("V_Z_am", data=V_z_am, compression="gzip", compression_opts=4)

    print(f"Output saved to {filename}")

    return None





