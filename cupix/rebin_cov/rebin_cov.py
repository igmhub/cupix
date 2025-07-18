import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import sys
import h5py

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

def compute_cov(px, weights):
    """Computes the covariance matrix using the subsampling technique

    Args:
        px: array of floats
            Px measurement in each healpix
        weights: array of floats
            Weights on the Px measurement in each healpix

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

    return mean_px, covariance

def bin_power(k_arr,Px_k,bin_info,bin_func='top_hat'):
    """ Function to bin power spectrum 

    Args:
        k_arr: array of floats
            FFT grid wavenumbers
        Px_k: array of floats
            Px measurement in each healpix
        bin_info: dictionary
            dictionary containing number of k bins 'Nk', edges of k bins 'k_edges', maximum k 'k_max'
        bin_func: string
            binnning function, chose from 'top_hat' 
            
    Returns:
        k bins, Px in k bins
        
    """
    
    Nk = bin_info['Nk']
    k_edges = bin_info['k_edges']
    k_max = bin_info['k_max']
    N_fft = bin_info['N_fft']
    
    #define bin function
    B_m=np.zeros([Nk,N_fft]) # includes negative k values
    if bin_func == 'top_hat':
        for i in range(Nk):
            inbin=(abs(k_arr)>k_edges[i]) & (abs(k_arr)<k_edges[i+1])
            B_m[i,inbin]=1

    k_A=np.zeros(Nk)
    for i in range(Nk):
        k_A[i]=np.sum(B_m[i]*abs(k_arr))/np.sum(B_m[i])
        if bin_func == 'top_hat':
            assert np.allclose(k_A[i],np.mean(abs(k_arr)[B_m[i]==1])) 
        
    px_Theta_A=np.zeros(Nk)  
    for i in range(Nk):
        px_Theta_A[i]=np.sum(B_m[i]*Px_k)/np.sum(B_m[i]) #/p1d_A_A[i] 
        
    return k_A, px_Theta_A


