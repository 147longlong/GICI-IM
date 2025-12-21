import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

"""
Find the Gaussian Distribution to Overbound the Data

Output: sigma_ob, the overbound sigma for the input data

Inputs:
     data: input data, must be a vector
     dataname: must be string
     force_b_zeror: make the mean of overbounding distribution is 0
     plot_flag: 
"""
def fcdf_overbounding(data, dataname, force_b_zero=False, plot_flag=False, flag_remove=False, Threshold=1e-5):
    p1 = np.arange(-5, math.log10(0.5), 0.1)
    p2 = 10 ** p1
    bound_p_left = np.zeros(len(p2))
    bound_p_right = np.zeros(len(p2))

    finite_data = data[np.isfinite(data)]
    
    # Remove data based on threshold if flag is set
    if flag_remove:
        # Calculate ECDF first to determine probabilities
        n = len(finite_data)
        sorted_data = np.sort(finite_data)
        # Probabilities for each data point
        probs = np.arange(1, n + 1) / n
        
        # Keep data within [Threshold, 1-Threshold]
        mask = (probs >= Threshold) & (probs <= (1 - Threshold))
        finite_data = sorted_data[mask]
        
        if len(finite_data) == 0:
            print(f"Warning: All data removed with threshold {Threshold}")
            return [0, 0]

    yh, xh = np.histogram(finite_data, bins=1000)
    xh = (xh[1:] + xh[:-1]) / 2
    yh = yh / len(finite_data) / np.mean(np.diff(xh))
    cdf = np.cumsum(yh) / sum(yh)

    for j in range(len(p2)):
        in_left = np.where((cdf <= 0.5) & (np.abs(cdf - p2[j]) == np.min(np.abs(cdf[cdf <= 0.5] - p2[j]))))[0]
        in_left = np.min(in_left)
        xh_left = xh[in_left]

        in_right = np.where((cdf >= 0.5) & (np.abs(1 - cdf - p2[j]) == np.min(np.abs(1 - cdf[cdf >= 0.5] - p2[j]))))[0]
        in_right = np.max(in_right)
        xh_right = xh[in_right]

        bound_p_left[j] = xh_left
        bound_p_right[j] = xh_right

    bound_left = bound_p_left
    bound_right = bound_p_right

    if force_b_zero:
        bnom = 0
    else:
        bnom = max(np.abs(bound_left[-1]), np.abs(bound_right[-1]))

    ## start overbounding
    # sigma = np.arange(0.5, 20, 0.001)
    # Adaptive sigma range based on data
    max_val = np.max(np.abs(finite_data))
    # Create a range that covers small to large values relative to data max
    # Using logspace to cover orders of magnitude efficiently
    sigma = np.logspace(np.log10(max_val/10000), np.log10(max_val*100), 5000)
    
    ovb_sigma = 0
    for i in range(len(sigma)):
        pnl = norm.cdf(bound_left, -bnom, sigma[i])
        pnr = 1 - norm.cdf(bound_right, bnom, sigma[i])
        if np.all(pnl - p2 > 0) and np.all(pnr - p2 > 0):  # overbound
            ovb_sigma = sigma[i]
            break

    if ovb_sigma == 0:
        print(f'something wrong -> too big sigma')
        return ovb_sigma

    if np.max(abs(bound_left)) > np.max(abs(bound_right)):
        bound_right = np.copy(-bound_left)
        pnr = np.copy(pnl)
    else:
        bound_left = np.copy(-bound_right)
        pnl = np.copy(pnr)

    if(plot_flag):
        plt.figure()
        yh, xh = np.histogram(finite_data, bins=1000)
        xh = (xh[1:] + xh[:-1]) / 2
        yh = yh / len(finite_data) / np.mean(np.diff(xh))
        cdf = np.cumsum(yh) / sum(yh)
        index = np.where(cdf >= 0.5)[0]
        cdf[index] = 1 - cdf[index]
        plt.semilogy(xh, cdf, 'b*', markersize=1, label='actual data')
        if np.max(abs(bound_p_left)) > np.max(abs(bound_p_right)):
            plt.semilogy(bound_left, pnl, 'r.-', label='overbound')
            plt.semilogy(bound_right, pnr, 'r-')
        else:
            plt.semilogy(bound_left, pnl, 'r-')
            plt.semilogy(bound_right, pnr, 'r.-', label='overbound')
        plt.grid(which='both', axis='both', color='gray', linestyle='dotted', alpha=0.3)
        plt.title(f'{dataname} - Data Points: {len(data)} - $sigma_{{ob}}$ = {ovb_sigma:.4f}')
        plt.ylim([1e-5, 1])
        plt.xlabel('Error Magnitude')
        plt.ylabel('Folded CDF')
        plt.legend()
        plt.show()

    return [ovb_sigma, bnom]