import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
"""
IIT Navigation Lab Code:
Find the Gaussian Distribution to Overbound the Data

Based On Previous Work By: Austin Gonzalez, Jaymin Patel
Modified By: Yawei Zhai  07/28/2017
Modified to Python By: Yulong Sun  06/15/2024

Output: matrix of overbound bias(first row) and its corresponding sigma(second row)

Inputs:
     data: input data, must be a vector
     dataname: must be string
 
     flag_bias_zero: 0(off) or 1(on), force the overbound bias to be zero
     flag_bias_mean: 0(off) or 1(on), force the overbound bias to be the mean of input data
     flag_bias_median: 0(off) or 1(on), force the overbound bias to be the median of input data
     flag_bias_optimal: 0(off) or 1(on), use the "optimal" overbound bias

     ------------------------- Rationale of Determining the "Optimal" Bias -----------------------------
     1. Defining a range of bias to investigate (starting from the mean of data), finding their 
         corresponding overbound sigma values. The relationship of bias and sigma can be ploted by 
         turning on flag_plot_b_sig.                              
     2. Evaluate the slope (sigma decreasement over bias increasement) for each bias value, find
         out the index of the bias that result in maximum slope.
     3. Use the bias value at (index + 1), and its corresponding sigma as the final option.
     --------------------------------------------------------------------------------------------------

     flag_remove: 0(off) or 1(on), remove data with small probabilities, i.e., threshold
     Threshold: the probability threshold, the data point below this threshold will be removed, works only when flag_remove is on. For example, if
     the threshold is set to be 1e-5, the data points that occurs less than this value will be removed.

     flag_plot_b_sig: 0(off) or 1(on), make plot of overbound bias versus sigma
"""
def paired_overbounding(data, dataname, flag_bias_zero, flag_bias_mean, flag_bias_median, flag_bias_optimal, flag_remove, Threshold, flag_plot_b_sig):
    # empirical CDF evaluated at each data point  
    prob_all, x_all = ecdf(data)
    prob = prob_all
    x = x_all

    # take out data smaller than a probability
    if flag_remove:
        ind_remove = np.where((prob < Threshold) | (prob > (1-Threshold)))
        x = np.delete(x, ind_remove)
        prob = np.delete(prob, ind_remove)

    data = data[np.abs(data) < 999]
    # raw data statistics
    mu_sample = np.nanmean(data)
    b_median = np.nanmedian(data)
    sigma_sample = np.nanstd(data, ddof=1)

    # define the ranges of the overbound bias and sigma
    bias_int = sigma_sample ##/10
    sigma_int = sigma_sample ##/10

    # IMPORTANT: Please note that sometimes it's necessary to modify the range of the 'bias_vec/sigma_vec' variable.
    num_steps = int((abs(mu_sample) + 100 * sigma_sample - abs(mu_sample)) / bias_int) + 1
    bias_vec = np.linspace(abs(mu_sample), abs(mu_sample) + 10000 * sigma_sample, num_steps)
    bias_vec_all = np.append(bias_vec, [0, b_median])

    num_steps = int((10 * sigma_sample - sigma_sample) / sigma_int) + 1
    sigma_vec = np.linspace(sigma_sample, 10 * sigma_sample, num_steps)

    # initialize the overbound sigma vector over all bias value
    ob_sigma_vec_all = np.zeros(len(bias_vec_all))

    # find all the overbound sigma values associated with all the biases
    for i in range(len(bias_vec_all)):
        mu_low_bd = -abs(bias_vec_all[i])
        mu_up_bd = abs(bias_vec_all[i])

        # initialize overbound sigma matrix at each single data point
        ob_sigma_tmp = np.full(len(x), np.inf)

        # find the overbound sigma value for each data point
        for j in range(len(prob)):
            # find overbound sigma index
            if prob[j] < 0.5:
                ind = np.argmin(np.abs(x[j] - norm.ppf(prob[j], mu_low_bd, sigma_vec)))
                if (x[j] - norm.ppf(prob[j], mu_low_bd, sigma_vec[ind])) > 0:
                    ind_ob_sigma = ind
                else:
                    ind_ob_sigma = ind + 1
            else:
                ind = np.argmin(np.abs(x[j] - norm.ppf(prob[j], mu_up_bd, sigma_vec)))
                if (x[j] - norm.ppf(prob[j], mu_up_bd, sigma_vec[ind])) > 0:
                    ind_ob_sigma = ind + 1
                else:
                    ind_ob_sigma = ind
        
            if ind_ob_sigma >= len(sigma_vec):
                continue
            ob_sigma_tmp[j] = sigma_vec[ind_ob_sigma]

        ob_sigma_vec_all[i] = np.max(ob_sigma_tmp)

    ob_sigma_vec = ob_sigma_vec_all[:-2]
    output = [bias_vec_all, ob_sigma_vec_all]


    # determine which pair of overbound bias and sigma to pick
    deriv_vec = np.diff(ob_sigma_vec) / np.diff(bias_vec)
    if np.isnan(deriv_vec).any():
        print('----------- WARNING: need to increase the predefined bias and sigma range!!!!!! ----------')
        return
    ind_deriv = np.where(abs(deriv_vec) <= 4 * (sigma_int / bias_int))
    if ind_deriv[0].size != 0:
        ind_ob = ind_deriv[0][0]
    else:
        ind_ob = np.where(ind_deriv == np.inf)[0][0] + 2

    bias_zero = 0
    sigma_zero = ob_sigma_vec_all[-2]
    bias_mean = bias_vec_all[0]
    sigma_mean = ob_sigma_vec_all[0]
    bias_median = bias_vec_all[-1]
    sigma_median = ob_sigma_vec_all[-1]

    bias_ob = bias_vec[ind_ob + 1]
    sigma_ob = ob_sigma_vec[ind_ob + 1]

    if flag_bias_zero:
        print(f'Forcing the bias to be 0, the overbound sigma for the input data is: {sigma_zero}')
        make_overbound_plots(prob_all, bias_zero, x, sigma_sample, sigma_zero, x_all, mu_sample, dataname, data)

    if flag_bias_mean:
        print(f'Forcing the bias to be the mean: {bias_mean}, the overbound sigma for the input data is: {sigma_mean}')
        make_overbound_plots(prob_all, bias_mean, x, sigma_sample, sigma_mean, x_all, mu_sample, dataname, data)

    if flag_bias_median:
        print(f'Forcing the bias to be the median: {bias_median}, the overbound sigma for the input data is: {sigma_median}')
        make_overbound_plots(prob_all, bias_ob, x, sigma_sample, sigma_ob, x_all, mu_sample, dataname, data)

    if flag_bias_optimal:
        print(f'The following Gaussian distribution can overbound the input data: bias = {bias_ob}, sigma = {sigma_ob}')
        make_overbound_plots(prob_all, bias_ob, x, sigma_sample, sigma_ob, x_all, mu_sample, dataname, data)

    if flag_plot_b_sig:
        plt.figure()
        plt.plot(bias_vec, ob_sigma_vec, linewidth=2)
        plt.xlabel('Overbound Bias')
        plt.ylabel('Overbound Sigma')
        plt.xlim([abs(mu_sample), abs(mu_sample) + sigma_sample / 10])
        plt.show()

    if flag_bias_optimal:
        return bias_ob, sigma_ob
    else:
        return output

def make_overbound_plots(prob_all, bias_ob, x, sigma_sample, sigma_ob, x_all, mu_sample, dataname, data):
    # # folded CDF (actual data)
    # prob_all[prob_all > .5] = 1 - prob_all[prob_all > .5]
    # if -abs(bias_ob) > min(x):
    #     xplotn = np.flip(np.arange(-abs(bias_ob),min(x), -0.0001))
    # else:
    #     xplotn = np.flip(np.arange(min(x), -abs(bias_ob), -0.0001))
    # xplotp = np.arange(abs(bias_ob), max(x), 0.0001)

    # xplot = np.concatenate((xplotn, xplotp))
    # ob_cdf_sample = np.concatenate((norm.cdf(xplotn, -abs(bias_ob), sigma_sample), 1 - norm.cdf(xplotp, abs(bias_ob), sigma_sample)))

    # if -abs(bias_ob) > 2 *  min(x):
    #     xplotn_2 = np.flip(np.arange(-abs(bias_ob), 2 * min(x), -0.0001))
    # else:
    #     xplotn_2 = np.flip(np.arange(2 * min(x), -abs(bias_ob), -0.0001))
    # xplotp_2 = np.arange(abs(bias_ob), 2 * max(x), 0.0001)
    # xplot_2 = np.concatenate((xplotn_2, xplotp_2))
    # ob_cdf_ob = np.concatenate((norm.cdf(xplotn_2, -abs(bias_ob), sigma_ob), 1 - norm.cdf(xplotp_2, abs(bias_ob), sigma_ob)))
    
    # folded CDF (actual data) - symmetrical
    prob_all[prob_all > .5] = 1 - prob_all[prob_all > .5]
    x_max = max(abs(x))
    bias_abs = abs(bias_ob)
    
    xplotn = np.flip(np.arange(-bias_abs, -x_max, -0.0001))
    xplotp = np.arange(bias_abs, x_max, 0.0001)
    xplot = np.concatenate((xplotn, xplotp))
    ob_cdf_sample = np.concatenate((norm.cdf(xplotn, -bias_abs, sigma_sample), 1 - norm.cdf(xplotp, bias_abs, sigma_sample)))
    
    xplotn_2 = np.flip(np.arange(-bias_abs, -2 * x_max, -0.0001))
    xplotp_2 = np.arange(bias_abs, 2 * x_max, 0.0001)
    xplot_2 = np.concatenate((xplotn_2, xplotp_2))
    ob_cdf_ob = np.concatenate((norm.cdf(xplotn_2, -bias_abs, sigma_ob), 1 - norm.cdf(xplotp_2, bias_abs, sigma_ob)))
    
    # make plot
    plt.figure()
    plt.semilogy(x_all, prob_all, 'b*', markersize=2)
    plt.semilogy(xplot, ob_cdf_sample, 'g')
    plt.semilogy(xplot_2, ob_cdf_ob, 'r', linewidth=2)
    plt.legend([f'{dataname} - Data Points: {len(data)} \n bias = {mu_sample:.6f}', f'$sigma_{{SAMPLE}}$ = {sigma_sample:.4f} units, $mean_{{OB}}$ = ({-abs(bias_ob):.4f}, {abs(bias_ob):.4f}) units', f'$sig_{{OB}}$ = {sigma_ob:.4f} units, $mean_{{OB}}$ = ({-abs(bias_ob):.4f}, {abs(bias_ob):.4f}) units'], loc='lower left')
    plt.ylabel("Folded CDF")
    plt.xlabel("Error Magnitude")
    plt.title(dataname)
    plt.show()


def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    unique_x = np.unique(data)

    # y-data for the ECDF: y
    y = np.array([np.sum(data <= x_val) for x_val in unique_x]) / n

    return np.concatenate([[0], y]), np.concatenate([[unique_x[0]], unique_x])