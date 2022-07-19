#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 10:02:04 2022

@author: Marin Govorcin
"""
import numpy as np


def midas(dates, disp, steps=None, tolerance=0.001, print_msg=False, display=False):
    '''
    Python version of Blewitt's MIDAS estimator, original code is written in f90
    based on numpy module.
    Median Interannual Difference Adjusted for Skewness (MIDAS) is a customized version
    of Theil-Sen estimator, insensitivity to seasonal variation and step discontinuities
    in GNSS timeseries (Blewitt 2016).

    Parameters
    ----------
    dates : datetime list/ 1D array  - timeseries dates [datetime.datetime(2022,2,2,0,0)]
    disp  : 1D/2D array              - timeseries displacements, in N-S, E-W, and Up-Down
                                        np.array((n_obs, n_components))
                                        where n_components are  d_e, d_n, d_u

    steps : datetime list /1D array  - timeseries dates of steps [datetime.datetime(2022,2,2,0,0)]
                                        NOTE : step valid if the distance between
                                        gnss site and earthquake epicentar is less
                                        than 10^(Magnitude /2 - 0.79)
                                        eqstep_check = lambda M, D: np.power(10, M/2 - 0.79) < D
    tolerance : float                - Tolerance around 1yr separation for data pair selection
                                       [1yr - tolerance > pair < 1yr + tolerance], optional
                                       The default is 0.001.

    Returns
    -------
    vel       :   1D/2D array                 - MIDAS velocity (slope)
    vel_std   :   1D/2D array                 - MIDAS velocity uncertainty
    intercept :   1D/2D array                 - MIDAS intercept
    residuals :   1D/2D array                 - MIDAS residuals

    ----------
    Reference: Blewitt et al. (2016): MIDAS robust trend estimator for accurate GPS
               station velocities without step detection,
               J. Geophys. Res. Solid Earth, 121, 2054–2068, doi:10.1002/2015JB012552.

    NOTES : Authors state tyat MIDAS does not necessarily represent strain accumulation
            part of seismic cycle only

            TODO : add option for AVR (Allen Variance of the Rate) estimation of MIDAS
            velocity uncertainty, to provide more accurate estimate
            https://github.com/langbein-usgs/midas_vr

            Hackl et al. (2011): Estimation of velocity uncertainties from GPS time series:
              Examples from the analysis of the South African TrigNet network,
              J. Geophys. Res. Solid Earth, 116 (B11),  https://doi.org/10.1029/2010JB008142

            Example of AVR code:
            https://github.com/kmaterna/GNSS_TimeSeries_Viewers/blob/master/GPS_TOOLS/lssq_model_errors.py

    '''
    if disp.ndim==1:
        disp = disp[:, np.newaxis]

     # Adjust zero-crossing to first epoch
    adjust_disp = disp - disp[0, :]

    # Convert the format of dates array to datetime64
    dates = dates.astype('datetime64[s]')

    #  Data pair slection based on Relaxed Interannual Theil-Sen Estimator
    #  Runs twice - forward and backward
    f_ix, b_ix = select_pairs(dates, steps=steps, tolerance=tolerance, print_msg=print_msg)

    #join the selection
    pairs_ix = np.concatenate([f_ix, b_ix], axis=1)

    #Calculate the slopes
    vp = (adjust_disp[pairs_ix[1,:], :] - adjust_disp[pairs_ix[0,:], :]) \
        / date_difference_ydec(dates[pairs_ix[0,:]], dates[pairs_ix[1,:]])

    ### Step 1
    ## Median of the distribution
    v_dot = np.nanmedian(vp, axis=0, keepdims=True)
    # Scaled Median Absolute Deviation (MAD)
    sigma = 1.4826 * np.nanmedian(np.absolute(vp - v_dot), axis=0, keepdims=True)

    # Step 2 {q = p} for all |v(p)-v_dot|
    # Remove outliers beyond 2 standard deviation
    index = np.absolute((vp - v_dot)) >= 2*sigma
    vp[index] = np.nan

    #Repeat Median and MAD estimation
    vel = np.nanmedian(vp, axis=0, keepdims=True)
    sigma = 1.4826 * np.nanmedian(np.absolute(vp - vel), axis=0, keepdims=True)

    # Velocity uncertainty
    N = (~np.isnan(vp)).sum(axis=0, keepdims=True) // 4
    vel_std = 3 * np.sqrt(np.pi / 2)  * (sigma / np.sqrt(N))

    #Intercept and residuals
    intercepts = adjust_disp  - vel * date_difference_ydec(dates[0], dates)
    intercept = (np.nanmedian(intercepts, axis=0, keepdims=True)) + disp[0, :]
    residuals = intercepts - intercept

    if display is True:
        plot_midas(dates, disp*1000, vel*1000, vel_std*1000, intercept*1000)

    return vel, vel_std, intercept, residuals

def select_pairs(dates, steps=None, tolerance=0.001, print_msg=False):
    '''
    MIDAS Data pair selection - runs twice, first, in time order (“forward”)
    and secondly, in reverse time order (“backward”), if they exist, the priority
    is given to pairs 1 year apart, but if there is no matching pair 1 year apart,
    the next available data point that has not yet been matched is selected

    Parameters
    ----------
    dates : datetime list/ 1D array  - timeseries dates [datetime.datetime(2022,2,2)]
    steps : datetime list/ 1D array  - timeseries dates of steps [datetime.datetime(2022,2,2)],
    tolerance : float                - Tolerance around 1yr separation for data pair selection
                                       [1yr - tolerance > pair < 1yr + tolerance], optional
                                       The default is 0.001.

    Returns
    -------
    forward_selected_pairs   : 1D array - forward data pair selection
    backward_selected_pairs  : 1D array - backward data pair selection

    '''

    n_dates = len(dates)

    # Convert dates array dtype to datetime64[s]
    if dates.dtype != '<M8[s]':
        dates = dates.astype('datetime64[s]')

    # Add second axis if does not exist, needed for np.repeat
    if dates.ndim == 1:
        dates = dates[:, np.newaxis]

    if isinstance(steps, list):
        steps = np.array(steps, dtype='datetime64[s]')

    for _, direction in enumerate(['forward', 'backward']):
        if print_msg:
            print(f'RUNNING : {direction} data pair selection\n')

        # flip the dates array if the selection goes backwards
        if direction == 'backward':
            dates = dates[::-1]

        # Matrix of dates along x axis
        date1 = np.repeat(dates, n_dates, axis=1)

        # Matrix of dates along y axis
        date2 = np.repeat(dates.T, n_dates, axis=0)

        #Create time-separation matrix
        date_diff = np.abs(np.triu(np.float64(date2 - date1) / 86400 / 365.25))

        ## Steps - do not select pairs that span or include the step epoch
        ## if step is not find in date list, select the closest date
        if steps is not None:
            isteps = [np.where(np.abs(dates - step) == np.min(np.abs(dates - step)))[0][-1]
                          for step in steps]
            isteps = np.sort(np.append(isteps, 0))

            #mask the values
            for i in range(len(isteps)-1):
                date_diff[isteps[i]:isteps[i+1], isteps[i+1]:] = 0.0

        ## The Interannual Theil-Sen Estimator
        pairs_ix = np.where((date_diff > (1.0 - tolerance)) & (date_diff < (1.0 + tolerance)))

        ## Repeat to find skiped pairs, match with the next closes pair
        dates_ix = np.linspace(0, n_dates-1, n_dates-1, dtype=np.int32)

        skipped_pairs = dates_ix[~np.isin(dates_ix, pairs_ix[0])]
        skipped_date_dif = date_diff[skipped_pairs, :]

        #Remove the pairs with date separation under 1 yr
        skipped_date_dif[skipped_date_dif < (1.0 - tolerance)] = 0.0
        #Select the first pair after 1 yr
        skipped_ix = np.where(skipped_date_dif > (1.0 - tolerance))
        _, index = np.unique(skipped_ix[0], return_index=True)
        #Create the index tuple
        skipped_ix = (skipped_pairs[skipped_ix[0][index]], skipped_ix[1][index])

        #Return the array of selected pairs
        pairs = np.sort(np.hstack([pairs_ix, skipped_ix]))

        if direction == 'backward':
            pairs = (n_dates - 1) - pairs[::-1,:]
            backward_selected_pairs = pairs
        else:
            forward_selected_pairs = pairs

    return forward_selected_pairs, backward_selected_pairs


def date_difference_ydec(date1, date2):
    '''

    Parameters
    ----------
    date1 : 1D numpy array  - array of dates (first epoch) in datetime format
    date2 : 1D numpy array  - array of dates (second epoch) in datetime format

    Raises
    ------
    ValueError - Raise error if date1 and date2 does not have same format (dtype)

    Returns
    -------
    delta_dt : 1D datetime64 array - array of time duration between two dates in
                                    decimal years
    '''

    # Convert to array if lists are given as input
    if isinstance(date1, list):
        date1 = np.array(date1, dtype='datetime64[s]')
    if isinstance(date2, list):
        date2 = np.array(date2, dtype='datetime64[s]')

    # Double check that both array have same format, if not raise error!
    if date1.dtype != date2.dtype:
        raise ValueError(f'Different dtype between date1: {date1.dtype} and date2: {date2.dtype}')
    # If numpy array has a format "object", convert it to datetime64
    if date1.dtype == 'O':
        date1 = date1.astype('datetime64[s]')
        date2 = date2.astype('datetime64[s]')

    # Find the time duration between two dates in decimal years depening on
    # format of date1/2 array
    if date1.dtype == '<M8[s]':
        return (np.float64(date2 - date1) / 86400 / 365.25).reshape(-1,1)
    elif date1.dtype == '<M8[D]':
        return (np.float64(date2 - date1) / 365.25).reshape(-1,1)

def plot_midas(dates, disp, vel, vel_std, intercept):
    '''
    Short function to plot midas results
    '''
    from matplotlib import pyplot as plt

    n_disp = vel.size
    date_dt = date_difference_ydec(np.array(dates[0]), dates)

    fig, axs = plt.subplots(n_disp, 1, figsize=(14,10), sharex=True)
    plot = [(axs[i].plot(dates, disp[:, i] - np.mean(disp[:, i]), 'b.'),
            axs[i].plot(dates, (intercept[0, i] + v * date_dt) - np.mean(intercept[0, i] + v * date_dt),
            'r-', label = f'MIDAS v {v:.2f} \u00B1 {std:.2f}'))
            for i, (v, std) in enumerate(zip(np.squeeze(vel), np.squeeze(vel_std)))]

    labels = [(axs[i].set_ylabel(txt), axs[i].legend(loc='upper left'))
               for i, txt in enumerate(['East (mm)', 'North (mm)', 'UP (mm)'])]
