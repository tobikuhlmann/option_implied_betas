import numpy as np
import pylab as pl
import pandas as pd
import matplotlib.pyplot as plt
import random
import time

import glob
import os

import datetime as dt


class OptionImpliedBetas():
  """
  Implements calculation of option implied correlations following Buss, A. and Vilkov, G., 2012.
  Measuring equity risk with option-implied correlations. The Review of Financial Studies, 25(10), pp.3113-3140.
  """
  def __init__(self):
    """
    needed input:
    n_constituents = 500
    t = 4000
    # 500 stock return series
    single_stock_returns = np.random.rand(t, n_constituents)
    # weight vectors for every t
    w = np.random.rand(t, n_constituents)
    w[1,:].shape # specific t
    # implied vol sigma vector of all single stocks for every t
    sigma_single_stocks = np.random.rand(t, n_constituents)
    sigma_single_stocks.shape
    sigma_single_stocks[1,:].shape # specific t
    # implied vol sigma of index for every t
    sigma_index = np.random.rand(t, 1)
    """



  def log_returns(x):
    """
    Calculates log returns

    Parameters
    ----------
    x : array_like, shape (time, n_constituents)
        DataFrame with constituents in columns and timeseries values in rows
    Returns
    -------
    log_returns : ndarray, shape (time, n_constituents)
        Returns log returns for every constituent
    """
    # convert to pandas DataFrame calc log returns
    single_stock_returns = pd.DataFrame(x)
    # calc log returns
    return np.log(single_stock_returns / single_stock_returns.shift(1, axis=0))

  def rolling_window_corr(df, rolling_window=252):
    """
    Calculates rolling window correlation matrices between constituents for every time point

    Parameters
    ----------
    df : array_like, shape (time, n_constituents)
        DataFrame with constituents in columns and timeseries values in rows
    rolling_window: array_like, scalar, shape (1,1)
        Number of time steps to use for rolling window. Default 252 for yearly
    Returns
    -------
    rolling_corrs : ndarray, shape (time, n_constituents, n_constituents)
        Returns correlation matrices for every time step in 3D structure
    """
    # rolling window correlations
    rolling_corrs = df.rolling(window=rolling_window, axis=0).corr(pairwise=True)
    # convert to numpy 3D ndarray, shape (time, constituents, constituents)
    rolling_corrs_3D = rolling_corrs.values.reshape(
      (-1, single_stock_returns.shape[1], single_stock_returns.shape[1]))
    return rolling_corrs_3D

  def strip_rolling_window_nan(sigma_q_m, w, sigma, roh):
    """
    Strips nan resulting from rolling window correlation calculation

    Parameters
    ----------
    sigma_q_m: scalar or array_like, shape (1, 1)
        Index implied volatility at time t
    w : array_like, shape (n_constituents, 1)
        Weight vector of all index constituents at time t.
    sigma: array_like, shape (n_constituents, 1)
        Volatility vector of all index constituents at time t.
    roh: array_like, shape (n_constituents, n_constituents)
        Correlation matrix of all index constituents at time t.
    Returns
    -------
    sigma_q_m: scalar or array_like, shape (1, 1)
        Stripped nan: Index implied volatility at time t
    w : array_like, shape (n_constituents, 1)
        Stripped nan: Weight vector of all index constituents at time t.
    sigma: array_like, shape (n_constituents, 1)
        Stripped nan: Volatility vector of all index constituents at time t.
    roh: array_like, shape (n_constituents, n_constituents)
        Stripped nan: Correlation matrix of all index constituents at time t.
    """

    return sigma_q_m, w, sigma, roh

  def sum_of_sum(w, sigma, roh):
    """
    Calculates sum of sum with more efficient matrix build in multiplication

    Parameters
    ----------
    w : array_like, shape (n_constituents, 1)
        Weight vector of all index constituents at time t.
    sigma: array_like, shape (n_constituents, 1)
        Volatility vector of all index constituents at time t.
    roh: array_like, shape (n_constituents, n_constituents)
        Correlation matrix of all index constituents at time t.
    Returns
    -------
    sumsum : scalar, shape (1,1)
        Returns sum of sum.
    """
    # outer product with matrix multiplication
    w_outer_prod = w @ w.T
    sigma_outer_prod = sigma @ sigma.T

    # element wise multiplication
    prod_prod = w_outer_prod * sigma_outer_prod * roh
    # sum all elements
    return sum(sum(prod_prod))

  def calc_alpha_t(sigma_q_m, w, sigma, roh):
    """
    Calculates alpha_t

    Parameters
    ----------
    sigma_q_m: scalar or array_like, shape (1, 1)
        Index implied volatility at time t
    w : array_like, shape (n_constituents, 1)
        Weight vector of all index constituents at time t.
    sigma: array_like, shape (n_constituents, 1)
        Volatility vector of all index constituents at time t.
    roh: array_like, shape (n_constituents, n_constituents)
        Correlation matrix of all index constituents at time t.
    Returns
    -------
    alpha_t : scalar, shape (1,1)
        Returns alpha at time t.
    """
    sum1 = sum_of_sum(w=w, sigma=sigma, roh=roh)
    sum2 = sum_of_sum(w=w, sigma=sigma, roh=1 - roh)

    alpha_t = (sigma_q_m - sum1) / sum2

    return alpha_t

  def calc_q_correlation(roh_p_t, alpha_t):
    """
    Calculates roh_q_t, risk neutral correlation at time t

    Parameters
    ----------
    roh_p_t : array_like, shape (n_constituents, n_constituents)
        P correlation under empirical measure at time t.
    alpha_t: scalar or array_like, shape (1, 1)
        alpha scaling value at time t
    Returns
    -------
    sumsum : array_like, shape (n_constituents, n_constituents)
        Correlation matrix under Q, risk neutral measure, at time t.
    """
    return roh_p_t - alpha_t * (1 - roh_p_t)

  def q_beta(sigma_q_m, w, sigma, roh):
    """
    Calculates Q betas under risk neutral measure

    Parameters
    ----------
    sigma_q_m: scalar or array_like, shape (1, 1)
        Index implied volatility at time t
    w : array_like, shape (n_constituents, 1)
        Weight vector of all index constituents at time t.
    sigma: array_like, shape (n_constituents, 1)
        Volatility vector of all index constituents at time t.
    roh: array_like, shape (n_constituents, n_constituents)
        Correlation matrix of all index constituents at time t.
    Returns
    -------
    q_beta : scalar, shape (n_constituents, 1)
        Returns Q beta for every constituent under risk neutral measure at time t.
    """

    return beta_i_t

