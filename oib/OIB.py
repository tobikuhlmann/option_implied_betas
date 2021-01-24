import numpy as np
import pandas as pd
import time

class OptionImpliedBetas():
  """
  Implements calculation of option implied correlations following Buss, A. and Vilkov, G., 2012.
  Measuring equity risk with option-implied correlations. The Review of Financial Studies, 25(10), pp.3113-3140.
  """

  def __init__(self, single_stock_prices, sigma_stocks, sigma_index, w):
    """
    Parameters
    ----------
    single_stock_prices : array_like, shape (time, n_constituents)
        DataFrame with constituents in columns and timeseries values in rows
    sigma_stocks: array_like, shape (t, n_constituents)
        Volatility vector of all index constituents over time t.
    sigma_index: scalar or array_like, shape (t, 1)
        Index implied volatility over time t
    w : array_like, shape (t, n_constituents)
        Weight vector of all index constituents over time t.
    """
    self.single_stock_prices = single_stock_prices
    self.w = w
    self.sigma_stocks = sigma_stocks
    self.sigma_index = sigma_index

    self.r_log = None
    self.corr_tensor = None
    self.alpha_t = None
    self.q_correlation = None
    self.beta_t = None

    # assert shapes


  def fit(self, rolling_window=252):
    """
    calculates first correlation and then alpha_t and q_correlation for every t
    :return:
    """
    # start timing
    t0 = time.time()
    # log returns
    self.log_returns()
    # rolling window correlations
    self.rolling_window_corr(rolling_window=rolling_window)
    # end timing
    t1 = time.time()
    print(f"correlation calculation execution time: {t1 - t0} seconds")

    # strip nan from corr calc
    self.strip_rolling_window_nan()

    # start timing
    t0 = time.time()
    self.alpha_t = np.empty(self.sigma_index.shape)
    self.q_correlation = np.empty(self.corr_tensor.shape)
    self.beta_t = np.empty(self.corr_tensor[:, :, 0].shape)
    # for every time step
    for t in range(0, self.w.shape[0]):
      self.alpha_t[t] = OptionImpliedBetas.calc_alpha_t(sigma_q_m=self.sigma_index[t],
                                                   w=self.w[t, :],
                                                   sigma=self.sigma_stocks[t, :],
                                                   roh=self.corr_tensor[t, :, :])
      self.q_correlation[t] = OptionImpliedBetas.calc_q_correlation(roh_p_t=self.corr_tensor[t, :, :],
                                                               alpha_t=self.alpha_t[t])
      self.beta_t[t] = OptionImpliedBetas.q_beta(sigma_q_m=self.sigma_index[t],
                                                 w=self.w[t, :],
                                                 sigma=self.sigma_stocks[t, :],
                                                 roh=self.q_correlation[t])[:, 0]
    # end timing
    t1 = time.time()
    print(f"Option implied correlations execution time: {t1 - t0} seconds")


  def log_returns(self):
    """
    Calculates log returns

    Parameters
    ----------
    x : array_like, shape (time, n_constituents)
        DataFrame with constituents in columns and price timeseries values in rows
    Returns
    -------
    log_returns : ndarray, shape (time, n_constituents)
        Returns log returns for every constituent
    """
    # convert to pandas DataFrame calc log returns
    single_stock_prices = pd.DataFrame(self.single_stock_prices)
    # calc log returns
    self.r_log = np.log(single_stock_prices / single_stock_prices.shift(1, axis=0))

  def rolling_window_corr(self, rolling_window=252):
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
    rolling_corrs = self.r_log.rolling(window=rolling_window, axis=0).corr(pairwise=True)
    # convert to numpy 3D ndarray, shape (time, constituents, constituents)
    self.corr_tensor = rolling_corrs.values.reshape(
      (-1, self.single_stock_prices.shape[1], self.single_stock_prices.shape[1]))

  def strip_rolling_window_nan(self):
    """
    Strips nan resulting from rolling window correlation calculation
    """
    # get number of nan's
    rolling_window = self.corr_tensor[np.isnan(self.corr_tensor[:, 1, 1])].shape[0]

    self.sigma_index = self.sigma_index[rolling_window:]
    self.w = self.w[rolling_window:, :]
    self.sigma_stocks = self.sigma_stocks[rolling_window:, :]
    self.corr_tensor = self.corr_tensor[rolling_window:, :, :]


  @staticmethod
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
    # reshape for correct matrix multiplication
    w = w.reshape(w.shape[0], 1)
    sigma = sigma.reshape(sigma.shape[0], 1)

    # outer product with matrix multiplication
    w_outer_prod = w @ w.T
    sigma_outer_prod = sigma @ sigma.T

    # element wise multiplication
    prod_prod = w_outer_prod * sigma_outer_prod * roh
    # sum all elements
    return sum(sum(prod_prod))

  @staticmethod
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
    # reshape for correct matrix multiplication
    w = w.reshape(w.shape[0], 1)
    sigma = sigma.reshape(sigma.shape[0], 1)

    sum1 = OptionImpliedBetas.sum_of_sum(w=w, sigma=sigma, roh=roh)
    sum2 = OptionImpliedBetas.sum_of_sum(w=w, sigma=sigma, roh=(1 - roh))

    alpha_t = - (sigma_q_m ** 2 - sum1) / sum2

    return alpha_t

  @staticmethod
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

  @staticmethod
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
        Q Correlation matrix of all index constituents at time t.
    Returns
    -------
    q_beta : scalar, shape (n_constituents, 1)
        Returns Q beta for every constituent under risk neutral measure at time t.
    """
    # reshape for correct matrix multiplication
    w = w.reshape(w.shape[0], 1)
    sigma = sigma.reshape(sigma.shape[0], 1)

    numerator = sigma * ((sigma * w).T @ roh).T
    denominator = sigma_q_m ** 2

    beta_t = numerator / denominator
    return beta_t


if __name__ == '__main__':
  x = np.arange(start=1, stop=2, step=0.1).reshape((10, 1))
  single_stock_prices = np.repeat(a=x, repeats=10, axis=1)
  sigma_single_stocks = np.full(shape=(10, 10), fill_value=0.1)
  sigma_index = np.full(shape=(10, 1), fill_value=0.1)
  w = np.full(shape=(10, 10), fill_value=0.1)

  oib_ob = OptionImpliedBetas(single_stock_prices=single_stock_prices,
                                  sigma_stocks=sigma_single_stocks,
                                  sigma_index=sigma_index,
                                  w=w)

  oib_ob.fit(rolling_window=5)

  print(oib_ob.r_log)
  print(oib_ob.corr_tensor)
  print(oib_ob.alpha_t)
  print(oib_ob.q_correlation)
  print(oib_ob.beta_t)