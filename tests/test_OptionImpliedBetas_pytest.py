import numpy as np
import pandas as pd
import time
import pytest

from oib import OIB


class TestOptionImpliedBetas:

    @pytest.fixture
    def test_object_init(self):
        """
        Setup code to create test data
        """
        x = np.arange(start=1, stop=2, step=0.1).reshape((10, 1))
        single_stock_prices = np.repeat(a=x, repeats=10, axis=1)
        sigma_stocks = np.full(shape=(10, 10), fill_value=0.1)
        sigma_index = np.full(shape=(10, 1), fill_value=0.1)
        w = np.full(shape=(10, 10), fill_value=0.1)

        # init object
        return OIB.OptionImpliedBetas(single_stock_prices=single_stock_prices,
                               sigma_stocks=sigma_stocks,
                               sigma_index=sigma_index,
                               w=w)

    @pytest.fixture
    def test_object_fit(self):
        """
        Setup code to create test data
        """
        x = np.arange(start=1, stop=2, step=0.1).reshape((10, 1))
        single_stock_prices = np.repeat(a=x, repeats=10, axis=1)
        sigma_stocks = np.full(shape=(10, 10), fill_value=0.1)
        sigma_index = np.full(shape=(10, 1), fill_value=0.1)
        w = np.full(shape=(10, 10), fill_value=0.1)

        # init object
        oib_ob = OIB.OptionImpliedBetas(single_stock_prices=single_stock_prices,
                               sigma_stocks=sigma_stocks,
                               sigma_index=sigma_index,
                               w=w)
        oib_ob.fit(rolling_window=5)
        return oib_ob

    def test_init_shapes(self, test_object_init):
        """
        test shapes
        """
        assert test_object_init.single_stock_prices.shape == (10, 10)
        assert test_object_init.sigma_stocks.shape == (10, 10)
        assert test_object_init.sigma_index.shape == (10, 1)
        assert test_object_init.w.shape == (10, 10)

        assert test_object_init.r_log is None
        assert test_object_init.corr_tensor is None
        assert test_object_init.alpha_t is None
        assert test_object_init.q_correlation is None
        assert test_object_init.beta_t is None

    def test_fit(self, test_object_fit):
        """
        test if object fitted correctly
        """
        assert test_object_fit.r_log is not None
        assert test_object_fit.corr_tensor is not None
        assert test_object_fit.alpha_t is not None
        assert test_object_fit.q_correlation is not None
        assert test_object_fit.beta_t is not None

        assert test_object_fit.r_log.shape == (10, 10)
        assert test_object_fit.corr_tensor.shape == (5, 10, 10)
        assert test_object_fit.alpha_t.shape == (5, 1)
        assert test_object_fit.q_correlation.shape == (5, 10, 10)
        assert test_object_fit.beta_t.shape == (5, 10)

    def test_log_returns(self, test_object_fit):
        """
        test calculation of log returns
        """
        # calculate manually
        single_stock_prices = pd.DataFrame(test_object_fit.single_stock_prices)
        # calc log returns
        r_log = np.log(single_stock_prices / single_stock_prices.shift(1, axis=0))

        assert (r_log.dropna() == test_object_fit.r_log.dropna()).all().all()
        assert r_log.shape == (10, 10)

    def test_rolling_window_corr(self, test_object_fit):
        """
        test calculation of rolling window correlations
        """
        # rolling window correlations
        rolling_corrs = test_object_fit.r_log.rolling(window=5, axis=0).corr(pairwise=True)
        rolling_corrs.shape
        # convert to numpy 3D ndarray, shape (time, constituents, constituents)
        corr_tensor = rolling_corrs.values.reshape((-1, test_object_fit.single_stock_prices.shape[1],
                                                    test_object_fit.single_stock_prices.shape[1]))

        assert (corr_tensor[5:, :, :] == test_object_fit.corr_tensor).all()

        assert test_object_fit.corr_tensor.shape == (5, 10, 10)
        assert corr_tensor.shape == (10, 10, 10)
        assert corr_tensor[5:, :, :].shape == (5, 10, 10)

    def test_strip_rolling_window_nan(self, test_object_init, test_object_fit):
        """
        test strip of nans, which get created by rolling window calculation
        """
        # run fit pipeline until nans get stripped to do manual calculation
        test_object_init.log_returns()
        test_object_init.rolling_window_corr(rolling_window=5)

        # calculate length of rolling window, should be 5
        rolling_window = test_object_init.corr_tensor[np.isnan(test_object_init.corr_tensor[:, 1, 1])].shape[0]
        assert rolling_window == 5

        # assert manual calculation shapes equals function calls
        assert test_object_init.sigma_index[rolling_window:].shape == test_object_fit.sigma_index.shape
        assert test_object_init.w[rolling_window:, :].shape == test_object_fit.w.shape
        assert test_object_init.sigma_stocks[rolling_window:, :].shape == test_object_fit.sigma_stocks.shape
        assert test_object_init.corr_tensor[rolling_window:, :, :].shape == test_object_fit.corr_tensor.shape

        # assert manual calculation values equals function calls
        assert (test_object_init.sigma_index[rolling_window:] == test_object_fit.sigma_index).all()
        assert (test_object_init.w[rolling_window:, :] == test_object_fit.w).all()
        assert (test_object_init.sigma_stocks[rolling_window:, :] == test_object_fit.sigma_stocks).all()
        assert (test_object_init.corr_tensor[rolling_window:, :, :] == test_object_fit.corr_tensor).all()

    def test_sum_of_sum(self, test_object_fit):
        """
        test matrix calculation to replace double sum
        """
        # data needed: w, sigma, roh at specific time t, let's take 0
        t = 0
        w = test_object_fit.w[t, :]
        sigma = test_object_fit.sigma_stocks[t, :]
        roh = test_object_fit.corr_tensor[t, :, :]

        w = w.reshape(w.shape[0], 1)
        sigma = sigma.reshape(sigma.shape[0], 1)

        # manual result
        w_outer_prod = w @ w.T
        sigma_outer_prod = sigma @ sigma.T
        prod_prod = w_outer_prod * sigma_outer_prod * roh
        manual_result = sum(sum(prod_prod))
        # package result
        package_result = test_object_fit.sum_of_sum(w=w, sigma=sigma, roh=roh)

        assert manual_result.shape == package_result.shape
        assert manual_result == package_result

    def test_calc_alpha_t(self, test_object_fit):
        """
        test calculation of alpha_t
        """
        # data needed: w, sigma, roh at specific time t, let's take 0
        t = 0
        w = test_object_fit.w[t, :]
        sigma = test_object_fit.sigma_stocks[t, :]
        roh = test_object_fit.corr_tensor[t, :, :]

        w = w.reshape(w.shape[0], 1)
        sigma = sigma.reshape(sigma.shape[0], 1)

        # manual calculation
        sum1 = test_object_fit.sum_of_sum(w=w, sigma=sigma, roh=roh)
        sum2 = test_object_fit.sum_of_sum(w=w, sigma=sigma, roh=(1 - roh))

        manual_alpha_t = (test_object_fit.sigma_index - sum1) / sum2

        # compare to package result
        package_alpha_t = test_object_fit.calc_alpha_t(sigma_q_m=test_object_fit.sigma_index, w=w, sigma=sigma, roh=roh)

        assert manual_alpha_t.shape == package_alpha_t.shape
        assert (manual_alpha_t == package_alpha_t).all()

    def test_calc_q_correlation(self, test_object_fit):
        """
        test calculation of q beta
        """
        # data needed: w, sigma, roh at specific time t, let's take 0
        t = 0
        w = test_object_fit.w[t, :]
        sigma = test_object_fit.sigma_stocks[t, :]
        roh = test_object_fit.corr_tensor[t, :, :]

        w = w.reshape(w.shape[0], 1)
        sigma = sigma.reshape(sigma.shape[0], 1)

        # manual calculation
        sum1 = test_object_fit.sum_of_sum(w=w, sigma=sigma, roh=roh)
        sum2 = test_object_fit.sum_of_sum(w=w, sigma=sigma, roh=(1 - roh))

        manual_alpha_t = (test_object_fit.sigma_index - sum1) / sum2

        # compare to package result
        package_alpha_t = test_object_fit.calc_alpha_t(sigma_q_m=test_object_fit.sigma_index, w=w, sigma=sigma, roh=roh)

        assert manual_alpha_t.shape == package_alpha_t.shape
        assert (manual_alpha_t == package_alpha_t).all()

    def test_q_beta(self, test_object_fit):
        """
        test calculation of q beta
        """
        # data needed: w, sigma, roh at specific time t, let's take 0
        t = 0
        w = test_object_fit.w[t, :]
        sigma = test_object_fit.sigma_stocks[t, :]
        roh = test_object_fit.q_correlation[t]
        sigma_q_m = test_object_fit.sigma_index[t]

        w = w.reshape(w.shape[0], 1)
        sigma = sigma.reshape(sigma.shape[0], 1)

        # manual
        numerator = sigma * ((sigma * w).T @ roh).T
        denominator = sigma_q_m ** 2
        manual_beta_t = numerator / denominator

        # package
        package_beta_t = test_object_fit.q_beta(sigma_q_m=test_object_fit.sigma_index[t],
                                       w=test_object_fit.w[t, :],
                                       sigma=test_object_fit.sigma_stocks[t, :],
                                       roh=test_object_fit.q_correlation[t])
        assert manual_beta_t.shape == package_beta_t.shape
        assert (manual_beta_t == package_beta_t).all()



