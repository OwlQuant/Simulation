import datetime as dt

import numpy as np
import pandas as pd
import logging
from typing import Union
from util._typing import Float, Int, Array2d, FloatOrNpArray, DateType, BoolType, Frame


class StockPrice:
    @classmethod
    def _is_correlation_matrix(cls, arr: Array2d, tol: float = 1e-8) -> bool:
        """ Check if passed variable arr is correlation matrix. Returns True/False """
        # 0. if variable is not an nd array
        if not isinstance(arr, np.ndarray):
            return False

        # 1. 2d and square
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            return False

        n = arr.shape[0]
        # 2. Symmetry check
        if not np.allclose(arr, arr.T, atol=tol):
            logging.info(f"Correlation matrix passed is not symmetric: \n{arr}")
            return False

        # 3. diagonal elements = 1
        if not np.allclose(np.diag(arr), np.ones(n), atol=tol):
            logging.info(f"Correlation matrix passed doesn't have diagonal elements = 1: \n{arr}")
            return False

        # 4 off-diagonal elements between -1 and 1
        off_diag = arr - np.eye(n)
        if np.any(off_diag < -1 - tol) or np.any(off_diag > 1 + tol):
            logging.info(f"Correlation matrix has values outside range -1 to +1: \n{arr}")
            return False

        # 5. Positive semidefinite check (eigenvalues >= 0)
        eigvals = np.linalg.eigvalsh(arr)  # symmetric => faster eig solver
        if np.any(eigvals < -tol):  # allow tiny negative values from rounding
            logging.info(f"Correlation matrix is not positive semidefinite (based on eigen values): \n{arr}")
            return False
        return True

    @classmethod
    def gbm(cls, s0: FloatOrNpArray = 100.0, vol: FloatOrNpArray = 0.2, t: Float = 1, rf: FloatOrNpArray = 0.01,
            dvd: FloatOrNpArray = 0.0, corr: FloatOrNpArray = 0.0, n_paths: Int = 10000) -> Union[Array2d, None]:

        """
        Geometric Brownian Motion simulation
        :param s0: Start Price, could be market value
        :param vol: Volatility (Annualized). 0.2 => 20%
        :param t: Time (Years)
        :param rf: Riskfree Rate (Annual). 0.01 => 1%
        :param dvd: Dividend Rate (Annual). 0.01 => 1%
        :param corr: Flat Correlation or correlation matrix [num_stocks x num_stocks]. Correlation range -1 to +1
        :param n_paths: Number of paths needed
        :return: Nx1 numpy array of simulated prices or None if simulation fails
        """
        num_stocks = len(s0) if isinstance(s0, (list, np.ndarray)) else 1

        rng = np.random.default_rng()
        rand_matrix = None

        if corr is None or (isinstance(corr, (int, float)) and corr == 0.0):
            rand_matrix = rng.standard_normal([n_paths, num_stocks])
        elif isinstance(corr, (int, float)) and np.abs(corr) <= 1.0:
            means = np.zeros(num_stocks)
            cov_matrix = (np.ones([num_stocks, num_stocks]) * corr)  # given stdev=1 corr and cov the same
            np.fill_diagonal(cov_matrix, 1.0)
            assert cls._is_correlation_matrix(
                cov_matrix)  # issue with corr value passed. is it very negative resulting in not positive semidefinite?
            rand_matrix = rng.multivariate_normal(means, cov_matrix, n_paths)
        elif cls._is_correlation_matrix(corr) and corr.shape[0] == num_stocks:
            means = np.zeros(num_stocks)
            cov_matrix = corr  # given stdev=1 corr and cov the same
            # if proper conversion needed, use cov_matrix = stdev * corr_matrix * stdev.T
            rand_matrix = rng.multivariate_normal(means, cov_matrix, n_paths)
        else:
            logging.warning(f"Incorrect correlation matrix passed: \n{corr}")
            assert False

        if isinstance(vol, (int, float)):
            vol_array = np.ones(num_stocks) * vol
        elif isinstance(vol, np.ndarray) and vol.ndim == 1 and len(vol) == num_stocks:
            vol_array = vol
        else:
            logging.warning(f"Incorrect volatility list passed: \n{vol}")
            assert False  # Incorrect volatility list passed

        if isinstance(rf, (int, float)):
            rf_array = np.ones(num_stocks) * rf
        elif isinstance(rf, np.ndarray) and rf.ndim == 1 and len(rf) == num_stocks:
            rf_array = rf
        else:
            logging.warning(f"Incorrect Rates list passed: \n{rf}")
            assert False  # Incorrect Rates list passed

        if isinstance(dvd, (int, float)):
            dvd_array = np.ones(num_stocks) * dvd
        elif isinstance(dvd, np.ndarray) and dvd.ndim == 1 and len(dvd) == num_stocks:
            dvd_array = dvd
        else:
            logging.warning(f"Incorrect Dividend list passed: \n{dvd}")
            assert False  # Incorrect Dividend list passed
        mu_array = rf_array - dvd_array

        # Note - rand_matrix is standard normal matrix constructed using 0 mean / unit stdev and used in equation below.
        # Actual stdev (vol) is not used in random generation. It is used when simulating results below
        return s0 * np.exp((mu_array - 0.5 * (vol_array ** 2)) * t + vol_array * rand_matrix * np.sqrt(t))

    @classmethod
    def simulate_price(cls, start_date: DateType = dt.date.today(),
                       end_date: DateType = dt.date.today() + dt.timedelta(days=365),
                       start_price: FloatOrNpArray = 250.0, vol_ann: FloatOrNpArray = 0.20, rf: FloatOrNpArray = 0.02,
                       ticker=None,
                       corr: FloatOrNpArray = 0.0, include_weekend: BoolType = False,
                       melted_output: BoolType = True) -> Frame:
        """
        Simulate daily stock prices based on parameters.
        Returns one set of prices between start and end date for all the instruments requested (based on #elements in start_price)

        :param start_date: Start Date of simulation. Default: today
        :param end_date: End Date of simulation. Default: today + 365 days
        :param start_price: Start Price of Stock being simulated as of end of Start Date. Number or np.array for multi-stock simulation
        :param vol_ann: Volatility (Annualized). Number or np.array for multi-stock simulation
        :param rf: Riskfree Rate. Number or np.array for multi-stock simulation
        :param ticker:
        :param corr: Flat Correlation or correlation matrix [num_stocks x num_stocks]
        :param include_weekend: Include prices for weekend
        :param melted_output: Output format in pivot or melted (unpivoted) format. Default melted format
        :return: pd.DataFrame of Dates, Tickers, Prices. Either in Pivot or Melted format

        start_price, vol_ann, rf can be entered as numbers or as 1D np.array (size num_stocks) for multi-stock simulated prices
        corr can be entered as number or as 2D np.array for multi-stock simulated prices

        Arrays need to be np.array.
        """

        num_stocks = len(start_price) if isinstance(start_price, (list, np.ndarray)) else 1
        ticker = ticker if ticker is None or isinstance(ticker, (list, np.ndarray)) else [ticker]

        t = 1.0 / (365.0 if include_weekend else 252.0)  # daily simulation so time = 1 day
        count_days = (end_date - start_date).days + 1  # including start/end date

        # simulate 1+daily returns as 2d array: (r)count_days x (c)num_stocks
        daily_returns = StockPrice.gbm(s0=np.ones(num_stocks), vol=vol_ann, t=t, rf=rf, corr=corr, n_paths=count_days)

        # daily_returns = np.exp((rf - vol_ann * vol_ann / 2.0) * t + vol_ann * np.random.standard_normal(
        #     [count_days, num_stocks]) * np.sqrt(t))

        cols = cls._get_default_tickers(num_stocks) if ticker is None else ticker
        if include_weekend:
            idx = pd.date_range(start_date, end_date)
        else:
            idx = pd.bdate_range(start_date, end_date)

        daily_returns[0] = 1.0  # zero return on first day.
        # Line could be removed if start price is considered as beginning-of-day price
        daily_returns = daily_returns[:len(idx)]  # in case weekend exclusion needs less rows
        cum_returns = daily_returns.cumprod(axis=0)
        prices = start_price * cum_returns
        return_df = pd.DataFrame(prices, columns=cols, index=idx)
        return_df.index.name = 'Value Date'

        if melted_output:
            # convert date(i) vs ticker(c) prices(value) pivot
            # to date(c), ticker(c), value(c) table
            return_df = return_df.melt(value_vars=return_df.columns, ignore_index=False, var_name='Ticker',
                                       value_name='Value').reset_index()

        return return_df

    @classmethod
    def get_price_similation_stats(cls, df_price: Frame, days_per_year: int = 252) -> Frame:
        """
        Returns similation stats - Count, Mean, Standard Deviation, Skew, Kurtosis, Correlation
        - Mean is expected to match rf (annualized) input
        - Standard Deviation is expected to match volatility (annualized) input
        - Correlation is expected to match correlation input
        - Skew and Kurtosis is expected to be 0 for normal distribution
        This function can be used to check accuracy of simulation results
        """
        df_return = (df_price - df_price.shift(1)) / df_price.shift(1)
        df_out = pd.concat([
            df_price.count(axis=0),
            df_return.mean(axis=0) * days_per_year,  # expected as rf(ann) for normal distribution
            df_return.std(axis=0) * np.sqrt(days_per_year),  # expected as volatility(ann) for normal distribution
            df_return.skew(axis=0) / np.sqrt(days_per_year),  # expected 0 for normal distribution
            df_return.kurtosis(axis=0) / days_per_year,  # this is Excess Kurtosis (so 0 for normal distribution)
        ], axis=1).transpose()
        df_out.index = ['Count Price', 'Mean Return', 'Std Dev Ret', 'Skew Return', 'Kurtosis Return']

        df_corr = df_return.corr()
        df_corr = df_corr.where(np.tril(np.ones(df_corr.shape), k=0).astype(bool))  # show only lower triangular
        df_corr.index = 'Corr:' + df_corr.index
        return pd.concat([df_out, df_corr])

    @classmethod
    def _get_default_tickers(cls, num_stocks=10):
        """
        Get Default Tickers in format Stock 1, Stock 2, etc. Used in case tickers are None
        :param num_stocks:
        :return: Array of default tickers
        """
        return [f'Stock {i + 1}' for i in range(num_stocks)]


if __name__ == '__main__':
    print("GBM simulation")
    print("--------------")
    print("Single GBM price")
    gbm = StockPrice.gbm(s0=1000, vol=0.2, t=1.1, rf=0.01, n_paths=5)
    print(gbm.round(2))
    print(pd.DataFrame(gbm).describe())

    print("Multiple GBM prices")
    gbm = StockPrice.gbm(s0=np.ones(5) * 1000, vol=0.2, t=0.1, rf=0.01, corr=.25, n_paths=10000)
    print(f'Prices:\n{gbm.round(2)}')
    print(f'Describe:\n{pd.DataFrame(gbm).describe()}')
    print(f'Correlation:\n{np.corrcoef(np.log(gbm / 1000), rowvar=False).round(2)}')

    print("Multiple GBM prices with corr matrix")
    gbm = StockPrice.gbm(s0=np.ones(3) * 1000, vol=np.array([0.2, .4, .01]), t=0.1, rf=0.01,
                         corr=np.array([[1, -.5, 0.2], [-.5, 1, 0], [0.2, 0, 1]]), n_paths=10000)
    print(f'Prices:\n{gbm.round(2)}')
    print(f'Describe:\n{pd.DataFrame(gbm).describe()}')
    print(f'Correlation:\n{np.corrcoef(np.log(gbm / 1000), rowvar=False).round(2)}')

    print("Stock Price simulation")
    print("----------------------")
    start_date = dt.date(2025, 1, 1)
    end_date = dt.date(2075, 1, 15)

    print("5 general stocks with flat correlation")
    df = StockPrice.simulate_price(start_date=start_date, end_date=end_date, start_price=np.ones(5) * 1000,
                                   vol_ann=0.15, corr=0.2, rf=0.5, melted_output=False)
    # 5 general stocks
    print(f'{df}\n')
    print(f'{StockPrice.get_price_similation_stats(df).round(3).fillna("")}\n')

    print("3 specific stocks, with corr/vol matrix")
    df = StockPrice.simulate_price(start_date=start_date, end_date=end_date,
                                   start_price=np.array([1000, 1000, 1000]), ticker=['Aapl', 'Fb', 'Goog'],
                                   vol_ann=np.array([.10, .15, .20]), rf=np.array([.01, -.02, .02]),
                                   corr=np.array([[1, .75, .5], [.75, 1, -.10], [.5, -.10, 1]]),
                                   melted_output=False)
    print(f'{df}\n')
    print(f'{StockPrice.get_price_similation_stats(df).round(3).fillna("")}\n')
