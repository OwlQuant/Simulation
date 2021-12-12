import datetime as dt
import numpy as np
import pandas as pd
from util._typing import Float, Int, Array2d, FloatOrNpArray, DateType, BoolType


class StockPrice:
    @staticmethod
    def gbm(s0: FloatOrNpArray = 100.0, vol: FloatOrNpArray = 0.2, t: Float = 1, rf: FloatOrNpArray = 0.01,
            corr: FloatOrNpArray = 0.0, n_paths: Int = 10000) -> Array2d:

        """
        Geometric Brownian Motion simulation
        :param s0: Start Price
        :param vol: Volatility (Annualized)
        :param t: Time (Years)
        :param rf: Riskfree Rate (Annual)
        :param corr: Flat Correlation or correlation matrix [num_stocks x num_stocks]
        :param n_paths: Number of paths needed
        :return: Nx1 numpy array of simulated prices
        """
        num_stocks = len(s0) if isinstance(s0, (list, np.ndarray)) else 1
        if not isinstance(corr, np.ndarray) and corr == 0.0:  # todo use np.ndim()
            rand_matrix = np.random.standard_normal([n_paths, num_stocks])
        else:
            means = np.zeros(num_stocks)
            cov_matrix = (np.ones([num_stocks, num_stocks]) * corr)  # given stdev=1 corr and cov the same
            # if proper conversion needed, use cov_matrix = stdev * corr_matrix * stdev.T
            np.fill_diagonal(cov_matrix, 1)
            rand_matrix = np.random.multivariate_normal(means, cov_matrix, n_paths)
        # TODO - rand_matrix is constructed using unit stdev and used in equation below.
        #  Actual stdev (vol) is not used in random - check if this is accurate? Seems okay in excel verification
        return s0 * np.exp((rf - vol * vol / 2.0) * t + vol * rand_matrix * np.sqrt(t))


    @staticmethod
    def simulate_price(start_date: DateType = dt.date.today(),
                       end_date: DateType = dt.date.today() + dt.timedelta(days=365),
                       start_price: FloatOrNpArray = 250.0, vol_ann: FloatOrNpArray = 0.20, rf: FloatOrNpArray = 0.02,
                       ticker=None,
                       corr: FloatOrNpArray = 0.0, include_weekend: BoolType = True, melted_output: BoolType = True):
        """
        Simulate stock prices based on parameters.
        Returns prices between start and end date for all the instruments requested (based on #elements in start_price)

        :param start_date: Start Date of simulation. Default: today
        :param end_date: End Date of simulation. Default: today + 365 days
        :param start_price: Start Price of Stock being simulated. Number or np.array for multi-stock simulation
        :param vol_ann: Volatility (Annualized). Number or np.array for multi-stock simulation
        :param rf: Riskfree Rate. Number or np.array for multi-stock simulation
        :param ticker:
        :param corr: Flat Correlation or correlation matrix [num_stocks x num_stocks]
        :param include_weekend: Include prices for weekend
        :param melted_output: Output format in pivot or melted format. Default melted format
        :return: pd.DataFrame of Dates, Tickers, Prices. Either in Pivot or Melted format

        start_price, vol_ann, rf can be entered as numbers or as 1D np.array for multi-stock simulated prices
        corr can be entered as number or as 2D np.array for multi-stock simulated prices

        Arrays need to be np.array.
        """

        num_stocks = len(start_price) if isinstance(start_price, (list, np.ndarray)) else 1
        ticker = ticker if ticker is None or isinstance(ticker, (list, np.ndarray)) else [ticker]

        t = 1.0 / 365  # daily simulation so time = 1 day
        count_days = (end_date - start_date).days + 1  # including start/end date

        # simulate 1+daily returns as 2d array: count_days x num_stocks
        daily_returns = StockPrice.gbm(s0=np.ones(num_stocks), vol=vol_ann, t=t, rf=rf, corr=corr, n_paths=count_days)

        # daily_returns = np.exp((rf - vol_ann * vol_ann / 2.0) * t + vol_ann * np.random.standard_normal(
        #     [count_days, num_stocks]) * np.sqrt(t))

        daily_returns[0] = 1.0  # zero return on first day.
        # Line could be removed if start price is considered as beginning-of-day price

        cum_returns = daily_returns.cumprod(axis=0)
        prices = start_price * cum_returns
        idx = pd.date_range(start_date, end_date)
        cols = StockPrice.get_default_tickers(num_stocks) if ticker is None else ticker
        return_df = pd.DataFrame(prices, columns=cols, index=idx)
        if not include_weekend:  # this just excludes weekend. Fri-Mon vol is corresponding to 3 days though (TODO)
            bus_idx = pd.bdate_range(start_date, end_date)
            return_df = return_df.loc[bus_idx].copy()
        return_df.index.name = 'Value Date'

        if melted_output:
            # convert date(i) vs ticker(c) prices(value) pivot
            # to date(c), ticker(c), value(c) table
            return_df = return_df.melt(value_vars=return_df.columns, ignore_index=False, var_name='Ticker',
                                       value_name='Value').reset_index()

        return return_df

    @classmethod
    def get_default_tickers(cls, num_stocks=10):
        """
        Get Default Tickers in format Stock 1, Stock 2, etc. Used in case tickers are None
        :param num_stocks:
        :return: Array of default tickers
        """
        return [f'Stock {i + 1}' for i in range(num_stocks)]
