# import datetime as dt
import simulation as sim
import numpy as np
import pandas as pd
from _typing import Float, Int, Array2d, Array1d, FloatOrNpArray, Frame, DateType, BoolType


class Portfolio:
    def __init__(self, positions: Frame):
        self.positions = positions
        positions.index.name = 'Ticker'

    @classmethod
    def from_num_positions(cls, num_stocks: Int = 20, nav: Float = 1000000.0):
        """
        Constructs Portfolio based on number of positions
        :param num_stocks: number of stocks in portfolio
        :param nav: total value of the portfolio
        :return: Portfolio

        """
        ticker = sim.StockPrice.get_default_tickers(num_stocks)
        value = np.ones([num_stocks, 1]) * nav/num_stocks
        positions = pd.DataFrame(value, index=ticker, columns=['Market Value'])
        return Portfolio(positions)

    @classmethod
    def from_initial_positions(cls, positions: pd.DataFrame):
        """
        Constructs Portfolio based on initial positions
        :param positions: DataFrame with Index = Instrument Identifier, Columns: Market Value, Classifications
        :return: Portfolio

        df = pd.DataFrame([[100], [150], [200]], index=['AAPL', 'GOOG', 'FB'], columns=['Market Value'])
        """
        # positions = pd.DataFrame([[100], [150], [200]], index=['AAPL', 'GOOG', 'FB'], columns=['Market Value'])
        # positions.index.name = 'Stock'
        return Portfolio(positions)

    def _simulate(self, stock_price, start_date=None, end_date=None):
        """
        Apply stock prices to portfolio between start_date and end_date
        :param stock_price: in melted format. Columns: Stock, Price, Value Date
        :param start_date:
        :param end_date:
        :return: DataFrame
        """
        price = stock_price[['Value Date', 'Ticker', 'Value']]
        if start_date is not None:
            price = price[price['Value Date'] >= pd.to_datetime(start_date)]
        if end_date is not None:
            price = price[price['Value Date'] <= pd.to_datetime(end_date)]
        pos = self.positions.reset_index()  # so that Ticker is a column
        df = pos.merge(price, how='left', on='Ticker')

        return df

    def portfolio_value(self, stock_price, start_date=None, end_date=None):
        """
        Apply stock prices to portfolio between start_date and end_date
        :param stock_price: in melted format. Columns: Stock, Price, Value Date
        :param start_date:
        :param end_date:
        :return: DataFrame
        """
        df = self._simulate(stock_price, start_date, end_date)
        df_return = df.groupby(["Value Date"]).agg({"Value": np.sum})

        return df_return
