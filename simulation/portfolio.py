import datetime as dt
import simulation as sim
import numpy as np
import pandas as pd
from typing import List
from util._typing import Float, Int, Frame, Series
from util.static_data import companies, start_fx_to_usd, start_prices


class Portfolio:
    instrument_static = None

    def __init__(self, positions: Series, cash: Series = None):
        self._init_positions: Series = positions
        self._init_cash: Series = cash

        self._ticker_currency_map = {c['ticker']: c['currency'] for c in companies if
                                     c['ticker'] in self._init_positions.index}
        self._transactions: Series = None

    def get_pos_tickers(self) -> List[str]:
        return self._init_positions.index.to_list()

    def get_cash_tickers(self) -> List[str]:
        return self._init_cash.index.to_list()

    def get_ccy_tickers_pos_cash(self) -> List[str]:
        """ Get unique currency tickers [excl USD] corresponding to positions and cash """
        pos_tickers = self.get_pos_tickers()
        pos_ccy_tickers = [self._ticker_currency_map[tc] for tc in pos_tickers]
        cash_tickers = self.get_cash_tickers()
        return list(set(pos_ccy_tickers + cash_tickers) - {"USD"})

    @classmethod
    def generate_positions(cls, num_stocks: Int = 20, nav: Float = 1000000.0, cash_pct: Float = 0.1):
        """
        Constructs Portfolio based on number of positions. Tickers selected based on list of tickers in static_data
        :param num_stocks: number of stocks in portfolio
        :param nav: total value of the portfolio. Each position is given equal proportion to start
        :return: Portfolio

        """
        tickers = np.random.choice([c['ticker'] for c in companies], num_stocks, replace=False)  # array
        tckr_ccy_map = {c['ticker']: c['currency'] for c in companies if c['ticker'] in tickers}
        ticker_ccy = [tckr_ccy_map[tc] for tc in tickers]
        prices = [start_prices[t] for t in tickers]
        fx = [start_fx_to_usd[c] for c in ticker_ccy]
        prices_usd = np.array(prices) * np.array(fx)
        qty = nav * (1 - cash_pct) / len(tickers) / prices_usd

        s_positions = pd.Series(qty, tickers)

        # cash_ccy = ['EUR', 'USD']  # this can be updated as needed. Cash balance equally split between members
        cash_ccy = ['USD']  # this can be updated as needed. Cash balance equally split between members
        fx = np.array([1.0 if c == 'USD' else start_fx_to_usd[c] for c in cash_ccy])
        s_cash = pd.Series(nav * cash_pct / fx / len(cash_ccy), cash_ccy)

        return Portfolio(s_positions, s_cash)

    def get_positions_for_dates(self, dates: List[dt.date]) -> Frame:
        """
        Gets positions - tickers, quantity - for list of dates passed
        :param dates: list of dates for which positions are desired
        :return: DataFrame of positions with index = dates, columns = tickers, values = qty
        """
        pos_map = {d: self._init_positions for d in dates}
        df_return = pd.DataFrame.from_dict(pos_map, orient='index')
        return df_return

    def get_cash_for_dates(self, dates: List[dt.date]) -> Frame:
        """
        Gets cash - ccy_ticker, local amount - for list of dates passed
        :param dates: list of dates for which cash is desired
        :return: DataFrame of positions with index = dates, columns = currency tickers, values = local amount
        """
        pos_map = {d: self._init_cash for d in dates}
        df_return = pd.DataFrame.from_dict(pos_map, orient='index')
        return df_return

    ########## Portfolio Accounting formulas from here on ############
    def position_market_value(self, df_prices: Frame, df_fx: Frame) -> Frame:
        """
        Apply stock prices and fx to value portfolio positions
        :param df_prices: in pivot format. Index = Dates, Columns = Ticker, Value = Price
        :param df_fx: in pivot format. Index = Dates, Columns = Ticker, Value = Price
        :return: DataFrame
        """
        dates = df_prices.index.to_list()
        df_positions = self.get_positions_for_dates(dates)

        tickers = df_positions.columns.to_list()
        tckr_ccy = {c['ticker']: c['currency'] for c in companies if c['ticker'] in tickers}
        ccy = [tckr_ccy[tc] for tc in tickers]

        df_fx_positions = df_fx.copy()
        df_fx_positions['USD'] = 1
        df_fx_positions = df_fx_positions[ccy]
        df_fx_positions.columns = df_prices.columns
        df_pos_values = df_positions * df_prices * df_fx_positions  # in USD
        return df_pos_values

    def position_market_value_local(self, df_prices: Frame) -> Frame:
        """
        Apply stock prices to value portfolio positions
        :param df_prices: in pivot format. Index = Dates, Columns = Ticker, Value = Price
        :return: DataFrame
        """
        dates = df_prices.index.to_list()
        df_positions = self.get_positions_for_dates(dates)

        tickers = df_positions.columns.to_list()
        tckr_ccy = {c['ticker']: c['currency'] for c in companies if c['ticker'] in tickers}
        ccy = [tckr_ccy[tc] for tc in tickers]

        df_pos_values = df_positions * df_prices  # in Local
        return df_pos_values

    def cash_market_value(self, df_fx: Frame) -> Frame:
        """
        Apply fx to value portfolio cash
        :param df_fx: in pivot format. Index = Dates, Columns = Ticker, Value = Price
        :return: DataFrame
        """
        dates = df_fx.index.to_list()
        df_cash = self.get_cash_for_dates(dates)
        df_fx = df_fx.copy()
        df_fx['USD'] = 1  # USD should always be valued at 1
        df_fx = df_fx[df_cash.columns]

        df_cash_values = df_cash * df_fx  # in USD
        return df_cash_values

    def cash_market_value_local(self) -> Frame:
        """
        Value portfolio cash
        :return: DataFrame
        """
        dates = df_fx.index.to_list()
        df_cash = self.get_cash_for_dates(dates)

        df_cash_values = df_cash  # in Local
        return df_cash_values

    def portfolio_nav(self, df_prices: Frame, df_fx: Frame) -> Frame:
        """ Summarize valuation of entire portfolio for various dates as 'Position', 'Cash', 'Total' """
        df_position_value = self.position_market_value(df_prices, df_fx)
        s_position_value = df_position_value.sum(axis=1)
        s_position_value.name = 'Positions'

        df_cash_value = self.cash_market_value(df_fx)
        s_cash_value = df_cash_value.sum(axis=1)
        s_cash_value.name = 'Cash'
        df_return = pd.concat([s_position_value, s_cash_value], axis=1)
        df_return['Total'] = df_return.sum(axis=1)
        return df_return

    def position_start_market_value(self, df_prices: Frame, df_fx: Frame) -> Frame:
        """ Get start Mkt Value of positions """
        df_return = self.position_market_value(df_prices, df_fx)
        return df_return.shift(1)

    def cash_start_market_value(self, df_fx: Frame) -> Frame:
        """ Get cash Mkt Value of cash """
        df_return = self.cash_market_value(df_fx)
        return df_return.shift(1)

    def position_start_market_value_local(self, df_prices: Frame) -> Frame:
        """ Get start Mkt Value of positions """
        df_return = self.position_market_value_local(df_prices)
        return df_return.shift(1)

    def cash_start_market_value_local(self) -> Frame:
        """ Get cash Mkt Value of cash """
        df_return = self.cash_market_value_local()
        return df_return.shift(1)

    def portfolio_start_nav(self, df_prices: Frame, df_fx: Frame) -> Frame:
        """ Get start NAV of portfolio """
        df_return = self.portfolio_nav(df_prices, df_fx)
        return df_return.shift(1)

    def position_pl(self, df_prices: Frame, df_fx: Frame) -> Frame:
        """ Get PL for positions """
        df_return = self.position_market_value(df_prices, df_fx) - self.position_start_market_value(df_prices, df_fx)
        return df_return

    def position_pl_local(self, df_prices: Frame) -> Frame:
        """ Get PL for positions """
        df_return = self.position_market_value_local(df_prices) - self.position_start_market_value_local(df_prices)
        return df_return

    def position_pl_fx_raw(self, df_prices: Frame, df_fx: Frame) -> Frame:
        """ Get daily FX PL for positions
        FX PL = Start MV Local * (End FX - Start FX)
        Non-FX PL = Start FX * (End MV Local - Start MV Local)
        Cross-FX PL = (End MV Local - Start MV Local) * (End FX - Start FX)
        PL = End MV Local * End FX - Start MV Local * Start FX
        """
        df_return = self.position_market_value(df_prices.shift(1), df_fx
                                               ) - self.position_start_market_value(df_prices, df_fx)
        return df_return

    def position_pl_non_fx_raw(self, df_prices: Frame, df_fx: Frame) -> Frame:
        """ Get daily Non-FX PL for positions """
        df_return = self.position_market_value(df_prices, df_fx.shift(1)
                                               ) - self.position_start_market_value(df_prices, df_fx)
        return df_return

    def position_pl_cross_fx_raw(self, df_prices: Frame, df_fx: Frame) -> Frame:
        """ Get daily cross-FX PL for positions. This can be combined with fx or non-fx PL """
        df_return = self.position_pl(df_prices, df_fx) - (
                self.position_pl_fx_raw(df_prices, df_fx) + self.position_pl_non_fx_raw(df_prices, df_fx))
        return df_return

    def position_pl_non_fx(self, df_prices: Frame, df_fx: Frame) -> Frame:
        """ Get daily Non-FX PL for positions
        Non-FX PL = Start FX * (End MV Local - Start MV Local) + (End MV Local - Start MV Local) * (End FX - Start FX)
                  = End FX * (End MV Local - Start MV Local) = PL value if FX didn't change
        """
        df_return = self.position_pl_non_fx_raw(df_prices, df_fx) + self.position_pl_cross_fx_raw(df_prices, df_fx)
        return df_return

    def position_pl_fx(self, df_prices: Frame, df_fx: Frame) -> Frame:
        """ Get daily FX PL for positions
         FX PL = Start MV Local * (End FX - Start FX)
         """
        df_return = self.position_pl_fx_raw(df_prices, df_fx)
        return df_return

    def cash_pl(self, df_fx: Frame) -> Frame:
        """ Get PL for cash. This is entirely FX P&L """
        df_return = self.cash_market_value(df_fx) - self.cash_start_market_value(df_fx)
        return df_return

    def cash_pl_fx(self, df_fx: Frame) -> Frame:
        """ Get PL for cash. This is entirely FX P&L """
        return self.cash_pl(df_fx)

    def cash_pl_non_fx(self, df_fx: Frame) -> Frame:
        """ Get PL for cash. This is entirely FX P&L. So non-FX P&L = 0 """
        df_return = self.cash_pl(df_fx) * 0
        return df_return

    def cash_pl_local(self) -> Frame:
        """ Get PL for cash """
        df_return = self.cash_market_value_local() - self.cash_start_market_value_local()
        return df_return

    def portfolio_pl(self, df_prices: Frame, df_fx: Frame) -> Frame:
        """ Get PL for portfolio """
        df_return = self.portfolio_nav(df_prices, df_fx) - self.portfolio_start_nav(df_prices, df_fx)
        return df_return

    def report_all(self, df_prices: Frame, df_fx: Frame):
        """
        Overall detailed report of entire portfolio
        """
        def _melt_as(df_data: Frame, col:str) -> Frame:
            df_data = df_data.melt(value_vars=df_data.columns, ignore_index=False, var_name='Ticker',
                                   value_name=col).reset_index(names='Value Date')
            df_data = df_data.set_index(['Value Date', 'Ticker'])
            return df_data

        df_out_pos = df_out_cash = None
        # Quantity
        df_pos = self.get_positions_for_dates(df_prices.index.to_list())
        df_pos = _melt_as(df_pos, col='Quantity')
        df_cash = self.get_cash_for_dates(df_fx.index.to_list())
        df_cash = _melt_as(df_cash, col='Quantity')
        df_out_pos = pd.concat([df_out_pos, df_pos], axis=1)
        df_out_cash = pd.concat([df_out_cash, df_cash], axis=1)

        # MV Local
        df_pos = self.position_market_value_local(df_prices)
        df_pos = _melt_as(df_pos, col='MV Local')
        df_cash = self.cash_market_value_local()
        df_cash = _melt_as(df_cash, col='MV Local')
        df_out_pos = pd.concat([df_out_pos, df_pos], axis=1)
        df_out_cash = pd.concat([df_out_cash, df_cash], axis=1)

        # MV
        df_pos = self.position_market_value(df_prices, df_fx)
        df_pos = _melt_as(df_pos, col='MV')
        df_cash = self.cash_market_value(df_fx)
        df_cash = _melt_as(df_cash, col='MV')
        df_out_pos = pd.concat([df_out_pos, df_pos], axis=1)
        df_out_cash = pd.concat([df_out_cash, df_cash], axis=1)

        # Start MV Local
        df_pos = self.position_start_market_value_local(df_prices)
        df_pos = _melt_as(df_pos, col='Start MV Local')
        df_cash = self.cash_market_value_local()
        df_cash = _melt_as(df_cash, col='Start MV Local')
        df_out_pos = pd.concat([df_out_pos, df_pos], axis=1)
        df_out_cash = pd.concat([df_out_cash, df_cash], axis=1)

        # Start MV
        df_pos = self.position_start_market_value(df_prices, df_fx)
        df_pos = _melt_as(df_pos, col='Start MV')
        df_cash = self.cash_start_market_value(df_fx)
        df_cash = _melt_as(df_cash, col='Start MV')
        df_out_pos = pd.concat([df_out_pos, df_pos], axis=1)
        df_out_cash = pd.concat([df_out_cash, df_cash], axis=1)

        # PL
        df_pos = self.position_pl(df_prices, df_fx)
        df_pos = _melt_as(df_pos, col='PL')
        df_cash = self.cash_pl(df_fx)
        df_cash = _melt_as(df_cash, col='PL')
        df_out_pos = pd.concat([df_out_pos, df_pos], axis=1)
        df_out_cash = pd.concat([df_out_cash, df_cash], axis=1)

        # PL Local
        df_pos = self.position_pl(df_prices, df_fx)
        df_pos = _melt_as(df_pos, col='PL Local')
        df_cash = self.cash_pl_local()
        df_cash = _melt_as(df_cash, col='PL Local')
        df_out_pos = pd.concat([df_out_pos, df_pos], axis=1)
        df_out_cash = pd.concat([df_out_cash, df_cash], axis=1)

        # PL FX
        df_pos = self.position_pl_fx(df_prices, df_fx)
        df_pos = _melt_as(df_pos, col='PL FX')
        df_cash = self.cash_pl_fx(df_fx)
        df_cash = _melt_as(df_cash, col='PL FX')
        df_out_pos = pd.concat([df_out_pos, df_pos], axis=1)
        df_out_cash = pd.concat([df_out_cash, df_cash], axis=1)

        # PL Non-FX
        df_pos = self.position_pl_non_fx(df_prices, df_fx)
        df_pos = _melt_as(df_pos, col='PL Non-FX')
        df_cash = self.cash_pl_non_fx(df_fx)
        df_cash = _melt_as(df_cash, col='PL Non-FX')
        df_out_pos = pd.concat([df_out_pos, df_pos], axis=1)
        df_out_cash = pd.concat([df_out_cash, df_cash], axis=1)

        # Additional cols and combine
        df_out_pos['Asset Class'] = 'Position'
        df_out_cash['Asset Class'] = 'Cash'

        # Add attributes related to Tickers

        df_out = pd.concat([df_out_pos, df_out_cash], axis=0).reset_index()

        return df_out



if __name__ == '__main__':
    print("Portfolio simulation")
    print("--------------------")

    portfolio = Portfolio.generate_positions(2, nav=1000, cash_pct=.10)

    start_date = dt.date(2025, 1, 1)
    end_date = dt.date(2025, 2, 1)

    pos_tickers = portfolio.get_pos_tickers()
    ccy_tickers = portfolio.get_ccy_tickers_pos_cash()  # currencies associated with positions or cash balances
    start_price = np.array([start_prices[t] for t in pos_tickers])
    start_fx = np.array([start_fx_to_usd[t] for t in ccy_tickers])

    df_prices = sim.StockPrice.simulate_price(start_date=start_date, end_date=end_date, start_price=start_price,
                                              ticker=pos_tickers, melted_output=False)
    if len(ccy_tickers) == 0:  # all USD based
        df_fx = pd.DataFrame(index=df_prices.index)
    else:
        df_fx = sim.StockPrice.simulate_price(start_date=start_date, end_date=end_date, start_price=start_fx,
                                              ticker=ccy_tickers, melted_output=False)

    df_pos = portfolio.position_market_value(df_prices, df_fx)
    df_pos_n = portfolio.position_market_value_local(df_prices)
    df_cash = portfolio.cash_market_value(df_fx)
    df_cash_n = portfolio.cash_market_value_local()
    df_port = portfolio.portfolio_nav(df_prices, df_fx)

    df_spos = portfolio.position_start_market_value(df_prices, df_fx)
    df_spos_n = portfolio.position_start_market_value_local(df_prices)
    df_scash = portfolio.cash_start_market_value(df_fx)
    df_scash_n = portfolio.cash_start_market_value_local()
    df_sport = portfolio.portfolio_start_nav(df_prices, df_fx)

    df_pos_pl = portfolio.position_pl(df_prices, df_fx)
    df_cash_pl = portfolio.cash_pl(df_fx)
    df_port_pl = portfolio.portfolio_pl(df_prices, df_fx)

    df_pos_pl_n = portfolio.position_pl_local(df_prices)
    df_cash_pl_n = portfolio.cash_pl_local()

    df_pos_pl_fx = portfolio.position_pl_fx(df_prices, df_fx)
    df_pos_pl_non_fx = portfolio.position_pl_non_fx(df_prices, df_fx)

    df_all = portfolio.report_all(df_prices, df_fx)

    print(f"Portfolio: \n{df_port}")

    print(f"Report: \n{df_all}")
