import simulation as sim
import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt


def gbm_test():
    print("GBM simulation")
    # GBM simulation
    gbm = sim.StockPrice.gbm(s0=np.ones(5) * 1000, vol=0.2, t=0.1, rf=0.01, corr=1, n_paths=10000)
    # gbm = sim.StockPrice.gbm(s0=1000, vol=0.2, t=1.1, rf=0.01, n_paths=10)

    plt.scatter(gbm[:, 0], gbm[:, 1])
    plt.show()
    print(gbm)


def price_test():
    print("Stock Price simulation")
    start_date = dt.date(2021, 1, 1)
    end_date = dt.date(2053, 1, 1)

    df = sim.StockPrice.simulate_price(start_date=start_date, end_date=end_date,
                                       start_price=np.array([1000, 1000, 1000]), ticker=['Aapl', 'Fb', 'Goog'],
                                       vol_ann=np.array([.10, .10, 0.10]), rf=0.01,
                                       corr=np.array([[1, .75, .5], [.75, 1, -.10], [.5, -.1, 1]]),
                                       melted_output=False)

    df = sim.StockPrice.simulate_price(start_date=start_date, end_date=end_date, start_price=np.ones(200) * 1000,
                                       vol_ann=0.15, corr=0.2, melted_output=False)

    df.to_clipboard()
    print(df)


def portfolio_test():
    # print("Portfolio simulation")
    start_date = dt.date(2021, 1, 1)
    end_date = dt.date(2031, 1, 1)
    num_stocks = 500
    # tickers = ['AAPL', 'GOOG', 'FB', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    tickers = sim.StockPrice.get_default_tickers(num_stocks)
    # ['AAPL', 'GOOG', 'FB', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    initial_values = np.ones(num_stocks) * 100
    # [100, 150, 200, 10, 10, 10, 10, 10, 10, 10, 10]

    positions = pd.DataFrame([initial_values], columns=tickers, index=['Start Value']).transpose()
    positions.index.name = 'Ticker'

    df_prices = sim.StockPrice.simulate_price(start_date=start_date, end_date=end_date, start_price=initial_values,
                                              ticker=tickers)
    # print(positions)
    portfolio = sim.Portfolio.from_initial_positions(positions)
    # df = portfolio._simulate(df_prices_melted)
    df = portfolio.portfolio_value(df_prices)
    # print(df)
    return df


def portfolio_test2():
    # print("Portfolio simulation")
    start_date = dt.date(2021, 1, 1)
    end_date = dt.date(2022, 1, 1)
    num_stocks = 500
    nav = 100000
    portfolio = sim.Portfolio.from_num_positions(num_stocks, nav)
    tickers = np.array(portfolio.positions.index)
    initial_values = portfolio.positions.to_numpy().flatten()

    df_prices = sim.StockPrice.simulate_price(start_date=start_date, end_date=end_date, start_price=initial_values,
                                              ticker=tickers)
    # print(positions)
    df = portfolio._simulate(df_prices)
    df = portfolio.portfolio_value(df_prices)
    # print(df)
    return df


if __name__ == '__main__':
    # gbm_test()
    price_test()
    # print(portfolio_test())
