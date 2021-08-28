import datetime as dt
import numpy as np
import pandas as pd

# Geometric Brownian Motion simulation
def gbm(s0=100, vol=0.2, t=1, rf=0.01, n_paths=10000):
    return s0 * np.exp((rf - vol*vol/2)*t + vol * np.random.standard_normal(n_paths) * np.sqrt(t))

# start_price, vol_ann, rf can be entered as numbers or as np.array for multi-stock simulated prices
def simulate_price(start_date=dt.date(2021,1,1), end_date=dt.date(2021,3,31), start_price=250, vol_ann=0.20, rf=0.06, tickers=None, include_weekend=True):
    num_stocks = start_price.size if isinstance(start_price, np.ndarray) else 1

    t = 1 / 365  # daily simulation
    count_days = (end_date-start_date).days + 1  # including start/end date

    daily_returns = np.exp((rf - vol_ann * vol_ann / 2) * t + vol_ann * np.random.standard_normal([count_days, num_stocks]) * np.sqrt(t))
    daily_returns[0] = 1.0  # zero return on first day.Line could be removed if start price = beginning of day price
    cum_returns = daily_returns.cumprod(axis=0)
    prices = start_price * cum_returns
    idx = pd.date_range(start_date, end_date)
    cols = [f'Stock {i+1}' for i in range(num_stocks)] if tickers is None or len(tickers) != num_stocks else tickers
    return_df = pd.DataFrame(prices, columns=cols, index=idx)
    if not include_weekend:
        bus_idx = pd.bdate_range(start_date, end_date)
        return_df = return_df.loc[bus_idx].copy()
    return return_df

def simulate_portfolio(start_date=dt.date(2021,1,1), end_date=dt.date(2021,6,30), assets=1000000, num_stocks=10):
    # Start MV based on roughly equal weights +/- 5%
    start_price = np.random.randint(950, 1050, num_stocks)/1000.0 * assets/num_stocks
    portfolio = simulate_price(start_date=start_date, end_date=end_date, start_price=start_price)


