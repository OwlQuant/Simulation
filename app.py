# app.py
from flask import Flask, request, jsonify
import datetime as dt
import numpy as np
import pandas as pd
import simulation as sim

# import simulation.stock as stock
# import src.simulation as sim
# from simulation import .src.simulation import gbm
# from src import simulation as sim
# import simulation as sim
app = Flask(__name__)

@app.route('/api/gbm/v1', methods=['GET'])
def gbm_simulation():
    """
    Generate GBM Simulation using parameters from request
    Get method provides simplified parameters. Post method provides full control and access
    Parameters Implied in Request object
    s   float start price. Could be array or float
    v   float volatility in numeric
    t   float time in years
    r   float riskfree rate
    c   float correlation. Could be flat correlation or number < 1
    n   int number of simulations

    :return: 2D array of simulated prices
    """
    s_raw = request.args.get("s", default="1000", type=str)
    v_raw = request.args.get("v", default=0.20, type=float)
    t_raw = request.args.get("t", default=1.0, type=float)
    r_raw = request.args.get("r", default=0.0, type=float)
    c_raw = request.args.get("c", default=0.0, type=float)
    n_raw = request.args.get("n", default=100, type=int)
    try:
        s = [float(x) for x in s_raw.split(",")]
    except ValueError:
        print("Could not parse s into float/[float]: " + s_raw)
        s = [1000]
    s = np.array(s)
    gbm = sim.StockPrice.gbm(s0=s, vol=v_raw, t=t_raw, rf=r_raw, corr=c_raw, n_paths=n_raw)
    # gbm = None
    # print(s_raw,v_raw,t_raw, r_raw, c_raw, n_raw)
    # gbm = sim.StockPrice.gbm(s0=1000, vol=0.2, t=1.1, rf=0.01, n_paths=10)

    # print(gbm.shape)

    # sim.Simulation
    # p = stock.simulate_price()
    # p.insert(0, "Date", p.index)
    # p["Date"] = p["Date"].dt.strftime("%Y-%m-%d")
    # # print(p.to_dict())
    # return p.to_json(orient="values", indent=2)
    return jsonify(gbm.tolist())

@app.route('/api/stockprice/v1', methods=['GET'])
def stock_simulation():
    """
    Generate Stock Simulation using parameters from request
    Get method provides simplified parameters. Post method provides full control and access
    Parameters Implied in Request object
    from   Date    specifies start date
    to     Date    specifies End Date
    price   float  start price as float or list of float
    v       float   volatility
    r       float riskfree rate
    c       float correlation. Could be flat correlation or number < 1
    ticker  String  ticker or list of tickers

    :return: 2D array of simulated prices
    """

    from_raw = request.args.get("from", default=None, type=lambda x: dt.datetime.strptime(x, "%Y-%m-%d").date())
    to_raw = request.args.get("to", default=None, type=lambda x: dt.datetime.strptime(x, "%Y-%m-%d").date())
    price_raw = request.args.get("price", default="1000", type=str)
    v_raw = request.args.get("v", default=0.02, type=float)
    r_raw = request.args.get("r", default=0.0, type=float)
    c_raw = request.args.get("c", default=0.0, type=float)
    ticker_raw = request.args.get("ticker", default=None, type=str)

    try:
        price = [float(x) for x in price_raw.split(",")]
    except ValueError:
        print("Could not parse s into float/[float]: " + price_raw)
        price = [1000]
    price = np.array(price)

    if ticker_raw is not None:
        ticker = [x for x in ticker_raw.split(",")]
        ticker = np.array(ticker)
    else:
        ticker = None

    df_prices = sim.StockPrice.simulate_price(start_date=from_raw, end_date=to_raw, start_price=price,
                                              ticker=ticker, vol_ann=v_raw, rf=r_raw, corr=c_raw, melted_output=False)
    print(from_raw,to_raw,price_raw, v_raw, r_raw, c_raw, ticker_raw)
    # print(df_prices)

    # sim.Simulation
    # p = stock.simulate_price()
    # p.insert(0, "Date", p.index)
    # p["Date"] = p["Date"].dt.strftime("%Y-%m-%d")
    # # print(p.to_dict())
    # return p.to_json(orient="values", indent=2)
    df_prices['Value Date'] = df_prices.index.strftime("%Y-%m-%d")
    return jsonify(df_prices.to_dict(orient='records'))

@app.route('/')
def index():
    text = '''
    <!DOCTYPE html>
<html>
  <head>
    <title>simulation</title>
  </head>
  <body>
    <h3>Available Simulations: </h3>
  </body>
</html>
    '''
    return text


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)
