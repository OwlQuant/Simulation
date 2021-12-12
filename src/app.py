# app.py
from flask import Flask, request, jsonify
# import json
import numpy as np

# import simulation.stock as stock
# import src.simulation as sim
# from simulation import .src.simulation import gbm
# from src import simulation as sim
# import simulation as sim
app = Flask(__name__, static_folder="./src")

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
    #gbm = sim.StockPrice.gbm(s0=s, vol=v_raw, t=t_raw, rf=r_raw, corr=c_raw, n_paths=n_raw)
    gbm = None
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
