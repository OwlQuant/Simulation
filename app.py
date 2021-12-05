# app.py
from flask import Flask

import simulation.stock as stock
import simulation as sim

app = Flask(__name__)

@app.route('/stock/')
def stock_simulation():
    sim.Simulation
    p = stock.simulate_price()
    p.insert(0, "Date", p.index)
    p["Date"] = p["Date"].dt.strftime("%Y-%m-%d")
    # print(p.to_dict())
    return p.to_json(orient="values", indent=2)

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