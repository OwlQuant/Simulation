# Simulation Library
Numpy based fast library for  
* Simulating Geometric Brownian Motion
* Simulating Stock Prices
* Constructing Portfolios 


## Examples:
Geometic Brownian Motion
```python
import simulation as sim 
import numpy as np

# Basic Usage with default parameters 
sim.StockPrice.gbm()

# Simulate 5 stocks
sim.StockPrice.gbm(s0=np.ones(5) * 1000, vol=0.2, t=0.1, 
                   rf=0.01, corr=0.75, n_paths=10000)

```

Stock Prices
```python
import simulation as sim 
import numpy as np
import datetime as dt

# Basic Usage with default parameters 
sim.StockPrice.simulate_price()

# Simulate prices for 20 stocks between start and end dates 
sim.StockPrice.simulate_price(start_date=dt.date(2021,1,1), end_date=dt.date(2022,1,1),
                              start_price=np.ones(20) * 1000,
                              vol_ann=0.15, corr=0.2, melted_output=False)

# Simulate 3 stocks with provided tickers and granular control of parameters
sim.StockPrice.simulate_price(start_date=dt.date(2021,1,1), end_date=dt.date(2025,1,1),
                              start_price=np.array([750, 500, 200]), ticker=['AAPL', 'FB', 'MMM'],
                              vol_ann=np.array([.10, .10, 0.10]), rf=0.01,
                              corr=np.array([[1, .75, .5], [.75, 1, -.10], [.5, -.1, 1]]),
                              melted_output=False)



```