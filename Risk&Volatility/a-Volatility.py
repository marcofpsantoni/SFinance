import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Let's reproduce the S&P 500 its returns and annualised volatility first
data = pd.read_csv('../source/tr_eikon_eod_data.csv', error_bad_lines=False, index_col=0, parse_dates=True)
data = pd.DataFrame(data['.SPX'])
data.dropna(inplace=True)
data.info()
data['rets'] = np.log(data / data.shift(1))
data['vola'] = data['rets'].rolling(252).std() * np.sqrt(252)
data[['.SPX', 'rets', 'vola']].plot(subplots=True, figsize=(10, 6))

"""
Let's see the "leverage effect" between the S&P500 and the fear factor (VIX)
VIX, is an index of the implied volatility of 30-day options on the S&P 500 calculated
from a wide range of calls and puts.
We compute the correlation. 
"""
raw = pd.read_csv('../source/tr_eikon_eod_data.csv', index_col=0, parse_dates=True)
data = raw[['.SPX', '.VIX']].dropna()

# let's take a few yars to better observe the leverage effect
data.loc[:'2013-12-31'].plot(secondary_y='.VIX', figsize=(10, 6))


# The correlation between the returns and the scatter plot
rets = np.log(data / data.shift(1))
rets.dropna(inplace=True)
reg = np.polyfit(rets['.SPX'], rets['.VIX'], deg=1)
ax = rets.plot(kind='scatter', x='.SPX', y='.VIX', figsize=(10, 6))
ax.plot(rets['.SPX'], np.polyval(reg, rets['.SPX']), 'r', lw=2)

# The tot average correlation and the rolling 1 year corr
plt.figure(figsize=(10, 6))
ax = rets['.SPX'].rolling(window=252).corr(rets['.VIX']).plot()
ax.axhline(rets.corr().iloc[0, 1], c='r')

plt.show()

