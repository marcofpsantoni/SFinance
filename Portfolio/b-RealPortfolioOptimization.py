import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
Let's assume that the past correlation is a good indicator
for the future correlation of 2 stocks + 1 index + 1 commodity
Let's find the best portfolio form historical data
"""

filename = '../source/tr_eikon_eod_data.csv'
raw = pd.read_csv(filename).dropna()
symbols = ['AAPL.O', 'MSFT.O', 'SPY', 'GLD']
r = 0.01
noa = len(symbols)
data = raw[symbols]
rets = np.log(data / data.shift(1))
#rets.hist(bins=40, figsize=(10, 8))


def port_ret(weights):
    return np.sum(rets.mean() * weights) * 252


def port_vol(weights):
    return np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))


prets = []
pvols = []
for p in range(2500):
    weights = np.random.random(noa)
    weights /= np.sum(weights)
    prets.append(port_ret(weights))
    pvols.append(port_vol(weights))
prets = np.array(prets)
pvols = np.array(pvols)

import scipy.optimize as sco


def min_func_sharpe(weights):
    return -port_ret(weights) / port_vol(weights)

#I put -r to consider sharpe-ratio
def min_func_sharpe_with_riskless_int(weights, r):
    return -(port_ret(weights)-r) / port_vol(weights)


cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
bnds = tuple((0, 1) for x in range(noa))
eweights = np.array(noa * [1. / noa])
opts = sco.minimize(min_func_sharpe, eweights, method='SLSQP', bounds=bnds, constraints=cons)
print(opts)
optv = sco.minimize(port_vol, eweights, method='SLSQP', bounds=bnds, constraints=cons)

cons = ({'type': 'eq', 'fun': lambda x:  port_ret(x) - tret},
                 {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})

bnds = tuple((0, 1) for x in weights)

trets = np.linspace(0.05, 0.2, 50)
tvols = []
for tret in trets:
    res = sco.minimize(port_vol, eweights, method='SLSQP', bounds=bnds, constraints=cons)
    tvols.append(res['fun'])
tvols = np.array(tvols)
m = (port_ret(opts['x']) / port_vol(opts['x']))
plt.figure(figsize=(10, 6))
plt.scatter(pvols, prets, c=prets / pvols, marker='.', alpha=0.8, cmap='coolwarm')
plt.plot(tvols, trets, 'b', lw=3.0)
plt.plot(port_vol(opts['x']), port_ret(opts['x']),'y*', markersize=15.0)
plt.plot(port_vol(optv['x']), port_ret(optv['x']), 'r*', markersize=15.0)
plt.xlabel('expected volatility')
plt.grid(True)
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')

"""
I want the tangent line fron riskless interest rate:
Capital market line must be done with efficient frontier
"""


import scipy.interpolate as sci

ind = np.argmin(tvols)  # Index position of minimum volatility portfolio
#print(ind)
evols = tvols[ind:]  # Relevant portfolio volatility values
erets = trets[ind:]  # Relevant portfolio return values
#print(evols[2:5], erets[2:5])
tck = sci.splrep(evols, erets)  # Cubic splines interpolation on these values.


def f(x):
    ''' Efficient frontier function (splines approximation). '''
    return sci.splev(x, tck, der=0)


def df(x):
    ''' First derivative of efficient frontier function. '''
    return sci.splev(x, tck, der=1)


def equations(p, rf=0.01):
    eq1 = rf - p[0]
    eq2 = rf + p[1] * p[2] - f(p[2])
    eq3 = p[1] - df(p[2])
    return eq1, eq2, eq3


opt = sco.fsolve(equations, [0.01, 0.5, 0.15])
print(np.round(equations(opt), 6))

plt.figure(figsize=(10, 6))
plt.scatter(pvols, prets, c=(prets - 0.01) / pvols, marker='.', cmap='coolwarm')
plt.plot(evols, erets, 'b', lw=4.0)
cx = np.linspace(0.0, 0.3)
plt.plot(cx, opt[0] + opt[1] * cx, 'r', lw=1.5)
plt.plot(opt[2], f(opt[2]), 'y*', markersize=15.0)
plt.grid(True)
#plt.axhline(0, color='k', ls='--', lw=2.0)
#plt.axvline(0, color='k', ls='--', lw=2.0)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')

cons = ({'type': 'eq', 'fun': lambda x:  port_ret(x) - f(opt[2])},
        {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
res = sco.minimize(port_vol, eweights, method='SLSQP', bounds=bnds, constraints=cons)

#print(f(opt[2]), opt[2], f(opt[2])/opt[2])
print(res['x'].round(4))
#print(port_ret(res['x']))
#print(port_vol(res['x']))
#print(port_ret(res['x']) / port_vol(res['x']))

plt.show()