import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as sco
import scipy.interpolate as sci

"""
Let's assume that past mean returns and correlations 
are a good indicator for the future beahavior of 2 stocks + 1 index + 1 commodity
Let's find out the best portfolio form historical data
"""

filename = '../data/tr_eikon_eod_data.csv'
raw = pd.read_csv(filename).dropna()
symbols = ['AAPL.O', 'MSFT.O', 'SPY', 'GLD']
r = 0.015
noa = len(symbols)
data = raw[symbols]
rets = np.log(data / data.shift(1))
# rets.hist(bins=40, figsize=(8, 8))

"""
Two functions: annualized mean and st. dev.
"""


def port_ret(w):
    return np.sum(rets.mean() * w) * 252


def port_vol(w):
    return np.sqrt(np.dot(w.T, np.dot(rets.cov() * 252, w)))


"""
MC Simulation
"""

mcps = 2500  # number of MC portfolio simulations
prets = np.zeros(mcps)
pvols = np.zeros_like(prets)
for p in range(mcps):
    weights = np.random.random(noa)
    weights /= np.sum(weights)
    prets[p] = port_ret(weights)
    pvols[p] = port_vol(weights)

"""
The negative SR: we want to minimize -SR
"""


def neg_sharpe_ratio(w, rr=r):
    return -(port_ret(w)-rr) / port_vol(w)


cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})  # The sum of the weights must be 1
bnds = tuple((0, 1) for x in range(noa))  # each weight 0<=w<=1
eweights = np.array(noa * [1. / noa])  # starting point w=1/noa
# The optimization of the portfolio is the minimization of -SR
opts = sco.minimize(neg_sharpe_ratio, eweights, method='SLSQP', bounds=bnds, constraints=cons)
print(opts)

"""
Let's find the efficient frontier.
The minimization of the variance (or volaility) 
for target returns (an additional constraint)
"""
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

plt.figure(figsize=(8, 8))
plt.scatter(pvols, prets, c=prets / pvols, marker='.', alpha=0.8, cmap='coolwarm')
plt.plot(tvols, trets, 'b', lw=3.0)
plt.plot(port_vol(opts['x']), port_ret(opts['x']), 'y*', markersize=15.0)
plt.plot(port_vol(optv['x']), port_ret(optv['x']), 'r*', markersize=15.0)
plt.xlabel('expected volatility')
plt.grid(True)
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
plt.savefig('Fig/b-ScatterEffFront.png')


"""
I want the tangent line fron riskless interest rate:
Capital market line must be done with efficient frontier
the intercept is rf
"""

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


def equations(p, rf=r):
    eq1 = rf - p[0]
    eq2 = rf + p[1] * p[2] - f(p[2])
    eq3 = p[1] - df(p[2])
    return eq1, eq2, eq3


opt = sco.fsolve(equations, [0.01, 0.5, 0.15])
# print(np.round(equations(opt), 6))
plt.figure(figsize=(12, 8))
plt.scatter(pvols, prets, c=(prets - 0.01) / pvols, marker='.', cmap='coolwarm')
plt.plot(evols, erets, 'b', lw=4.0)
cx = np.linspace(0.0, 0.3)
plt.plot(cx, opt[0] + opt[1] * cx, 'r', lw=1.5)
plt.plot(opt[2], f(opt[2]), 'y*', markersize=15.0)
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')



cons = ({'type': 'eq', 'fun': lambda x:  port_ret(x) - f(opt[2])},
        {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
res = sco.minimize(port_vol, eweights, method='SLSQP', bounds=bnds, constraints=cons)
# print(f(opt[2]), opt[2], f(opt[2])/opt[2])
print(res['x'].round(4))
"""
The portfolio is very close to the sharpe ratio maximization one.
More instruments could bring to a different result. 
"""
# print(port_ret(res['x']))
# print(port_vol(res['x']))
# print(port_ret(res['x']) / port_vol(res['x']))
plt.savefig('Fig/b-CapMarketLine.png')
plt.show()
