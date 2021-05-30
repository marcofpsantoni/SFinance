import math
import numpy as np
import pandas as pd
from pylab import plt
import statsmodels.api as sm
import scipy.stats as scs

'''
Let's merge 2 data set of S&P 500
'''

filename = '../data/tr_eikon_eod_data.csv'
raw = pd.read_csv(filename, index_col=0, parse_dates=True).dropna()
symbol = '.SPX'
data = pd.DataFrame(raw[symbol])
data['returns'] = np.log(data / data.shift(1))
data.dropna(inplace=True)

filename2 = '../data/SPY20052021HistoricalData.csv'
raw2 = pd.read_csv(filename2, index_col=0, parse_dates=True).dropna()
symbol2 = 'Price'
data2 = pd.DataFrame(raw2[symbol2])
data2['returns'] = np.log(data2 / data2.shift(-1))
data2.dropna(inplace=True)
data2.head()

d2 = data2.iloc[::-1]
d2 = d2.loc['2018-07-02':]
data = data.append(d2, sort=True)

#data['returns'].cumsum().apply(np.exp).plot(legend=True, figsize=(8, 8))

mu = data.returns.mean() * 252  # Calculates the annualized return.
sigma = data.returns.std() * 252 ** 0.5  # Calculates the annualized volatility.
r = 0.0179  # 1 year treasury rate
f = (mu - r) / sigma ** 2  # Calculates the optimal Kelly fraction to be invested in the strategy
print(f)
equs = []  # preallocating space for our simulations


def kelly_strategy(f):
    global equs
    equ = 'equity_{:.2f}'.format(f)
    equs.append(equ)
    cap = 'capital_{:.2f}'.format(f)
    data[equ] = 1  # Generates a new column for equity and sets the initial value to 1.
    data[cap] = data[equ] * f  # Generates a new column for capital and sets the initial value to 1*f.
    for i, t in enumerate(data.index[1:]):
        t_1 = data.index[i]  # Picks the right DatetimeIndex value for the previous values.
        data.loc[t, cap] = data[cap].loc[t_1] * math.exp(data['returns'].loc[t])
        data.loc[t, equ] = data[cap].loc[t] - data[cap].loc[t_1] + data[equ].loc[t_1]
        data.loc[t, cap] = data[equ].loc[t] * f


kelly_strategy(f * 0.5)  # Values for 1/2 KC
kelly_strategy(f * 0.66)  # Values for 2/3 KC
kelly_strategy(f)  # Optimal KC
ax = data['returns'].cumsum().apply(np.exp).plot(legend=True, figsize=(8, 8))
plt.title('Varied KC Values on SPY, Starting from $1')
plt.xlabel('Years')
plt.ylabel('$ Return')
data[equs].plot(ax=ax, legend=True)
plt.savefig('Fig/b-1-KellySP500')

"""
The optimal KC gives big jumps. The strategy is not smooth.
Let's see how much the S&P500 returns devietaes from a bell curve
"""


def print_statistics(array):

    '''
    Prints selected statistics.
    Parameters
    ==========
    array: ndarray
    object to generate statistics on
    '''

    sta = scs.describe(array)
    print('%14s %15s' % ('statistic', 'value'))
    print(30 * '-')
    print('%14s %15.5f' % ('size', sta[0]))
    print('%14s %15.5f' % ('min', sta[1][0]))
    print('%14s %15.5f' % ('max', sta[1][1]))
    print('%14s %15.5f' % ('mean', sta[2]))
    print('%14s %15.5f' % ('std', np.sqrt(sta[3])))
    print('%14s %15.5f' % ('skew', sta[4]))
    print('%14s %15.5f' % ('kurtosis', sta[5]))


print_statistics(data['returns'])
print(data['returns'].describe())
sm.qqplot(data['returns'], line='s')
plt.title('SPY')
plt.xlabel('theoretical quantiles')
plt.ylabel('sample quantiles')
plt.savefig('Fig/b-2-SP500GaussDeviation')

plt.show()

