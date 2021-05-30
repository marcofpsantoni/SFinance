#
# Valuation of European call options
# in Black-Scholes-Merton model
# Vega and Implied Volatility
#
import pandas as pd
import numpy as np
from pylab import plt
import scipy.interpolate as spi


def bsm_call_value(S0, K, T, r, sigma,  option='call'):
    ''' Valuation of European call option in BSM model.
    Analytical formula.

    Parameters
    ==========
    S0: float
        initial stock/index level
    K: float
        strike price
    T: float
        maturity date (in year fractions)
    r: float
        constant risk-free short rate
    sigma: float
        volatility factor in diffusion term

    Returns
    =======
    value: float
        present value of the European call option
    '''
    from math import log, sqrt, exp
    from scipy import stats

    #print(stats.norm.cdf(40, 4, 30))
    S0 = float(S0)
    d1 = (log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = (log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    # stats.norm.cdf --> cumulative distribution function
    #                    for normal distribution
    if option == 'call':
        value = (S0 * stats.norm.cdf(d1, 0.0, 1.0) -
                K * exp(-r * T) * stats.norm.cdf(d2, 0.0, 1.0))
    else:
        value = (-S0 * stats.norm.cdf(-d1, 0.0, 1.0) +
                K * exp(-r * T) * stats.norm.cdf(-d2, 0.0, 1.0))
    return value


def bsm_vega(S0, K, T, r, sigma):
    ''' Vega of European option in BSM model.

    Parameters
    ==========
    S0: float
        initial stock/index level
    K: float
        strike price
    T: float
        maturity date (in year fractions)
    r: float
        constant risk-free short rate
    sigma: float
        volatility factor in diffusion term

    Returns
    =======
    vega: float
        partial derivative of BSM formula with respect
        to sigma, i.e. Vega

    '''
    from math import log, sqrt
    from scipy import stats

    S0 = float(S0)
    d1 = (log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    vega = S0 * stats.norm.pdf(d1, 0.0, 1.0) * sqrt(T)
    return vega


# Implied volatility function
def bsm_call_imp_vol(S0, K, T, r, C0, sigma_est, it=100):
    ''' Implied volatility of European call option in BSM model.

    Parameters
    ==========
    S0: float
        initial stock/index level
    K: float
        strike price
    T: float
        maturity date (in year fractions)
    r: float
        constant risk-free short rate
    C0: float
        call value
    sigma_est: float
        estimate/starting point impl. volatility
    it: integer
        number of iterations

    Returns
    =======
    simga_est: float
        numerically estimated implied volatility
    '''
    for i in range(it):
        sigma_est -= ((bsm_call_value(S0, K, T, r, sigma_est) - C0) /
                      bsm_vega(S0, K, T, r, sigma_est))
    return sigma_est

#Let's check the function for the imp vol
print(bsm_call_imp_vol(100, 105, 1, 0.02, bsm_call_value(100, 105, 1, 0.02, 0.2), 0.9))

# Let's reproduce the dataframe Volatility in Hull pg 438

tempo = [1/12, 1/4, 1/2, 1, 2, 5]
colonne = [0.9, 0.95, 1, 1.05, 1.1]
df = pd.DataFrame(columns=colonne, index=tempo)
df[0.9] = [14.2, 14.0, 14.1, 14.7, 15.0, 14.8]
df[0.95] = [13.0, 13.0, 13.3, 14.0, 14.4, 14.6]
df[1] = [12.0, 12.0, 12.5, 13.5, 14.0, 14.4]
df[1.05] = [13.1, 13.1, 13.4, 14.0, 14.5, 14.7]
df[1.1] = [14.5, 14.2, 14.3, 14.8, 15.1, 15.0]

# Mesh for the Surface
x = np.array(df.columns)
y = np.array(df.index)
X, Y = np.meshgrid(x, y)
xd = np.linspace(0.9, 1.1, 25)
yd = np.linspace(1/12, 5, 25)
Xd, Yd = np.meshgrid(xd, yd)
f = spi.interp2d(X, Y, df, kind='linear')
#splines (kind ='cubic') do not work! Why?
sig = f(1.075, 1.5)
print(sig)
print(bsm_call_value(100, 105, 1.5, 0.01, sig/100))
zd = f(xd, yd)

fig = plt.figure(figsize=(8, 8))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, df, rstride=2, cstride=2,
                       cmap='coolwarm', linewidth=0.5,
                       antialiased=True)
surf3 = ax.plot_wireframe(Xd, Yd, zd, rstride=2, cstride=2,
                          label='interpolation', antialiased=True)
ax.set_xlabel('K/S0')
ax.set_ylabel('Time in years')
ax.set_zlabel('Imp Vol')
ax.legend()
fig.colorbar(surf3, shrink=0.5, aspect=5)
plt.savefig('Fig/b-ImpVolInterpolation.png')

plt.show()
