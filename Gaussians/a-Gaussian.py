from math import log, sqrt, exp, erf, pi
from scipy import stats
import numpy as np
from pylab import mpl, plt
import scipy.stats as scs


S0 = 100.
K = 150.
r = 0.02
sigma = 0.5
T = 1

d1 = (log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
d2 = (log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
# stats.norm.cdf --> cumulative distribution function
#                    for normal distribution

value = (S0 * stats.norm.cdf(d1, 0.0, 1.0) - K * exp(-r * T) * stats.norm.cdf(d2, 0.0, 1.0))

#Another way is to use the commection between erf and cumulative
def gaus_cdf(x, m=0., s=1.):
    x = (x-m)/(sqrt(2)*s)
    return 0.5*(1+erf(x))

v2 = S0 * gaus_cdf(d1) - K * exp(-r * T) * gaus_cdf(d2)
#They are roughly the same:
#print(value, v2)

#plot normal distrubtion
domain = np.linspace(-3, 3, 1000)
plt.plot(domain, scs.norm.pdf(domain, 0, 1))
plt.plot(domain, scs.norm.cdf(domain, 0, 1))
plt.axvline(d1, color='r', ls='--', lw=1.0)
plt.axvline(d2, color='b', ls='--', lw=1.0)
plt.savefig('Fig/a-Gauss')
plt.show()
