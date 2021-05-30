import math
import numpy as np
from pylab import plt

# Parameter Values
S0 = 100.  # initial index level
K = 105.  # strike price
T = 1.0  # time-to-maturity
r = 0.05  # riskless short rate
sigma = 0.2  # volatility
n_p = 100000  # Number of paths  # number of simulations

# Valuation Algorithm
z = np.random.standard_normal(n_p)  # pseudo-random numbers
# stock/index values at maturity (static vectorized expression)
ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * math.sqrt(T) * z)
hT = np.maximum(ST - K, 0)  # payoff at maturity
C0 = math.exp(-r * T) * np.mean(hT)  # Monte Carlo estimator

C00 = math.exp(-r * T)*hT
plt.hist(C00, bins=50, weights=np.ones(len(C00)) / len(C00))
plt.xlabel('Eur Call MC value')
plt.ylabel('frequency')

plt.savefig('Fig/a-EurC0MC.png')
# Result Output
print('Value of the European call option %5.3f.' % C0)
plt.show()