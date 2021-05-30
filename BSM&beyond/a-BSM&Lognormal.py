import math
import numpy as np
from pylab import plt

"""
The lognormal distribuion is the core of the BSM model.
Let's implement a simple MC simulation in a dynamic fashion.
"""

nti = 100  # number of time intervals
n_p = 50000  # Number of paths
S0 = 100.  # Initial Value
T = 1.0  # Time in years (1 year)
r = 0.06  # Riskless interest rate
sigma = 0.2  # Volatility
dt = T/nti  # Deltat = single time interval

'''
Using the numpy power
'''

S = np.zeros((nti+1, n_p))
rn = np.random.standard_normal((nti+1, n_p))
S[0] = S0
for t in range(1, nti + 1):
    S[t] = S[t - 1] * np.exp((r - sigma ** 2 / 2) * dt + sigma * math.sqrt(dt) * rn[t])
plt.figure(figsize=(8, 8))
plt.hist(S[-1], bins=50, weights=np.ones(len(S[-1])) / len(S[-1]))
plt.xlabel('stock/index value')
plt.ylabel('frequency')
plt.savefig('Fig/a-LogNorm.png')
plt.show()
