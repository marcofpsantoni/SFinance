import math
import numpy as np
from pylab import plt

"""
The lognormal distribuion is the core of the BSM model.
Let's implement a simple MC simulation in a dynamic fashion.
"""

M = 100  # Time intervals
I = 50000  # Paths
S0 = 100.  # Initial Value
T = 1.0  # Time in years
r = 0.06  # Riskless interest rate
sigma = 0.2  # Volatility
dt = M/I  # time interval

'''
Using numpy power
'''

S = np.zeros((M+1, I))
rn = np.random.standard_normal((M+1, I))
S[0] = S0
for t in range(1, M + 1):
    S[t] = S[t - 1] * np.exp((r - sigma ** 2 / 2) * dt + sigma * math.sqrt(dt) * rn[t])
plt.figure(figsize=(10, 6))
plt.hist(S[-1], bins=50, weights=np.ones(len(S[-1])) / len(S[-1]))
plt.xlabel('index level')
plt.ylabel('frequency')
plt.show()
