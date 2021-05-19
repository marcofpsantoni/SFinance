'''
Merton's Jump with constant volatility
Evolution of a stock including a jump diffusion model
you can model a bimodal distributions (left fat tails).
'''

import numpy as np
import numpy.random as npr
from pylab import plt


#Numerical parametrization

S0 = 100. # stock initial value
r = 0.05 # constant (short riskless) interest rate
sigma = 0.2 # constant volatility
lamb = 0.75 # poissonian lambda for the jump
mu = -0.6 # poissonian mu of the jump (time - independent)
delta = 0.25 # jump volatility
rj = lamb * (np.exp(mu + 0.5 * delta ** 2) - 1)  # drift correction

T = 1.0
M = 50
I = 10000
dt = T / M

S = np.zeros((M + 1, I))
S[0] = S0
sn1 = npr.standard_normal((M + 1, I))  # normal variable for the stock
sn2 = npr.standard_normal((M + 1, I))  # normal variable for the jump (volatility)
poi = npr.poisson(lamb * dt, (M + 1, I))  # poisson variable for the jump
for t in range(1, M + 1, 1):
    S[t] = S[t - 1] * (np.exp((r - rj - 0.5 * sigma ** 2) * dt +
                    sigma * np.sqrt(dt) * sn1[t]) +
                    (np.exp(mu + delta * sn2[t]) - 1) *
                    poi[t])
    S[t] = np.maximum(S[t], 0)

plt.figure(figsize=(10, 6))
plt.hist(S[-1], bins=50)
plt.xlabel('value')
plt.ylabel('frequency')

plt.figure(figsize=(10, 6))
plt.plot(S[:, :10], lw=1.5)
plt.xlabel('time')
plt.ylabel('index level')

plt.show()
