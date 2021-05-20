import numpy as np
import numpy.random as npr
from pylab import plt
import scipy.stats as scs

'''
The BSM model with the Euler scheme discretisation is 
a dynamic view of the stock evolution.
The short rate and the volatility can vary in time. 
Here we apply the square root diffusion to the volatility 
(Heston stochastic volatility model). 
With the Cholesky matrix we model
the correlation between the market and v(t).
'''

S0 = 100.
r = 0.05
v0 = 0.1  # Initial value of the volatility
kappa = 3.0  # Mean reversion factor (how fast it goes to the asymptotic value)
theta = 0.25  # Long-term mean value (asymptotic value)
sigma = 0.1
rho = -0.6  # Fixed correlation between the two Brownian motions.
T = 1.0
nti = 100  # number of time intervals
n_p = 100000  # Number of paths
dt = T/nti

corr_mat = np.zeros((2, 2))
corr_mat[0, :] = [1.0, rho]
corr_mat[1, :] = [rho, 1.0]
cho_mat = np.linalg.cholesky(corr_mat)  # Cholesky decomposition
print(cho_mat)  # resulting matrix
ran_num = npr.standard_normal((2, nti + 1, n_p))  # three-dimensional random number data set.
v = np.zeros_like(ran_num[0])
vh = np.zeros_like(v)
v[0] = v0
vh[0] = v0
for t in range(1, nti + 1):
    ran = np.dot(cho_mat, ran_num[:, t, :])
    vh[t] = (vh[t - 1] +
             kappa * (theta - np.maximum(vh[t - 1], 0)) * dt +
             sigma * np.sqrt(np.maximum(vh[t - 1], 0)) *
             np.sqrt(dt) * ran[1])

v = np.maximum(vh, 0)

S = np.zeros_like(ran_num[0])
S[0] = S0
for t in range(1, nti + 1):
    ran = np.dot(cho_mat, ran_num[:, t, :])
    S[t] = S[t - 1] * np.exp((r - 0.5 * v[t]**2) * dt + np.sqrt(v[t]) * ran[0] * np.sqrt(dt))

stat = scs.describe(S[-1])
#print("mean ", stat[2], np.mean(S[-1]))
#print("std ", stat[3]**0.5, np.std((S[-1])))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8))
ax1.hist(S[-1], bins=50, weights=np.ones(len(S[-1])) / len(S[-1]))
ax1.set_xlabel('index level')
ax1.set_ylabel('frequency')
ax2.hist(v[-1], bins=50,  weights=np.ones(len(v[-1])) / len(v[-1]))
ax2.set_xlabel('volatility')

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
ax1.plot(S[:, :10], lw=1.5)
ax1.set_ylabel('index level')
ax2.plot(v[:, :10], lw=1.5)
ax2.set_xlabel('time')
ax2.set_ylabel('volatility')

plt.show()
