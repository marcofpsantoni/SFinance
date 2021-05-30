import numpy as np
import numpy.random as npr
import math
from pylab import plt
import scipy.stats as scs

'''
Let's see VaR and ES with a simple simulation.

Value-at-risk (VaR) is a number denoted in currency units 
(e.g., USD, EUR, JPY) indicating a loss (of a portfolio, a single position, etc.) that is not exceeded 
with some confidence level (probability) over a given period of time. Consider a stock position, 
worth 1 million USD today, that has a VaR of 50,000 USD at a confidence level of 99% over a 
time period of 30 days (one month). 
The metric says that with a probability of 99%, the loss to be expected over a period of 
30 days will not exceed 50,000 USD. However, it does not say anything about the size of the loss once 
a loss beyond 50,000 USD occurs — i.e., if the maximum loss is 100,000 or 500,000 USD what the probability 
of such a specific “higher than VaR loss” is. All it says is that there is a 1% probability that a loss of 
a minimum of 50,000 USD or higher will occur. 

Assume the Black-Scholes-Merton setup and consider the 
following parameterization and simulation of index levels at a future date upper T = 30 /365 
(a period of 30 days). The estimation of VaR figures requires the simulated absolute profits and 
losses relative to the value of the position today in a sorted manner, i.e., 
from the severest loss to the largest profit.

ES is the average of the left tail.

'''

S0 = 100
r = 0.05
sigma = 0.25
T = 30 / 365.
n_p = 100000  # Number of paths

# MC simulation for GBM
ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * npr.standard_normal(n_p))
R_gbm = np.sort(ST - S0)
plt.figure(figsize=(8, 8))
plt.hist(R_gbm, bins=50, weights=np.ones(len(R_gbm)) / len(R_gbm))
plt.xlabel('Absolute return for Geometric Brownian Motion')
plt.ylabel('Frequency')
plt.savefig('Fig/c-1-GRWRet.png')
# Lets's add some jumps a la Merton

nti = 100  # number of time intervals
lamb = 0.5  # poissonian lambda for the jump
mu = -0.6  # poissonian mu of the jump (time - independent)
delta = 0.25  # jump volatility

dt = 30. / 365 / nti
rj = lamb * (math.exp(mu + 0.5 * delta ** 2) - 1)

S = np.zeros((nti + 1, n_p))
S[0] = S0
sn1 = npr.standard_normal((nti + 1, n_p))
sn2 = npr.standard_normal((nti + 1, n_p))
poi = npr.poisson(lamb * dt, (nti + 1, n_p))
for t in range(1, nti + 1):
    S[t] = S[t - 1] * (np.exp((r - rj - 0.5 * sigma ** 2) * dt
                    + sigma * math.sqrt(dt) * sn1[t])
                    + (np.exp(mu + delta * sn2[t]) - 1)
                    * poi[t])
    S[t] = np.maximum(S[t], 0)

R_jd = np.sort(S[-1] - S0)
plt.figure(figsize=(8, 8))
plt.hist(R_jd, bins=50, weights=np.ones(len(R_jd)) / len(R_jd))
plt.xlabel('absolute return GRW + JD')
plt.ylabel('frequency')
plt.savefig('Fig/c-2-GRW+JDRet.png')


#A Table with some ES and VaR values

percs2 = [0.01, 0.1, 1., 2.5, 5.0, 10.0]
jd_var1 = scs.scoreatpercentile(R_jd, 1)
gbm_var2 = scs.scoreatpercentile(R_gbm, percs2)
jd_var2 = scs.scoreatpercentile(R_jd, percs2)
jd_es, grw_es = [], []

# The table
print(100 * '-')
print('%16s %16s %16s %16s %16s' % ('CL', 'VaR GBM', 'VaR JD', 'ES BSM', 'ES BSM+JD'))
for j in range(len(percs2)):
    jd_es.append(np.mean([R_jd[i] for i in range(len(R_jd)) if R_jd[i] <= jd_var2[j]]))
    grw_es.append(np.mean([R_gbm[i] for i in range(len(R_gbm)) if R_gbm[i] <= gbm_var2[j]]))
    print('%16.2f %16.3f %16.3f %16.3f %16.3f ' % (100 - percs2[j], -gbm_var2[j], -jd_var2[j], -grw_es[j], -jd_es[j]))
print('%16s %16s %16s %16s %16s' % ('CL', 'VaR GBM', 'VaR JD', 'ES BSM', 'ES BSM+JD'))
print(100 * '-')

# A plot with the 2 VaRs and 2 ESs
percs = list(np.arange(0.0, 10.1, 0.1))
gbm_var = scs.scoreatpercentile(R_gbm, percs)
jd_var = scs.scoreatpercentile(R_jd, percs)
es_grw = np.array([np.mean([R_gbm[i] for i in range(len(R_gbm)) if R_gbm[i] <= gbm_var[j]]) for j in range(len(percs))])
es_jd = np.array([np.mean([R_jd[i] for i in range(len(R_jd)) if R_jd[i] <= jd_var[j]]) for j in range(len(percs))])
es_jd_std = np.array([np.std([R_jd[i] for i in range(len(R_jd)) if R_jd[i] <= jd_var[j]]) for j in range(len(percs))])


plt.figure(figsize=(8, 8))
plt.plot(percs, gbm_var, 'b', lw=1.5, label='VaR GBM')
plt.plot(percs, jd_var, 'r', lw=1.5, label='VaR JD')
plt.plot(percs, es_grw, 'y', lw=1.5, label='ES GBM')
plt.plot(percs, es_jd, 'c', lw=1.5, label='ES JD')
plt.legend(loc=4)
plt.xlabel('100 - confidence level [%]')
plt.ylabel('value-at-risk')
plt.ylim(ymax=0.0)
plt.savefig('Fig/c-3-VaRES.png')
# ES with JD +- sigma

plt.figure(figsize=(8, 8))
plt.plot(percs, es_jd, 'b', lw=1.5, label='ES JD')
plt.plot(percs, es_jd+es_jd_std, 'r', lw=1.5, label='+$\sigma$')
plt.plot(percs, es_jd-es_jd_std, 'r', lw=1.5, label='-$\sigma$')
plt.fill_between(percs, es_jd+es_jd_std, es_jd-es_jd_std, alpha=0.2)
plt.legend(loc=4)
plt.xlabel('100 - confidence level [%]')
plt.ylabel('ES $\pm\sigma$ in a BDM+JD model')
plt.ylim(ymax=0.0)

plt.savefig('Fig/c-4-ES+-sigma.png')

plt.show()
