import numpy as np
import numpy.random as npr
import numba as nb
import math
from pylab import mpl, plt
import scipy.stats as scs
import math

S0 = 100.
r = 0.05
sigma = 0.2
T = 1.
I = 100000  # Number of MC simulations

ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * npr.standard_normal(I))
L = 0.5
p = 0.01
D = npr.poisson(p * T, I)
D = np.where(D > 1, 1, D)

print("Discounted asset value: ", math.exp(-r * T) * np.mean(ST))
CVaR =  math.exp(-r * T) * np.mean(L * D * ST) # Discounted value *L *D
# Discounted average simulated value of the asset at T,
# adjusted for the simulated losses from default
S0_CVA = math.exp(-r * T) * np.mean((1 - L * D) * ST)
S0_adj = S0 - CVaR #Current price of the asset adjusted by the simulated CVaR

print(CVaR , S0_CVA, S0_adj)
plt.figure(figsize=(10, 6))
plt.hist(L * D * ST, bins=50)
plt.xlabel('loss')
plt.ylabel('frequency')
plt.ylim(ymax=175)


print(np.count_nonzero(L * D * ST))

#We can consider now the case of a European call option.

K = 100.
hT = np.maximum(ST - K, 0)
C0 = math.exp(-r * T) * np.mean(hT)
print("EO value= ", C0)
CVaR = math.exp(-r * T) * np.mean(L * D * hT)
print("CVaR EO= ", CVaR)
C0_CVA = math.exp(-r * T) * np.mean((1 - L * D) * hT)
print("C0_CVA EO=", C0_CVA)

print(np.count_nonzero(L * D * hT))
#~1/2 since you can have 0
print(np.count_nonzero(D))
print(I - np.count_nonzero(hT))
plt.figure(figsize=(10, 6))
plt.hist(L * D * hT, bins=50)
plt.xlabel('loss')
plt.ylabel('frequency')
plt.ylim(ymax=350)

plt.show()