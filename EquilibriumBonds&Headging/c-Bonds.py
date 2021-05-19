import sympy as sy
import numpy as np
from pylab import mpl, plt
import numba as nb
import scipy.optimize as sco
import numexpr as ne


np.set_printoptions(formatter={'float': lambda x: '%8.3f' % x})


# stima per il coupon mortgage
ys2 = 30 # fixed rate mortgage
int_rate_today = 0.06 # tasso di interesse oggi
inte = 0.08 # tasso di interesse del mortgage
yb2 = [inte for i in range(0, ys2+1)]
PV = 100.
face2 = 0.

"""
Simuliamo il mortgage non collable
"""

def cBondMortgageBI(cc):
    """
    :param cc: Coupon to be found with scipy
    :return: value of the mortgage
    """
    somma = 0.
    for i in np.arange(1, ys2+1):
        somma += cc*((1+yb2[i])**(-i))
    return np.abs(-PV + face2*(1+yb2[i])**(-ys2) + somma)


opt3 = sco.brute(cBondMortgageBI, ((0, 100, 1.),), finish=None)
print(opt3, 'coupon brute')
opt4 = float(sco.fmin(cBondMortgageBI, opt3, xtol=0.001, ftol=0.001, maxiter=15, maxfun=20))
print(opt4, 'coupon preciso')

"""
Calcoliamo il PV del mortgage ammortizzato con coupon opt4 e tasso di interesse al 6%
"""

def PVBondMortgageBI(coupon, interest, years):
    pv = [0. for e in range(years+1)]
    pv[years] = coupon
    print('tempo: ', years, ' valore: ', pv[years])
    for i in range(years-1, 0, -1): # il mutuo inizi a pagarlo al tempo 1
        pv[i] = (coupon + pv[i+1])/(1+interest)
        print('tempo: ', i, ' valore: ', pv[i])
    return pv

pv = PVBondMortgageBI(opt4, 0.06, ys2)
print(pv[0], pv[1])


"""
Calcoliamo il remaining balance per induzione
"""

def rem_balance(y, presval, inter, c):
    a = []
    a.append(presval)
    for i in range(y-1):
        a.append(a[i]*(1+inter)-c)
    a.append(0.)
    return a

ar = rem_balance(ys2, PV, inte, opt4)
print(ar)



"""
Simuliamo il tasso di interesse (nominale) variabile e lo sconto nominale 
"""
sigma = 0.2
drift = 0.
k = np.exp(sigma + drift)
u = k
d = 1/k
print(u, "   ", d )
def simulate_erre_sconto(r0=1, n=10, up = 2, do = 0.5):
    S = np.zeros((n + 1, n + 1))
    D = np.zeros((n + 1, n + 1))
    S[0, 0] = r0
    D[0, 0] = 1/(1+r0)
    z = 1
    for t in range(1, n+1):
        for i in range(z):
            S[i, t] = S[i, t-1] * up
            D[i, t] = 1/(1+S[i, t])
            S[i+1, t] = S[i, t-1] * do
            D[i + 1, t] = 1/(1+S[i+1, t])
        z += 1

    return S, D

simulate_erre_sconto_nb = nb.jit(simulate_erre_sconto)
interesse, sconto = simulate_erre_sconto_nb(int_rate_today, ys2, u, d)
print("\nerre\n", interesse)
print("\nsconto\n", sconto)


"""
Calcoliamo il valore del callable mortgage
"""

def simulate_BI_callable_mortgage(y, d, rb, p, co):
    """
    :param y: glia nni
    :param d: la matrice di sconto
    :param rb: array del remaining balance
    :param p: probability up
    :param co: agreed coupon
    :return: matrice del collable mortgage
    """
    BI = np.zeros((y + 1, y + 1))
    for i in range(0, y):
        BI[i, y] = 0
    for c in range(y - 1, -1, -1):
        for r in range(0, y):
            if r <= c:
                BI[r, c] = np.minimum(rb[c], d[r, c] *
                           (p * (co + BI[r, c + 1]) +
                           (1 - p) * (co + BI[r + 1, c + 1])))
            else:
                BI[r, c] = 0

    return BI


matrice = simulate_BI_callable_mortgage(ys2, sconto, ar, 0.5, opt4)
print(33 * "-", " Quanto vale il collable mortgage", 33 * "-")
print(matrice)