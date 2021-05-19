import sympy as sy
import numpy as np
from pylab import mpl, plt
import numba as nb
import scipy.optimize as sco
import numexpr as ne


ne.set_num_threads(4)

# Calcoliamoci i valori 0 dal sistema
# L'esempio che fa lui
x1, x2, x3, x4, x5 = sy.symbols('x1 x2 x3 x4 x5')
P = [100.1, 100.2, 100.3, 100.4, 100.5] # PV oggi dei diversi valori
y = [1, 2, 3, 4, 5] # interessi in %
eqTesoro = [sy.Eq((100+y[0])*x1, P[0]),
            sy.Eq(y[1]*x1 + (100+y[1])*x2, P[1]),
            sy.Eq(y[2]*x1 + y[2]*x2 + (100+y[2])*x3, P[2]),
            sy.Eq(y[3]*x1 + y[3]*x2 + +y[3]*x3 + (100+y[3])*x4, P[3]),
            sy.Eq(y[4]*x1 + y[4]*x2 + +y[4]*x3 + +y[4]*x4 + (100+y[4])*x5, P[4])
            ]
aa = sy.solve(eqTesoro) # trova quanto mi costa un dollaro al tempo t scambiando treseuries
print(aa)

#forward nominal interest rates (for Hull nominal is assumed)

print('i0f: ', 1/aa[x1]-1)
print('i1f: ', aa[x1]/aa[x2]-1)
print('i2f: ', aa[x2]/aa[x3]-1)
print('i3f: ', aa[x3]/aa[x4]-1)
print('i4f: ', aa[x4]/aa[x5]-1)

# stima annuity
def PVAnnuity(coupon, interest, years):
    return ne.evaluate('coupon*(1-(1+interest)**(-years))/interest')

print(PVAnnuity(7, 0.07, 10))

#stima PV0

def PVBond(coupon, interest, years, face):
    i = np.arange(1, years)
    f = 'coupon*(1+interest)**(-i)'
    PV = sum(ne.evaluate(f))
    PV += (face+coupon)*(1+interest)**(-years)
    return PV

pvbond = PVBond(12, 0.01, 30, 0)
print('Valore del bond %6.5f' % pvbond)
print(4/pvbond) # current yield

# stima PV(t) per backward induction

def PVBondMortgageBI(coupon, interest, years, face = 0.):
    PV = [0. for i in range(years)]
    PV[years-1] = (face + coupon[years-1])/(1+interest[years-1])
    print('tempo: ', years -1, ' valore: ', PV[years-1])
    for i in range(years-2, -1, -1):
        PV[i] = (coupon[i+1]+PV[i+1])/(1+interest[i])
        print('tempo: ', i,' valore: ', PV[i])
    print(PV)
    return PV

ys = 30
cp1 = 8
int1 = 0.07
coupon = [cp1 for i in range(ys)]
interest = [int1 for i in range(ys)]
PVBondMortgageBI(coupon, interest, ys)


# stima del tasso d'interesse

ys = 30
coupon = [6 for i in range(0, ys+1)]
coupon[0] = 102
face = 0 #principal
"""
def yBondMortgageBI(yb):
    somma = 0
    for i in range(1, ys+1):
        somma += coupon[i]*((1+yb)**(-i))
    return abs(-coupon[0] + face*(1+yb)**(-ys) + somma)
#Use Comprheansion
"""

def yield_bond_backward_induction(yb):
    return abs(-coupon[0] + face*(1+yb)**(-ys) + sum(coupon[i]*((1+yb)**(-i)) for i in range(1, ys+1)))


opt = sco.brute(yield_bond_backward_induction, ((0, 1.01, 0.001),), finish=None)
print(opt, 'interest brute')
opt2 = sco.fmin(yield_bond_backward_induction, opt, xtol=0.001, ftol=0.001, maxiter=15, maxfun=20)
print(opt2, 'interest preciso')

# prova il coupon
ys2 = 30
yb2 = [0.07 for i in range(0, ys+1)]
PV = 102
face2 = 1000

# stima per il coupon mortgage
ys2 = 30
yb2 = [0.07 for i in range(0, ys+1)]
PV = 100
face2 = 0



def cBondMortgageBI(cc):
    somma = 0.
    for i in np.arange(1, ys2+1):
        somma += cc*((1+yb2[i])**(-i))
    return np.abs(-PV + face2*(1+yb2[i])**(-ys2) + somma)


opt3 = sco.brute(cBondMortgageBI, ((0, 100, 1.),), finish=None)
print(opt3, 'coupon brute')
opt4 = sco.fmin(cBondMortgageBI, opt3, xtol=0.001, ftol=0.001, maxiter=15, maxfun=20)
print(opt4, 'coupon preciso')











