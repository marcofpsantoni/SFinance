import sympy as sy
import numpy as np
from pylab import mpl, plt
import numba as nb
import scipy.optimize as sco
import numexpr as ne


"""
Contious compounding 
1) passare da continuos a 1/m annual compounding and vice-versa 
2) yield
3) coupon
4) PV BI with changing yield
5) Duration and Convexity
6) Acc
"""

def cont_comp(rm, m):
    return m*np.log(1+rm/m)
#print(cont_comp(0.14,4))


def period_comp(rc,m):
    return m*(np.exp(rc/m)-1)
#print(period_comp(0.1376057, 4))

# stima del tasso d'interesse

anni = 30 #anni
period = 2 # 1 annual, 2 semiannual..
cedola = [6 for i in range(0, anni*period+1)] # coupon fisso ogni periodo
cedola[0] = 102 # Valore oggi (quanto costa = cedola negativa)
faccia = 100 # principal quello che ti danno all'ultimo pagamento
#When PV=Face par-value
cedola[anni*period] += faccia #l'ultimo pagamento face + coupon


def yield_bond_cont_comp(yb):
    return abs(-cedola[0] + sum(cedola[i]*np.exp(-yb*i/period) for i in range(1, anni*period+1)))

opt_comp = sco.brute(yield_bond_cont_comp, ((0, 1.01, 0.001),), finish=None)
#print(opt_comp, 'interest brute')
opt2_comp = sco.fmin(yield_bond_cont_comp, opt_comp, xtol=0.001, ftol=0.001, maxiter=15, maxfun=20)
print(opt2_comp, '= interest cont compounding')
print("Che corrisponde ha un seminanuale", period_comp(opt2_comp,2))


#Coupon
years = 30
periodo = 4 #3-months
facciamo = 0.07
y_bond = [facciamo for i in range(years*periodo+1)]
#print(len(y_bond))
pv = 130 # if PV = 100 par yield coupon
face = 100

def coupon_bond(cc):
    return abs(- pv + face*np.exp(-facciamo*years) + sum(cc*np.exp(-y_bond[i]*i/periodo) for i in range(1, len(y_bond))))

opt3 = sco.brute(coupon_bond, ((0, 100, 1.),), finish=None)
print(opt3, 'coupon brute')
opt4 = sco.fmin(coupon_bond, opt3, xtol=0.001, ftol=0.001, maxiter=15, maxfun=20)
print(opt4, 'coupon preciso')

# present value of an annual bond
zero_yield_prova = [2, 3, 4, 5.3, 6, 6.1, 6.2, 5, 4, 2]  # in percentage
face0y = 100

def pv_bond(r, face):
    y = len(r)
    return face*np.exp(-y*r[y-1]) + sum(r[i]*np.exp(-r[i]*i/100) for i in range(len(r)))


print(pv_bond(zero_yield_prova, face0y))
'''
Duration & Convexity
The duration gives you the time you repay the bond
Duration and Convexity can be used to approximate (first and second diff) a yield-variation
'''
def duration_convexity(co, yi, fa, ti, fr):
    b = [co*np.exp(-yi*i/fr) for i in range(1, fr*ti+1)]
    b[fr*ti-1] += fa*np.exp(-yi*ti)
    b = np.array(b)
    pv = sum(b)
    t = np.array([i/fr for i in range(1, fr*ti+1)])
    d = b*t/pv
    c = b*t*t/pv
    return pv, sum(d), sum(c)
# 3(30)- years semi-annual 10% coupon bond with y = 0.12 - > Get Duration
print(duration_convexity(5, 0.12, 100, 3, 2))
print(duration_convexity(5, 0.12, 100, 30, 2))
#Approximation
B = duration_convexity(5, 0.12, 100, 3, 2)[0]
DB = -duration_convexity(5, 0.12, 100, 3, 2)[0]+duration_convexity(5, 0.125, 100, 3, 2)[0]
Dy = 0.005
D = duration_convexity(5, 0.12, 100, 3, 2)[1]
C = duration_convexity(5, 0.12, 100, 3, 2)[2]
print("Approximations: ", DB/B, -D*Dy, -D*Dy+0.5*C*Dy*Dy)


#Accumolation

an = 20  # anni
inv = [10000 for i in range(an)]  # investo 10000 l'anno
real = 0.05  # tasso reale 5%
ret = np.zeros(an)

def accumulo(a, inv, r):
    ret[0] = 50000 #inv[0]
    for i in range(1, a):
        ret[i] = inv[i] + ret[i-1]*np.exp(r)
    return ret

print(accumulo(an,inv,real))
