import math
import scipy.optimize as sco


def Eu(p):
    s, b = p
    return -(0.5 * math.sqrt(s * 15 + b * 5) + 0.5 * math.sqrt(s * 5 + b * 12))


cons = ({'type': 'ineq', 'fun': lambda p: 100 - p[0] * 10 - p[1] * 10})
bnds = ((0, 1000), (0, 1000))
result = sco.minimize(Eu, [5, 5], method='SLSQP', bounds=bnds, constraints=cons)
print(result)

