import sympy as sy
import numpy as np
from pylab import mpl, plt

y = sy.Symbol('y')
x = sy.Symbol('x')
#print(3 + sy.sqrt(x) - 4 ** 2)
f = x ** 2 + 3 + 0.5 * x ** 2 + 3 / 2

equations = [sy.Eq(y + x - 3 , 0), sy.Eq(2*x - y - 12, 0)]
aa = sy.solve(equations)
print(aa)
sy.init_printing(pretty_print=False, use_unicode=False)
#print(sy.pretty(f))
#print(sy.pretty(sy.sqrt(x) + 0.5))
print(sy.solve(x ** 3 + 0.5 * x ** 2 - 1))
print(sy.solve(y ** 2 + x ** 2))

a, b = sy.symbols('a b')
I = sy.Integral(sy.sin(x) + 0.5 * x, (x, a, b))
print(sy.pretty(I))
int_func = sy.integrate(sy.sin(x) + 0.5 * x, x)
print(sy.pretty(int_func))
Fb = int_func.subs(x, 9.5).evalf()
Fa = int_func.subs(x, 0.5).evalf()
print(Fb - Fa)
print(int_func.diff())

# General Equilibrium example
# xa, xb, ya, yb, pxa pxb
ya, xa, yb, xb, py, px = sy.symbols('ya xa yb xb py px ')

Ua = 3 * sy.log(xa) / 4 + sy.log(ya) / 4
Ub = 2 * sy.log(xb) / 3 + sy.log(yb) / 3
ea = [2, 1]
eb = [1, 2]

equationsGE = [sy.Eq(xa + xb, 3),
                sy.Eq(ya + yb, 3),
                sy.Eq(px*xa + py*ya, px*2 + py),
                #sy.Eq(px*xb + py*yb, px + py*2),
                sy.Eq(py,1),
                sy.Eq(sy.diff(Ua, xa) / sy.diff(Ua, ya), px),
                sy.Eq(sy.diff(Ub, xb) / sy.diff(Ub, yb), px)
               ]
aa = sy.solve(equationsGE)
print('\n Soluzioni delle equazioni - esmpio 1 : \n', aa)

# Secondo esempio
# tempo 1 e 2 come due beni differenti x1 e x2
x1a, x2a, x1b, x2b, px1, px2 = sy.symbols('x1a x2a x1b x2b px1 px2')
Uaa = 2 * sy.log(x1a) / 3 + sy.log(x2a) / 3
Ubb = sy.log(x1b) / 2 + sy.log(x2b) / 2
#Possedimenti
ea = [1, 1]
eb = [1, 0]
treea = [1, 1/2]
treeb = [0, 1/2]
#dividendi dei 2 stock
dalfa = 1
dbeta = 2
# per Fisher i possedimenti vanno aumentati dai dividendi al tempo 2
eFishera = [ea[0], ea[1] + treea[0]*dalfa + treea[1]*dbeta]
eFisherb = [eb[0], eb[1] + treeb[0]*dalfa + treeb[1]*dbeta]
#print(eFishera, eFisherb)
#Financial equilibrium
pialfa, pibeta, erre = sy.symbols('pialfa pibeta erre')

equationsEqFeq = [sy.Eq(x1a + x1b, eFishera[0]+eFisherb[0]),
                sy.Eq(x2a + x2b, eFishera[1]+eFisherb[1]),
                sy.Eq(px1, 1),
                #sy.Eq(px1*x1b + px2*x2b, px1*eFisherb[0] + px2*eFisherb[1]),
                sy.Eq(px1*x1a + px2*x2a, px1*eFishera[0] + px2*eFishera[1]),
                sy.Eq(sy.diff(Uaa, x1a) / sy.diff(Uaa, x2a), px1/px2),
                sy.Eq(sy.diff(Ubb, x1b) / sy.diff(Ubb, x2b), px1/px2),
                sy.Eq(1+erre , px1/px2),
                sy.Eq(pialfa, dalfa / (1+erre)),
                sy.Eq(pibeta, dbeta / (1+erre))
               ]

esempio2 = sy.solve(equationsEqFeq)
print('\nSoluzioni delle equazioni - esmpio 2 : \n', esempio2)

