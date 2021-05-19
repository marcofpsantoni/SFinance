import math
import numpy as np
import numba
import time
from pylab import mpl, plt

np.set_printoptions(formatter={'float': lambda x: '%8.3f' % x})
"""
Primo esempio di BI sono le world season: un fan scommette a 2 su la squadra B che gioca
contro A alla meglio di n partite. La squadra A e' quotata con probabilita' maggiore di 1/2
per ogni singolo incontro contro B. Il primo valore corrisponde ai soldi che guadagnera' di sicuro.
"""


def simulate_BI_tree(M, u, d, p):
    """
    :param M: il numero di partite
    :param u: la vincita
    :param d: la perdita
    :param p: la prob nella singola partita
    :return: la matrice dei valori per ogni nodo nella serie
    """
    BI = np.zeros((M + 1, M + 1))
    for i in range(0, 4):
        BI[i, M] = u
        BI[i + 4, M] = d
    for c in range(M - 1, -1, -1):
        for r in range(0, M):
            if r <= c:
                BI[r, c] = p * BI[r, c + 1] + (1 - p) * BI[r + 1, c + 1]
            else:
                BI[r, c] = 0

    return BI


matrice = simulate_BI_tree(7, 100, -100, 0.6)
print(33 * "-", " Quanto vale la posizione", 33 * "-")
print(matrice)


"""
Gioco Rosso e Nero: vinci un dollaro ogni volta che esce il nero e puoi fermarti quando vuoi. 
L'opzione di fermarsi vale tanto. Qual e' la miglior strategia? 
Puoi valutare il valore delle posizioni in base alle carte rimaste nel mazzo
"""

def valore_rossoenero(v, p):
    """
    :param v: carte con cui vinci (e.g. carte nere rimaste nel mazzo)
    :param p: carte con cui perdi (e.g. carte rosse rimaste nel mazzo)
    :return: matrice dei valori del proseguimento del gioco:
    La miglior strategia consiste nel continuare se v>0
    """
    m = np.zeros((p+1, v+1))
    for i in range (v+1):
        m[i][0] = i
    for i in range(p+1):
        m[0][i] = 0
    for b in range(1, v + 1):
        for r in range(1, p + 1):
            m[b][r] = np.maximum(0., (b/(b+r))*(1+m[b-1][r]) + (r/(b+r))*(-1+m[b][r-1]))

    return m

print("\n\n", valore_rossoenero(26,26))

"""
Matrimonio 1000 donne uniformemente distribuite tra 0 e 1. Indovina la soglia: 999/1001.
"""

def mogliesoglia(n):
    """
    :param n: Numero degli appuntamenti che avere
    :return: Matrice dei valori della soglia di affinita' sopra la quale sposarsi
    """
    v = np.zeros(n)
    v[0] = 0
    for i in range(1, n):
        v[i] = 0.5*(1+v[i-1]**2)

    return v

a = mogliesoglia(100)

#Vediamo i valori di soglia in base al numero di appuntamenti rimasti
plt.figure(figsize=(10, 6))
plt.plot(a)
plt.xlabel('appuntamenti rimanenti')
plt.ylabel('soglia')

plt.show()