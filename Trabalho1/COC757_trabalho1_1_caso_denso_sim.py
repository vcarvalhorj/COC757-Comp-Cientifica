# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 16:23:50 2022

@author: User
"""

#COC-757 Trabalho 1
#1 - Caso denso
#Sistema linar A x = b

import time as t
inicio = t.time()
#A) - Montagem da matriz A

n = 10000                                                                #matriz quadrada n x n

#matriz zerada
A = []
for i in range (n):
    A += [[]]
for i in range (n):
    A [i] += [0. for j in range(n)]                                  #matriz = [[0 for i in range(n)] for j in range (n)].Montagem de matriz - lista de lista

#matriz tri - diagonal
for i in range (n):
    for j in range (n):
        if i==j:                                                    #parcela diagonal
            A [i][j] = 2.
        elif j-i == 1:                                              #diagonal superior
            A [i][j] = 1.
        elif j-i == -1:                                             #diagonal inferior
            A [i][j] = 1.

#B) Termos independentes (vetor b)

b = [1. for i in range(n)]

#C) - Inversão da matriz A (x = A_inv b)
#https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html

from numpy.linalg import inv
A_inv = inv(A)


#D) Calculo do vetor x: resolvendo a sistema

x = []
for i in range (n):
        x += [0.]
        for j in range (n):
            x[i] = x[i] + A_inv[i][j]*b[j]

print('Resultado utilizando algoritmo de inversão:',x)

#Alternativamente é possível resolver o sistema linear pelo "solve" do numpy.linalg sem prévia inversão de matriz
#https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html
#https://scipy.github.io/devdocs/reference/generated/scipy.linalg.solve.html#scipy.linalg.solve
import numpy as np
from numpy.linalg import solve
from scipy import linalg

matA = np.array(A)                                   #converte tipo lista em array Não é necessário para este programa.
vb = np.array(b)                                     #converte tipo lista em array. Não é necessário para este programa.

npx = solve(A,b)
scx = linalg.solve (A,b,sym_pos=False, lower=False, overwrite_a=False, overwrite_b=False, check_finite=True, assume_a='gen', transposed=False)
print('Resultado utilizando solve do Numpy:',npx)
print('Resultado utilizando solve do Scipy (matriz generica densa):',scx)
fim = t.time()
print('Tempo de processamento =', fim-inicio,'seg.')