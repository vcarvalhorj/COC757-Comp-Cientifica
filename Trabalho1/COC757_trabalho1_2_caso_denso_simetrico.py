# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 16:23:50 2022

@author: User
"""

#COC-757 Trabalho 1
#2 - Caso denso simétrico
#Sistema linar A x = b

import time as t
inicio = t.time()
#A) - Montagem da matriz A

n = 5                                                                #matriz quadrada n x n

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

#C) Resolvendo o sistema assuminda que a matriz A é simétrica através das funções abaixo
#https://scipy.github.io/devdocs/reference/linalg.html#module-scipy.linalg
#https://scipy.github.io/devdocs/reference/generated/scipy.linalg.solve.html#scipy.linalg.solve

from scipy import linalg

x = linalg.solve (A,b,sym_pos=False, lower=False, overwrite_a=False, overwrite_b=False, check_finite=True, assume_a='sym', transposed=False)

print('Resultado utilizando algoritmo solve do Scipy (matriz simetrica densa):',x)
fim = t.time()
print('Tempo de processamento =', fim-inicio,'seg.')