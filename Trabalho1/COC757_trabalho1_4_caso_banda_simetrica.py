# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 16:23:50 2022

@author: User
"""

#COC-757 Trabalho 1
#4 - Caso de matriz em banda simetrica
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

#C) Matriz em banda
#https://scipy.github.io/devdocs/reference/linalg.html#module-scipy.linalg
#https://scipy.github.io/devdocs/reference/generated/scipy.linalg.solveh_banded.html

l = 1                                                            #número de diagonais abaixo da diagonal principal diferente de zero
u = 1                                                            #número de diagonais acima da diagonal principal diferente de zero
diagtot = 1 + l + u                                              #quantidade total de diagonais

#matriz ab auxiliar zerada (quadrada)
ab_aux = []
ab_aux = [[0. for i in range(n)] for j in range (n)]
#montagem da matriz ab auxiliar (quadrada)
for i in range (n):
    for j in range (n):
        if i == n-1 and j == 0:
            pass
        else:
            ab_aux[u+i-j][j] = A [i][j]


#matriz ab zerada (banda de interesse superior - upper form)
ab = []
ab = [[0. for i in range(n)] for j in range (diagtot-u)]
#banda de interesse
for i in range (diagtot-u):
    for j in range (n):
        ab [i][j]= ab_aux [i][j]

#D) Solução
from scipy.linalg import solveh_banded
x = solveh_banded(ab, b,lower=False)                             #lower False é o default da fórmula
print('Resultado utilizando algoritmo solve do Scipy (matriz em banda simetrica):',x)
fim = t.time()
print('Tempo de processamento =', fim-inicio,'seg.')