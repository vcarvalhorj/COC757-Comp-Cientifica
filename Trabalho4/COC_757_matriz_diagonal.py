# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 12:55:28 2022

@author: User
"""

#Diagonal solve a partir do TDMA

#a)Caso tridiagonal já conhecido 
import numpy as np


def TDMAsolver(a, b, c, d):
    '''
    DMA solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    '''
    nf = len(d) # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d)) # copy arrays
    for it in range(1, nf):
        mc = ac[it-1]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1]
        dc[it] = dc[it] - mc*dc[it-1]
   
    xc = bc
    xc[-1] = dc[-1]/bc[-1]

    for il in range(nf-2, -1, -1):
        xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]

    return xc

n = 5

a = [1. for i in range(n)]             #subdiagonal inferior da matriz A (A x = b)
b = [2. for i in range(n)]             #diagonal principal da matriz A (A x = b)
c = [1. for i in range(n)]             #subdiagonal superior da matriz A (A x = b)
d = [1. for i in range(n)]             #Vetor dos coeficientes independentes b (A x = b)

x = TDMAsolver(a,b,c,d)

#b) Adaptação para o caso diagonal
#Verificação com vetores com subdiagonal 'a' e 'c' zerados 
def DMAsolver(b,d):
    '''
    TDMA solver, b d can be NumPy array type or Python list type.
    Adapted from refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    '''
    nf = len(d) # number of equations
    bc, dc = map(np.array, (b, d)) # copy arrays
    xc = np.zeros(nf)
    for it in range(nf):
        xc[it] = dc[it]/bc[it]

    return xc

b1 = [2. for i in range(n)]             #diagonal principal da matriz A (A x = b)
d1 = [1. for i in range(n)]             #Vetor dos coeficientes independentes b (A x = b)

x_d = DMAsolver(b1,d1)


#Verificação com vetores com subdiagonal 'a' e 'c' zerados 
a2 = [0. for i in range(n)]             #subdiagonal inferior da matriz A (A x = b)
b2 = [2. for i in range(n)]             #diagonal principal da matriz A (A x = b)
c2 = [0. for i in range(n)]             #subdiagonal superior da matriz A (A x = b)
d2 = [1. for i in range(n)]             #Vetor dos coeficientes independentes b (A x = b)

x_t = TDMAsolver(a2,b2,c2,d2)
