# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 16:04:32 2022

@author: User
"""

#COC 757 - Trabalho 4 - Otimizacao

import time as t
inicio = t.time()
#1) Leitura e armazenamento da matriz
#https://scipy.github.io/devdocs/reference/io.html
#https://scipy.github.io/devdocs/reference/generated/scipy.io.mmread.html#scipy.io.mmread   #importar Market file matrix ((extensions .mtx, .mtz.gz))
import scipy.io 
import matplotlib.pyplot as plt
import numpy as np

#https://sparse.tamu.edu/HB/bcsstk07
matriz = scipy.io.mmread('bcsstk07.mtx')
#https://scipy.github.io/devdocs/reference/generated/scipy.io.mminfo.html
print(scipy.io.mminfo('bcsstk07.mtx'))                                                          #informações da matriz

mat = matriz.todense()                                                                        #converte a matriz importada de 'coordinate' para o tipo 'array' com dimensões n x n

#Plota o padrão esparso da matriz convertida para tipo array
#https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.spy.html
plt.figure('Figura 1')
plt.spy(mat)
plt.title('bcsstk07.mtx: ' + str(scipy.io.mminfo('bcsstk07.mtx')))
plt.show()

#2) Verificação do condicionamento da matriz
#https://www.ufrgs.br/reamat/CalculoNumerico/livro-py/sdsl-condicionamento_de_sistemas_lineares.html
#https://numpy.org/doc/stable/reference/generated/numpy.linalg.cond.html
print(np.linalg.cond(mat,'fro'))
print(np.linalg.cond(mat,np.inf))
print(np.linalg.cond(mat,-np.inf))
print(np.linalg.cond(mat,1))
print(np.linalg.cond(mat,-1))
print(np.linalg.cond(mat,2))
print(np.linalg.cond(mat,-2))

#3) Conjugate gradient
from numpy import linalg as LA

def is_pos_def(x):
    """check if a matrix is symmetric positive definite"""
    return np.all(np.linalg.eigvals(x) > 0)

def conjugate_gradient(A, b, x):
    if (is_pos_def(A) == False) | (A != A.T).any():
        raise ValueError('Matrix A needs to be symmetric positive definite (SPD)')
    r = b - A @ x
    k = 0
    while LA.norm(r) > 1e-10:
        if k == 0:
            p = r
        else: 
            gamma = - (p @ A @ r)/(p @ A @ p)
            p = r + gamma * p
        alpha = (p @ r) / (p @ A @ p)
        x = x + alpha * p
        r = r - alpha * (A @ p)
        print('ite =', k, 'norma=', LA.norm(r))
        k += 1
    return x, k

A = np.array(mat)                              #Matriz A
b_vet = np.zeros(A.shape[-1])                  #Vertor b

#Vetor x0 inicial aleatório
#https://machinelearningmastery.com/how-to-generate-random-numbers-in-python/
# generate random floating point values
from random import seed, random
# seed random number generator
seed(1)
# generate random numbers between 0-1 x 10
x0_list= []
for i in range(len(b_vet)):
    value = random()
    x0_list += [value*2]

x0 = np.array(x0_list)                                #Vetor inicial x0

x = conjugate_gradient(A, b_vet, x0)                 #solucao numérica aproximada
xc=np.zeros(len(x[0]))                               #solução exata
fim = t.time()

#4)Resultados
print(x)
print('Máximo = ',x[0].max())
print('Mínimo = ',x[0].min())
print('Média = ',x[0].mean())
print('Desvio padrão=', x[0].std())
print('Tempo de processamento =', fim-inicio,'seg.')

#Gráfico dos pontos calculados
plt.figure('Figura 2')
plt.scatter(xc,x[0],marker = '*', color = 'g')
plt.title('Conjugate Gradient')
plt.xlabel('Solução exata')
plt.ylabel('Solução aproximada')
plt.grid(b=True, linestyle='--', which= 'major', color = 'gray')
plt.show()
