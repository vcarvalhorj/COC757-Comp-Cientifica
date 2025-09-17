# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 16:08:29 2022

@author: User
"""

#Exemplo menor para implementacao dos algoritmos
#Steppest descent com consicionamento
#Conjugate gradient com condicionamento

#https://sophiamyang.medium.com/descent-method-steepest-descent-and-conjugate-gradient-in-python-85aa4c4aac7b
#https://sophiamyang.github.io/DS/optimization/descentmethod/descentmethod2.html


import numpy as np
from numpy import linalg as LA
#solver A x = b para caso especifico de A ser diagonal
def DMAsolver(b,d):
    '''
    DMA solver, b d can be NumPy array type or Python list type.
    Adapted from refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    '''
    nf = len(d) # number of equations
    bc, dc = map(np.array, (b, d)) # copy arrays
    xc = np.zeros(nf)
    for it in range(nf):
        xc[it] = dc[it]/bc[it]

    return xc

def is_pos_def(x):
    """check if a matrix is symmetric positive definite"""
    return np.all(np.linalg.eigvals(x) > 0)
#Steepest Descent
def steepest_descent(A, m, b, x):
    """
    Solve Ax = b
    Parameter x: initial values
    """
    if (is_pos_def(A) == False) | (A != A.T).any():
        raise ValueError('Matrix A needs to be symmetric positive definite (SPD)')
    r = b - A @ x
    k = 0
    while LA.norm(r) > 1e-10 :
        p = DMAsolver(m, r)                            #precondicionamento
        q = A @ p
        alpha = (p @ r) / (p @ q)
        x = x + alpha * p
        r = r - alpha * q
        k += 1

    return x, k

# A x = 0 definição dos dados de entrada
A = np.array([[3, 2], [2, 3]])             #Matriz A
b = np.zeros(A.shape[-1])                  #Vertor b
M = np.diag(A)

#Vetor x0 inicial aleatório
#https://machinelearningmastery.com/how-to-generate-random-numbers-in-python/
# generate random floating point values
from random import seed, random
# seed random number generator
seed(5)
# generate random numbers between 0-1 x 10
x0_list= []
for i in range(2):
    value = random()
    x0_list += [value*10]

x0 = np.array(x0_list)              #Vetor inicial x0

print(steepest_descent(A, M, b, x0))

#Conjugate Gradient
def conjugate_gradient(A, m, b, x):
    if (is_pos_def(A) == False) | (A != A.T).any():
        raise ValueError('Matrix A needs to be symmetric positive definite (SPD)')
    r = b - A @ x
    k = 0
    #d = DMAsolver(m, r)
    while LA.norm(r) > 1e-10 :
        if k == 0:
            p = DMAsolver(m, r)                          #precondicionamento
        else: 
            gamma = - (p @ A @ r)/(p @ A @ p)
            p = DMAsolver(m, r) + gamma * p              #precondicionamento
        alpha = (p @ r) / (p @ A @ p)
        x = x + alpha * p
        r = r - alpha * (A @ p)
        k += 1
    return x, k

print(conjugate_gradient(A, M, b, x0))

