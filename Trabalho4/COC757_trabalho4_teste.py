# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 16:08:29 2022

@author: User
"""

#Exemplo menor para implementacao dos algoritmos
#Steppest descent
#Conjugate gradient

#https://sophiamyang.medium.com/descent-method-steepest-descent-and-conjugate-gradient-in-python-85aa4c4aac7b
#https://sophiamyang.github.io/DS/optimization/descentmethod/descentmethod2.html

#Steepest Descent
import numpy as np
from numpy import linalg as LA

def is_pos_def(x):
    """check if a matrix is symmetric positive definite"""
    return np.all(np.linalg.eigvals(x) > 0)

def steepest_descent(A, b, x):
    """
    Solve Ax = b
    Parameter x: initial values
    """
    if (is_pos_def(A) == False) | (A != A.T).any():
        raise ValueError('Matrix A needs to be symmetric positive definite (SPD)')
    r = b - A @ x
    k = 0
    while LA.norm(r) > 1e-10 :
        p = r
        q = A @ p
        alpha = (p @ r) / (p @ q)
        x = x + alpha * p
        r = r - alpha * q
        k += 1

    return x, k

# A x = 0 definição dos dados de entrada
A = np.array([[3, 2], [2, 3]])             #Matriz A
b = np.zeros(A.shape[-1])                  #Vertor b


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

print(steepest_descent(A, b, x0))

#Conjugate Gradient
def conjugate_gradient(A, b, x):
    if (is_pos_def(A) == False) | (A != A.T).any():
        raise ValueError('Matrix A needs to be symmetric positive definite (SPD)')
    r = b - A @ x
    k = 0
    while LA.norm(r) > 1e-10 :
        if k == 0:
            p = r
        else: 
            gamma = - (p @ A @ r)/(p @ A @ p)
            p = r + gamma * p
        alpha = (p @ r) / (p @ A @ p)
        x = x + alpha * p
        r = r - alpha * (A @ p)
        k += 1
    return x, k

print(conjugate_gradient(A, b, x0))


