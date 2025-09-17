# -*- coding: utf-8 -*-
"""
Created on Sun May  1 17:10:28 2022

@author: vcarv
"""

#https://www.youtube.com/watch?v=H7qMMudo3e8
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
#import os
#plt.rcParams['figure.figsize'] = [16,6]

A = imread('IMG-20200913-WA0020.jpg')   #leitura do arquivo.jpg na mesma pasta do script

X = np.mean(A,axis=2)                   # (A,-1)converte imagens RGB para escala de cinza
img = plt.imshow(X)  
img.set_cmap('gray')
plt.axis('off')
plt.show

#decomposição SVD
U, S, vt = np.linalg.svd(X,full_matrices=False)
S = np.diag(S)

#Reconstrução da imagem variando o modos r
j = 0
for r in (5, 20, 100):
    Xapprox = U[:,:r] @ S[0:r,:r] @ vt[:r,:]
    plt.figure(j+1)
    j += 1
    img = plt.imshow(Xapprox)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('r = ' + str(r))
    plt.show()

# f_ch01_ex02_2

plt.figure(1)
plt.semilogy(np.diag(S))
plt.title('Singular Values')
plt.show()

plt.figure(2)
plt.plot(np.cumsum(np.diag(S))/np.sum(np.diag(S)))
plt.title('Singular Values: Cumulative Sum')
plt.show()