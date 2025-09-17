# -*- coding: utf-8 -*-
"""
Created on Sun May  1 10:34:00 2022

@author: User
"""

#COC 757 - Trabalho 2 - SVD (imagem em escala de cinza)

#https://scikit-image.org/
#https://www.youtube.com/watch?v=H7qMMudo3e8
from skimage import io
import numpy as np
import matplotlib.pyplot as plt


#lendo imagem de arquivo
foto = io.imread("IMG-20200913-WA0020.jpg")


#convertendo a imagem para escala de cinza
fbw = np.mean(foto,axis=2)


#decomposição SVD
u, s, vh = np.linalg.svd(fbw, full_matrices=False)
S = np.diag(s)
ric = np.cumsum(np.diag(S))/np.sum(np.diag(S))



#Reconstrução da imagem variando o modos/ranks r (ver vídeo no youtube)
j = 0
for r in (5,20,150,959):
    f_red = u[:,:r] @ S[0:r,:r] @ vh[:r,:]
    plt.figure(j+1)                                           #para plotar a imagem no modo/rank r
    j += 1
    img = plt.imshow(f_red)
    img.set_cmap('gray')
    plt.title('r = ' + str(r) + '; RIC = '+ str(ric[r]))
    plt.show()

#Gráficos de valores singulares e RIC
plt.figure(1)
plt.semilogy(np.diag(S))
plt.title('Valores singulares')
plt.xlabel('m')
plt.ylabel('Sigma')
plt.show()

plt.figure(2)
plt.plot(ric)
plt.title('RIC')
plt.xlabel('m')
plt.ylabel('RIC')
plt.show()




