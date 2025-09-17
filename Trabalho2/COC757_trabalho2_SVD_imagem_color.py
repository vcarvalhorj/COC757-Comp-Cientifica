# -*- coding: utf-8 -*-
"""
Created on Sun May  1 10:34:00 2022

@author: User
"""

#COC 757 - Trabalho 2 - SVD (imagem RGB)

#https://scikit-image.org/
#https://www.youtube.com/watch?v=H7qMMudo3e8
from skimage import io
import numpy as np
import matplotlib.pyplot as plt



def compress_svd (img,k):                                   #definição de função específica decomposição SVD
    u, s, vh = np.linalg.svd(img, full_matrices=False)
    S = np.diag(s)
    f_red = u[:,:k] @ S[:k,:k] @ vh[:k,:]
    return f_red, s

#lendo imagem de arquivo
foto = io.imread("IMG-20200913-WA0020.jpg")



#Reconstrução da imagem variando o modos r para cada canal de cor RGB
#https://medium.com/@rameshputalapattu/jupyter-python-image-compression-and-svd-an-interactive-exploration-703c953e44f6
for r in (5,20,150,959):
    img_layers = [compress_svd(foto[:,:,i], r)[0] for i in range (3)]
    s_layers = [compress_svd(foto[:,:,i], r)[1] for i in range (3)]
    ric_layers = [np.cumsum(np.diag(np.diag(s_layers[i])))/np.sum(np.diag(np.diag(s_layers[i])))for i in range(3)]
    img_reconst = np.zeros(foto.shape)
    for i in range (3):
        img_reconst[:,:,i] = img_layers[i]/255    
    
    plt.imshow(img_reconst)                                              #para plotar a imagem no modo/rank r
    plt.title('r = ' + str(r) +'; RIC =' + str((ric_layers[0][r]+ric_layers[1][r]+ric_layers[2][r])/3))
    plt.show()

#imagem não reduzida (original)
plt.imshow(foto)
plt.title('Original')
plt.show()

#Gráficos de valores singulares e RIC
plt.figure(1)
plt.semilogy(np.diag(np.diag(s_layers[0])), label='R' )
plt.semilogy(np.diag(np.diag(s_layers[1])), label= 'G')
plt.semilogy(np.diag(np.diag(s_layers[2])), label= 'B')
plt.title('Valores singulares')
plt.xlabel('m')
plt.ylabel('Sigma')
plt.legend()
plt.show()

plt.figure(2)
plt.plot(ric_layers[0], label= 'R')
plt.plot(ric_layers[1], label= 'G')
plt.plot(ric_layers[2], label= 'B')
plt.title('RIC')
plt.xlabel('m')
plt.ylabel('RIC')
plt.legend()
plt.show()







