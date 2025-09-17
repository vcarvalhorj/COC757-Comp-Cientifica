# -*- coding: utf-8 -*-
"""
Created on Sun May  1 10:34:00 2022

@author: User
"""

#https://scikit-image.org/
from skimage import data, io, filters

image = data.coins()
# ... or any other NumPy array!
edges = filters.sobel(image)
io.imshow(edges)
io.show()
io.imshow(image)
io.show()

#https://scikit-image.org/docs/stable/user_guide/numpy_images.html#color-images
cat = data.chelsea()
io.imshow(cat)
io.show()

type(cat)
cat.shape


#lendo imagem de arquivo
lampada = io.imread("lampada.png")
io.imshow(lampada)
type(lampada)
lampada.shape


#lendo imagem de arquivo
foto = io.imread("IMG-20200913-WA0020.jpg")
io.imshow(foto)
type(foto)
foto.shape

#dados dos pixels
cat[10, 20]
foto[959,1279]
foto[0,0]
lampada[0,0]


#%%Fórmulas SVD
import numpy as np


a = np.random.randn(9, 6) + 1j*np.random.randn(9, 6)
b = np.random.randn(2, 7, 8, 3) + 1j*np.random.randn(2, 7, 8, 3)

u, s, vh = np.linalg.svd(a, full_matrices=True)

#Exemplo capitulo 3 COC757

matA = [[i+2.*j+1. for i in range(3)] for j in range(4)]
m= np.array(matA)

uu, ss, vvh = np.linalg.svd(matA, full_matrices=True)
uu.shape, ss.shape, vvh.shape
print(ss)
