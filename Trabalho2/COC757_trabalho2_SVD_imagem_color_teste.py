# -*- coding: utf-8 -*-
"""
Created on Sun May  1 10:34:00 2022

@author: User
"""

#https://scikit-image.org/
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.image import imread


def compress_svd (img,k):
    #decomposição SVD
    u, s, vh = np.linalg.svd(img, full_matrices=False)
    f_red = np.dot(u[:,:k],np.dot(np.diag(s[:k]),vh[:k,:]))
    #S = np.diag(s)
    #f_red = u[:,:k] @ S[:k,:k] @ vh[:k,:]
    return f_red, s



def imagem_reduzida (img,k):
    #img = color_image[img_nome]
    img_layers = [compress_svd(img[:,:,i], k)[0] for i in range (3)]
    img_reconst = np.zeros(img.shape)
    for i in range (3):
        img_reconst[:,:,i] = img_layers[i]/255    
    
    plt.imshow(img_reconst)
    plt.title('r = ' + str(k))
    plt.show()    


foto = io.imread("IMG-20200913-WA0020.jpg")

resp = imagem_reduzida(foto,5)



        
        
        







