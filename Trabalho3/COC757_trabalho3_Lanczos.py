# -*- coding: utf-8 -*-
"""
Created on Fri May 20 09:22:53 2022

@author: User
"""

#COC 757 - Trabalho 3 - Lanczos

#1) Leitura e armazenamento da matriz
#https://scipy.github.io/devdocs/reference/io.html
#https://scipy.github.io/devdocs/reference/generated/scipy.io.mmread.html#scipy.io.mmread   #importar Market file matrix ((extensions .mtx, .mtz.gz))
import scipy.io 
import matplotlib.pyplot as plt
import numpy as np

#https://sparse.tamu.edu/Bai/bfwb62
matriz = scipy.io.mmread('bfwb62.mtx')
#https://scipy.github.io/devdocs/reference/generated/scipy.io.mminfo.html
print(scipy.io.mminfo('bfwb62.mtx'))                                                          #informações da matriz

mat = matriz.todense()                                                                        #converte a matriz importada de 'coordinate' para o tipo 'array' com dimensões n x n

#Plota o padrão esparso da matriz convertida para tipo array
#https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.spy.html
plt.spy(mat)
plt.title('bfwb62.mtx: ' + str(scipy.io.mminfo('bfwb62.mtx')))
plt.show()

#2)Cálculo dos autovalores e autovetores
#https://scipy.github.io/devdocs/reference/generated/scipy.sparse.linalg.eigsh.html#scipy.sparse.linalg.eigsh
from scipy.sparse.linalg import eigsh
val5, vet5 = eigsh(mat, k=5)
val10, vet10 = eigsh(mat, k=10)
val62, vet62 = eigsh(mat, k=61)

plt.plot(abs(val5))                                                #verificação dos primeiros  autovalores
plt.plot(abs(val10))
plt.show()

#3) Comparação dos autovalores com os valores singulares da matriz
def compress_svd (matrix,k):                                   #definição de função específica decomposição SVD
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    S = np.diag(s)
    f_red = u[:,:k] @ S[:k,:k] @ vh[:k,:]
    return f_red, s


sig = compress_svd(mat, 10)[1]                                 #Valores singulares da matriz esparsa

plt.plot(abs(val62), label = 'Módulo dos autovalores')
plt.plot(sig, label = 'Valores singulares')
plt.title('Autovalores x valores singulares')
plt.legend()
plt.show()





