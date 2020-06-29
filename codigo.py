import numpy as np
from scipy import stats
from fitter import Fitter
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

import csv

# Inicializar una lista donde se almacenan los datos
xy = []

# Cargar documento CSV en la lista 'lote'
with open('xy.csv', newline='') as archivo:
	lectura = csv.reader(archivo)
	next(lectura, None) 	# no guardar el encabezado
	for fila in lectura:
		xy.append(fila)	# adjuntar cada fila a lote


'''
1.  A partir de los datos, encontrar la mejor curva de ajuste (modelo probabilístico)
 para las funciones de densidad marginales de X y Y.
'''
print('\n--- 1 ---\n')

#Creación de matriz

matriz = [
    [0.00262, 0.00177,0.00325,0.00353,0.003,0.00365	,0.00544,0.00466,0.00381,0.00445,0.00122,0.0049	,0.0022	,0.00359,0.00304,0.00212,0.00355,0.00247,0.00337,0.00184,0.00266],
    [0.00493,0.00307,0.00216,0.00186,0.00448,0.00356,0.0037	,0.00376,0.00381,0.00318,0.00201,0.00337,0.00301,0.00435,0.00369,0.00266,0.00396,0.00394,0.00495,0.0014,0.00387],
    [0.00283,0.00333,0.00369,0.00387,0.00396,0.00275,0.00569,0.00223,0.00364,0.00362,0.0041,0.00441,0.00644,0.00316,0.00389,0.00328,0.00244,0.00526,0.00488,0.00387,0.00593],
    [0.00288,0.00143,0.00456,0.00205,0.00273,0.00357,0.00175,0.00551,0.00627,0.01,0.00796,0.00881,0.0065,0.00431,0.00486,0.00457,0.00339,0.00276,0.00339,0.00305,0.00195],
    [0.00466,0.00404,0.00313,0.00318,0.00195,0.0043,0.00437	,0.00666,0.0103,0.01257	,0.01372,0.01399,0.00934,0.00784,0.00416,0.00298,0.00292,0.00323,0.0032	,0.00334,0.00238],
    [0.00277,0.00293,0.00373,0.00315,0.00509,0.00259,0.00512,0.00727,0.01273,0.01635,0.0176	,0.015,0.01172,0.00766,0.00644,0.00498,0.00359	,0.00396,0.00411,0.00149,0.00321],
    [0.00425,0.00405,0.00389,0.00315,0.00198,0.00418,0.00526,0.00645,0.00944,0.0125	,0.01554,0.01179,0.00905,0.00792,0.00482,0.00462,0.00296,0.00112,0.00238,0.00271,0.00366],
   [0.00361	,0.00461,0.0,0.00333,0.00158,0.00517,0.00364,0.00267,0.00303,0.00543,0.00341,0.00585,0.0043	,0.00318,0.00488,0.00426,0.00255,0.004,0.00351,0.00388,0.00397],
    [0.00277,0.00257,0.00327,0.00428,0.00498,0.00214,0.00292,0.00218,0.00302,0.00436,0.00276,0.00121,0.0035	,0.00121,0.004,0.00265,0.00234,0.00145,0.00358,0.0019,0.00268],
    [0.00336,0.0029	,0.0012	,0.00108,0.00243,0.00227,0.00562,0.00247,0.00363,0.00437,0.00272,0.00387,0.00385,0.00388,0.00257,0.00406,0.00393,0.00244,0.00333,0.00235,0.00286]
]


Filas = len(matriz)
print(Filas)
Columnas = len(matriz[0])
print(Columnas)

#Sumas de las filas
nueva_fila = []
for i in range(Filas):
    suma_xn = np.sum(matriz[i])
    nueva_fila.append(suma_xn)
print('Vector de valores de cada una de las filas de las Xs =',nueva_fila)
#Suma total de los valores de cada uno de las xs
suma_xs = np.sum(nueva_fila)
print('Valor total de las xs =',suma_xs)
#x = [1,2,3,4,5,6,7,8,9,10] #número de muestras
x = np.linspace(5,15,10)

#Suma de las columnas
nueva_columna = []
for j in range(Columnas):
    suma_yn = np.sum([Filas[j] for Filas in matriz])
    nueva_columna.append(suma_yn)
print('vector de valores de cada una de las columnas de las ys = ',nueva_columna)
#Suma total de cada una de las ys
suma_ys = np.sum(nueva_columna)
print('Valor total de las ys =',suma_ys)
#y = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21] #número de muestras
y = np.linspace(5,25,21)


#Encontrando los parámetros de la mejor curva de ajuste de la función marginal de xs
mu1 = 0.027031758655329845
sigma1 = 0.090172
#función la función marginal de xs
def norm1(x,mu1,sigma1):
    return  1/(np.sqrt(2*np.pi*sigma1**2)) * np.exp(-(x-mu1)) ** 2/(2*sigma1**2)

xs1 = np.linspace(5,15,10)
ys1 = norm1(xs1,mu1,sigma1)

#Encontrando los parámetros de la mejor curva de ajuste de la función marginal de ys
mu2 = 0.015403387353115333
sigma2 = 0.04293904761904762
#función la función marginal de xs
def norm2(x,mu2,sigma2):
    return  1/(np.sqrt(2*np.pi*sigma2**2)) * np.exp(-(x-mu2)) ** 2/(2*sigma2**2)

xs2 = np.linspace(5,25,21)
ys2 = norm2(xs2,mu2,sigma2)

#función del modelo gaussiano

def gaussiana(x,mu,sigma):
    return  1/(np.sqrt(2*np.pi*sigma**2)) * np.exp(-(x-mu)) ** 2/(2*sigma**2)

param1,_ = curve_fit(gaussiana,xs1,ys1)
print('Parámetros de la mejor curva de ajuste de las Xs es =',param1)
param2,_ = curve_fit(gaussiana,xs2,ys2)
print('Parámetros de la mejor curva de ajuste de las Ys',param2)


'''
2.  Asumir independencia de X y Y. Analíticamente, 
¿cuál es entonces la expresión de la función de densidad conjunta que modela los datos?
'''
print('\n--- 2 ---\n')

fx_fy = suma_xs * suma_ys
print('La expresión de la función de densidad conjunta asumiendo independencia que modela los datos es =',fx_fy)


'''
3.  Hallar los valores de correlación, covarianza y coeficiente de correlación (Pearson) 
para los datos y explicar su significado.
'''
print('\n--- 3 ---\n')

#importación de datos mediante pandas
data = pd.read_csv('xyp.csv',header=0)
#print(data)
#print(data['p'])

#Sumatoria de los valores de la probabilidad de xy
p_xy = data['p'].sum()
xy_p = p_xy
print('El valor de XY para la correlación es =',p_xy)

#Correlación de XY
correlacion_xy = p_xy * (suma_xs * suma_ys)
print('Correlación de XY es =',correlacion_xy)

#Obtención de la media de X y Y

c,d = stats.norm.fit(nueva_fila) # obtencion de los paramertros de las xs
print('El valor de c es =',c)
print('El valor de la media  de Xs es =',d)
x_media = d
e,f = stats.norm.fit(nueva_columna) # obtencion de los paramertros de las ys
print('El valor de e es =',e)
print('El valor de la media de Ys es =',f)
y_media = f

#covarianza de XY
covarianza_xy = (suma_xs - x_media)*(suma_ys - y_media)* suma_ys * suma_xs
print('La covarianza de XY es =',covarianza_xy)
