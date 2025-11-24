
#------------------------------------------------------------------------------------------

from sklearn import datasets
import numpy as np

# # Cargar el dataset de dígitos
datos_numeros = datasets.load_digits()

# Calcular la imagen promedio para cada dígito
imagenes_promedio = {}
for i in range(len(datos_numeros["images"])): #0 - 1796
    target = datos_numeros["target"][i]
    matriz_imagen = datos_numeros["images"][i]
    if target in imagenes_promedio:
        imagenes_promedio[target].append(matriz_imagen)
    else:
        imagenes_promedio[target] = [matriz_imagen]

# Calcular el promedio de las imágenes para cada dígito
for key in imagenes_promedio:
    imagenes_promedio[key] = np.mean(imagenes_promedio[key], axis=0)
    imagenes_promedio[key] = imagenes_promedio[key].astype(int)

#-----------------------------------------------------------------------

import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 5, figsize=(12, 6))
fig.suptitle('Imágenes Promedio de Dígitos')

# Mostrar las imágenes promedio en los subplots
for i in range(10):
    row = i // 5
    col = i % 5
    axs[row, col].imshow(imagenes_promedio[i], cmap='gray')
    axs[row, col].set_title(f'Dígito {i}')
    axs[row, col].axis('off')

plt.tight_layout()
plt.show()
print("")

# -----------------------------------------------------------------

import cv2
numero = input("Ingrese el nombre de la imagen que desea clasificar: ")
una_imagen = cv2.imread(f'{numero}.png', cv2.IMREAD_GRAYSCALE)
imagen_pequena = cv2.resize(una_imagen, (8, 8))
#invertir
imagen_pequena = 255 - imagen_pequena
#escalar a 16
imagen_pequena = (16/255) * imagen_pequena
imagen_pequena = imagen_pequena.astype(int) #para enteros

# -------------------------------------------------------------------

distancias = []
for imagen in datos_numeros["images"]:
    resta = imagen_pequena - imagen
    cuadrado = resta ** 2
    suma = np.sum(cuadrado)
    raiz = np.sqrt(suma)
    distancias.append(int(raiz))


# ----------------------------------------------------------------------

n = 3
while n<100:
    indices_mas_cercanos = sorted(range(len(distancias)), key=lambda i: distancias[i])[:n]
    targets = [datos_numeros["target"][indice] for indice in indices_mas_cercanos]
    mas_repetido = 0

    mayoria = False
    for target in targets:
      if targets.count(target) > len(targets) // 2:
        mas_repetido = int(target)
        mayoria = True
        break

    if mayoria:
        break
    else:
        n += 2

print(f'\nSe han considerado {n} vecinos más cercanos:')
for indice in indices_mas_cercanos:
    print(f'Índice: {indice}, Distancia: {distancias[indice]}, Target: {datos_numeros["target"][indice]}')

print(f'''
╔═══════════════════════════════════════════════════════╗
║ Soy la inteligencia artificial, y he detectado        ║
║ que el dígito ingresado corresponde al número {str(mas_repetido).ljust(7)} ║
╚═══════════════════════════════════════════════════════╝
                    ¯\_( ͡❛ ʖ ͡❛)_/¯ ''')

# #--------------------------------------------------------------------------

print(f"\nAhora calcularemos los vecinos con los promedios...")

distancias = []
for indice in imagenes_promedio:
  resta = imagen_pequena - imagenes_promedio[indice]
  cuadrado = resta ** 2
  suma = np.sum(cuadrado)
  raiz = np.sqrt(suma)
  distancias.append(int(raiz))


indices_mas_cercanos = sorted(range(len(distancias)), key=lambda i: distancias[i])[:1]
clasificacion = indices_mas_cercanos[0]

print(f'\nSe han considerado el vecino más cercano:')
print(f'Distancia: {distancias[indices_mas_cercanos[0]]}, Número: {clasificacion}')

print(f'''
╔═══════════════════════════════════════════════════════════╗
║ Soy la inteligencia artificial 2.0, y he detectado        ║
║ que el dígito ingresado corresponde al número {str(clasificacion).ljust(11)} ║
╚═══════════════════════════════════════════════════════════╝
                   ¯\_( ͠~ ʖ ͠° )_/¯ ''')