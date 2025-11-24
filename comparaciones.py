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


# -----------------------------------------------------------------

target_dataset = [1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,0,0,0,2,3,4,5,6,7,8,9, #30 0-29
                  0,0,0,1,1,1,2,2,3,3,3,4,5,5,5,4,6,6,6,7,7,7,8,8,8,9,9,9,2,4, #30 30-59
                  0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8,9,9,9] #30 60-89

aciertos_distancias = 0
aciertos_promedios = 0
datos = 0
import cv2

for x in range(60,90): 
    datos += 1
    
    target_actual = target_dataset[x]
    
    
    una_imagen = cv2.imread(f'dataset/{x}.jpeg', cv2.IMREAD_GRAYSCALE)
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
        
    n = 3
    mas_repetido = 0
    
    for x in range(101):
        indices_mas_cercanos = sorted(range(len(distancias)), key=lambda i: distancias[i])[:n]
        targets = [datos_numeros["target"][indice] for indice in indices_mas_cercanos]
        
    
        mayoria = False
        for target in targets:
          if targets.count(target) > len(targets) // 2:
            mas_repetido = int(target)
            mayoria = True
            break
    
        if mayoria:
            break
        else:
            mas_repetido = sorted(set(targets), key=targets.count, reverse=True)[0]
            mas_repetido = int(mas_repetido)
            n += 2
    
    if mas_repetido == target_actual:
        aciertos_distancias += 1

    # ----------------------------------------------------------------------
    distancias = []
    for indice in imagenes_promedio:
      resta = imagen_pequena - imagenes_promedio[indice]
      cuadrado = resta ** 2
      suma = np.sum(cuadrado)
      raiz = np.sqrt(suma)
      distancias.append(int(raiz))
    
    
    indices_mas_cercanos = sorted(range(len(distancias)), key=lambda i: distancias[i])[:1]
    clasificacion = indices_mas_cercanos[0]

    if clasificacion == target_actual:
        aciertos_promedios += 1

print(f""" 
╔═════════════════════════════════════════════════════════════╗
║ El número de acierto obtenidos por el prime metódo fue: {aciertos_distancias}  ║
║ Es decir: {int((aciertos_distancias*100) / datos)} %                                              ║
║                                                             ║
║ El número de acierto obtenidos por el segundo metódo fue: {aciertos_promedios}║
║ Es decir: {int((aciertos_promedios*100) / datos)} %                                              ║
╚═════════════════════════════════════════════════════════════╝
""")
