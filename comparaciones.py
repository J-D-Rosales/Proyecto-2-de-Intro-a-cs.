import cv2
import numpy as np
from sklearn import datasets

# =============================================================================
# 1. FUNCIONES (ARREGLADO)
# =============================================================================

def procesar_imagen(nombre_archivo):
    # Cargar imagen en escala de grises
    img = cv2.imread(nombre_archivo, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None

    # --- CORRECCIÓN CRÍTICA ---
    # Faltaba esta línea. Es la que invierte los colores (Blanco a Negro) y limpia el ruido.
    # Sin esto, comparas una "foto negativa" con una "positiva" y falla todo.
    _, img_binaria = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)

    # Reducir a 8x8 pixeles (Usamos img_binaria, que ya está invertida)
    img_pequena = cv2.resize(img_binaria, (8, 8))

    # Escalar valores de 0-255 a 0-16
    img_final = (img_pequena / 255) * 16

    return img_final.astype(int).flatten()


def calcular_distancia(imagen1, imagen2):
    # Distancia Euclidiana simple
    return np.sqrt(np.sum((imagen1 - imagen2) ** 2))


# =============================================================================
# 2. CARGA DE DATOS
# =============================================================================
digits = datasets.load_digits()
biblioteca_imagenes = digits.data
biblioteca_etiquetas = digits.target

# =============================================================================
# 3. CONFIGURACIÓN (TUS FOTOS)
# =============================================================================
mis_archivos = [
    # --- TANDA 1 (0-29) ---
    ("dataset/0.jpeg", 1), ("dataset/1.jpeg", 2), ("dataset/2.jpeg", 3),
    ("dataset/3.jpeg", 4), ("dataset/4.jpeg", 5), ("dataset/5.jpeg", 6),
    ("dataset/6.jpeg", 7), ("dataset/7.jpeg", 8), ("dataset/8.jpeg", 9),
    ("dataset/9.jpeg", 1), ("dataset/10.jpeg", 2), ("dataset/11.jpeg", 3),
    ("dataset/12.jpeg", 4), ("dataset/13.jpeg", 5), ("dataset/14.jpeg", 6),
    ("dataset/15.jpeg", 7), ("dataset/16.jpeg", 8), ("dataset/17.jpeg", 9),
    ("dataset/18.jpeg", 1), ("dataset/19.jpeg", 0), ("dataset/20.jpeg", 0),
    ("dataset/21.jpeg", 0), ("dataset/22.jpeg", 2), ("dataset/23.jpeg", 3),
    ("dataset/24.jpeg", 4), ("dataset/25.jpeg", 5), ("dataset/26.jpeg", 6),
    ("dataset/27.jpeg", 7), ("dataset/28.jpeg", 8), ("dataset/29.jpeg", 9),

    # --- TANDA 2 (30-59) ---
    ("dataset/30.jpeg", 0), ("dataset/31.jpeg", 0), ("dataset/32.jpeg", 0),
    ("dataset/33.jpeg", 1), ("dataset/34.jpeg", 1), ("dataset/35.jpeg", 1),
    ("dataset/36.jpeg", 2), ("dataset/37.jpeg", 2), ("dataset/38.jpeg", 3),
    ("dataset/39.jpeg", 3), ("dataset/40.jpeg", 3), ("dataset/41.jpeg", 4),
    ("dataset/42.jpeg", 5), ("dataset/43.jpeg", 5), ("dataset/44.jpeg", 5),
    ("dataset/45.jpeg", 4), ("dataset/46.jpeg", 6), ("dataset/47.jpeg", 6),
    ("dataset/48.jpeg", 6), ("dataset/49.jpeg", 7), ("dataset/50.jpeg", 7),
    ("dataset/51.jpeg", 7), ("dataset/52.jpeg", 8), ("dataset/53.jpeg", 8),
    ("dataset/54.jpeg", 8), ("dataset/55.jpeg", 9), ("dataset/56.jpeg", 9),
    ("dataset/57.jpeg", 9), ("dataset/58.jpeg", 2), ("dataset/59.jpeg", 4),

    # --- TANDA 3 (60-89) ---
    ("dataset/60.jpeg", 0), ("dataset/61.jpeg", 0), ("dataset/62.jpeg", 0),
    ("dataset/63.jpeg", 1), ("dataset/64.jpeg", 1), ("dataset/65.jpeg", 1),
    ("dataset/66.jpeg", 2), ("dataset/67.jpeg", 2), ("dataset/68.jpeg", 2),
    ("dataset/69.jpeg", 3), ("dataset/70.jpeg", 3), ("dataset/71.jpeg", 3),
    ("dataset/72.jpeg", 4), ("dataset/73.jpeg", 4), ("dataset/74.jpeg", 4),
    ("dataset/75.jpeg", 5), ("dataset/76.jpeg", 5), ("dataset/77.jpeg", 5),
    ("dataset/78.jpeg", 6), ("dataset/79.jpeg", 6), ("dataset/80.jpeg", 6),
    ("dataset/81.jpeg", 7), ("dataset/82.jpeg", 7), ("dataset/83.jpeg", 7),
    ("dataset/84.jpeg", 8), ("dataset/85.jpeg", 8), ("dataset/86.jpeg", 8),
    ("dataset/87.jpeg", 9), ("dataset/88.jpeg", 9), ("dataset/89.jpeg", 9)
]

# =============================================================================
# 4. PROGRAMA PRINCIPAL CON DESEMPATE
# =============================================================================

aciertos = 0
total = 0

print(f"\n{'ARCHIVO':<15} | {'REAL':<5} | {'PREDICHO':<8} | {'VECINOS (Ordenados por cercanía)'}")
print("-" * 75)

for nombre, real in mis_archivos:
    mi_dibujo = procesar_imagen(nombre)
    if mi_dibujo is None: continue

    # Calcular distancias
    lista_distancias = []
    for i, foto_biblio in enumerate(biblioteca_imagenes):
        dist = calcular_distancia(mi_dibujo, foto_biblio)
        label = biblioteca_etiquetas[i]
        lista_distancias.append((dist, label))

    # Ordenar: El índice 0 es el MÁS CERCANO (menor distancia)
    lista_distancias.sort(key=lambda x: x[0])
    tres_vecinos = lista_distancias[:3]
    votos = [v[1] for v in tres_vecinos]

    # --- LÓGICA DE VOTACIÓN Y DESEMPATE ---

    # 1. Contar votos
    conteo_votos = {}
    for v in votos:
        conteo_votos[v] = conteo_votos.get(v, 0) + 1

    # 2. Buscar cuántos votos tiene el ganador
    max_votos = max(conteo_votos.values())

    # 3. Ver quiénes empataron con ese máximo
    candidatos = [num for num, count in conteo_votos.items() if count == max_votos]

    if len(candidatos) == 1:
        # CASO 1: Mayoría simple (Gana el que tiene más votos)
        ganador = candidatos[0]
    else:
        # CASO 2: Empate (Ej: [9, 1, 8])
        # REGLA: Gana el vecino que estaba matemáticamente más cerca (índice 0)
        ganador = votos[0]

    # ---------------------------------------

    print(f"{nombre:<15} | {real:<5} | {ganador:<8} | {votos}")

    if ganador == real:
        aciertos += 1
    total += 1

print("-" * 75)
if total > 0:
    print(f"Precisión Final: {(aciertos / total) * 100:.2f}% ({aciertos}/{total})")