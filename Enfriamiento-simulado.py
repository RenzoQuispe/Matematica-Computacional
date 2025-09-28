import math, random

# Distancias entre ciudades (matriz simétrica)
distancias = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]

# Calcular distancia de un recorrido
def costo(ruta):
    total = 0
    for i in range(len(ruta)-1):
        total += distancias[ruta[i]][ruta[i+1]]
    total += distancias[ruta[-1]][ruta[0]]  # volver a la inicial
    return total

# Generar vecino: intercambio de dos ciudades
def generar_vecino(ruta):
    a, b = random.sample(range(len(ruta)), 2)
    nueva = ruta[:]
    nueva[a], nueva[b] = nueva[b], nueva[a]
    return nueva

# Enfriamiento simulado
def simulated_annealing_tsp(ruta_inicial, T=100, alpha=0.95, max_iter=50):
    """
    ENFRIAMIENTO SIMULADO PARA EL PROBLEMA DEL VIAJERO (TSP)

    IDEA GENERAL DEL ALGORITMO:
    1. Comenzar con una solución inicial y una temperatura alta.
    2. Generar un vecino (solución cercana).
    3. Calcular el cambio en el costo (Δ).
    4. Si Δ < 0 (mejora), aceptar el vecino.
       Si Δ >= 0, aceptar con probabilidad P = e^(-Δ/T)
    5. Reducir la temperatura (T = T * alpha).
    6. Repetir hasta que la temperatura sea baja o se alcance el máximo de iteraciones.
    VENTAJAS:
    - Puede escapar de óptimos locales
    - Simple de implementar
    - Flexible para diferentes problemas
    DESVENTAJAS:
    - Requiere ajuste de parámetros (T inicial, alpha)
    - Puede ser lento para converger
    - No garantiza encontrar la solución óptima
    """
    actual = ruta_inicial
    costo_actual = costo(actual)

    for iteracion in range(max_iter):
        vecino = generar_vecino(actual)
        costo_vecino = costo(vecino)

        delta = costo_vecino - costo_actual

        if delta < 0:
            aceptar = True
        else:
            P = math.exp(-delta / T)
            aceptar = random.random() < P   # random.random() genera [0.0, 1.0>, aceptar si P es mayor

        print(f"Iter {iteracion} | T={T:.2f} | costo_actual={costo_actual} | costo_vecino={costo_vecino} | Δ={delta} | aceptar={aceptar}")

        if aceptar:
            actual, costo_actual = vecino, costo_vecino

        T *= alpha
        if T < 0.001:
            break

    return actual, costo_actual

# Ejemplo
ruta_inicial = [0, 1, 2, 3]
mejor_ruta, mejor_costo = simulated_annealing_tsp(ruta_inicial)
print("\nMejor ruta encontrada:", mejor_ruta)
print("Costo:", mejor_costo)
