import copy

# objetivo
objetivo = [[1,2,3],
            [4,5,6],
            [7,8,0]]   # 0 representa el espacio en blanco

# Heurística: número de posiciones correctas
def heuristica(estado):
    correctas = 0
    for i in range(3):
        for j in range(3):
            if estado[i][j] == objetivo[i][j] and estado[i][j] != 0:
                correctas += 1
    return correctas

# Buscar la posición del 0
def encontrar_cero(estado):
    for i in range(3):
        for j in range(3):
            if estado[i][j] == 0:
                return i, j

# Generar vecinos moviendo el espacio en blanco
def generar_vecinos(estado):
    vecinos = []
    x, y = encontrar_cero(estado)
    movimientos = [(-1,0),(1,0),(0,-1),(0,1)]  # arriba, abajo, izq, der

    for dx, dy in movimientos:
        nx, ny = x+dx, y+dy
        if 0 <= nx < 3 and 0 <= ny < 3:
            nuevo_estado = copy.deepcopy(estado)
            nuevo_estado[x][y], nuevo_estado[nx][ny] = nuevo_estado[nx][ny], nuevo_estado[x][y]
            vecinos.append(nuevo_estado)
    return vecinos

# Búsqueda en Escalada
def busqueda_en_escalada(inicial):
    """
    BÚSQUEDA EN ESCALADA PARA EL PUZZLE 8

    IDEA GENERAL DEL ALGORITMO:
    1.Definir el estado inicial y el estado objetivo.
    2.Calcular la heurística h(n)
    3.Generar los vecinos
    4.Escoger el vecino con mejor valor heurístico (más piezas en su lugar).
    5.Si el mejor vecino es mejor que el estado actual, moverse a él.
    6.Si no hay mejora -> se estanca (posible óptimo local).
    7.Repetir hasta llegar al objetivo o estancarse.

    VENTAJAS:
    - Simple de implementar y entender
    - Muy eficiente en memoria
    - Rápido cuando funciona
    
    DESVENTAJAS:
    - Puede quedarse atrapado en máximos locales
    - No garantiza encontrar la solución
    - Depende mucho del estado inicial
    """
    actual = inicial
    while True:
        if actual == objetivo:
            print("¡Solución encontrada!")
            return actual

        vecinos = generar_vecinos(actual)
        vecinos.sort(key=lambda v: heuristica(v), reverse=True)
        mejor = vecinos[0]

        if heuristica(mejor) <= heuristica(actual):
            print("Estancado en óptimo local.")
            return actual

        actual = mejor

# Ejemplo 1
print("Ejemplo1:")
estado_inicial = [[2,8,3],
                  [1,6,4],
                  [7,0,5]]

resultado = busqueda_en_escalada(estado_inicial)
print("Resultado final:")
for fila in resultado:
    print(fila)

# Ejemplo 2
print("Ejemplo2:")
estado_inicial2 = [[1,2,3],
                  [0,5,6],
                  [4,7,8]]

resultado2 = busqueda_en_escalada(estado_inicial2)
print("Resultado final:")
for fila in resultado2:
    print(fila)
