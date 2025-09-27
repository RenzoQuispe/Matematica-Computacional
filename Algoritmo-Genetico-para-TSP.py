import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import copy
import math

class GeneticTSP:
    """
    ALGORITMO GENÉTICO PARA EL PROBLEMA DEL VIAJERO (TSP)

    - POBLACIÓN: Conjunto de individuos (soluciones candidatas)
    - INDIVIDUO/CROMOSOMA: Una ruta específica que visita todas las ciudades
    - GEN: Una ciudad en la ruta
    - FITNESS: Qué tan buena es una solución (inverso de la distancia total)
    - SELECCIÓN: Los individuos más aptos tienen mayor probabilidad de reproducirse
    - CRUZAMIENTO/CROSSOVER: Dos padres crean descendencia combinando sus características
    - MUTACIÓN: Pequeños cambios aleatorios para mantener diversidad genética
    - GENERACIÓN: Una iteración completa del algoritmo evolutivo
    """
    
    def __init__(self, distance_matrix: np.ndarray, population_size: int = 100, 
                 mutation_rate: float = 0.02, elite_size: int = 20, generations: int = 500):
        """
        INICIALIZACIÓN DEL ECOSISTEMA GENÉTICO
        
        Args:
            distance_matrix: Matriz de distancias entre ciudades (el "ambiente")
            population_size: Tamaño de la población (cuántos individuos coexisten)
            mutation_rate: Tasa de mutación (probabilidad de cambios genéticos aleatorios)
            elite_size: Número de mejores individuos que pasan a la siguiente generación
            generations: Número de generaciones a evolucionar
        """
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.generations = generations
        
        # Historia evolutiva para análisis
        self.fitness_history = []
        self.best_distance_history = []
    
    def create_individual(self) -> List[int]:
        """
        GÉNESIS: Crear un individuo (cromosoma) aleatorio
        
        En biología: Como la formación aleatoria de un nuevo organismo
        En TSP: Una ruta aleatoria que visita todas las ciudades exactamente una vez
        
        Returns:
            Lista de ciudades representando una ruta (cromosoma)
        """
        # Crear una permutación aleatoria de las ciudades (excluyendo la ciudad inicial 0)
        cities = list(range(1, self.num_cities))
        random.shuffle(cities)
        return [0] + cities  # Siempre empezamos desde la ciudad 0
    
    def create_initial_population(self) -> List[List[int]]:
        """
        CREACIÓN DE LA POBLACIÓN INICIAL
        
        En biología: Como la generación fundadora de una especie en un nuevo hábitat
        En TSP: Conjunto de rutas aleatorias que formarán la primera generación
        
        Returns:
            Lista de individuos (población inicial)
        """
        population = []
        for i in range(self.population_size):
            individual = self.create_individual()
            population.append(individual)
        
        print(f"   POBLACIÓN INICIAL CREADA: {self.population_size} individuos")
        print(f"   Cada individuo representa una ruta visitando {self.num_cities} ciudades")
        
        return population
    
    def calculate_fitness(self, individual: List[int]) -> float:
        """
        EVALUACIÓN DEL FITNESS (APTITUD)
        
        En biología: Capacidad de un organismo para sobrevivir y reproducirse
        En TSP: Qué tan buena es una ruta (menor distancia = mayor fitness)
        
        Args:
            individual: Ruta a evaluar
            
        Returns:
            Valor de fitness (mayor es mejor)
        """
        total_distance = 0
        
        # Calcular distancia total de la ruta
        for i in range(len(individual)):
            from_city = individual[i]
            to_city = individual[(i + 1) % len(individual)]  # Regresar al origen
            total_distance += self.distance_matrix[from_city][to_city]
        
        # FITNESS = 1 / distancia (menor distancia = mayor fitness)
        # Agregamos un pequeño valor para evitar división por cero
        fitness = 1 / (total_distance + 1e-10)
        
        return fitness
    
    def evaluate_population(self, population: List[List[int]]) -> List[Tuple[List[int], float]]:
        """
        EVALUACIÓN DE TODA LA POBLACIÓN
        
        En biología: Como evaluar la aptitud de todos los individuos en un ecosistema
        En TSP: Calcular el fitness de todas las rutas en la población
        
        Args:
            population: Lista de individuos
            
        Returns:
            Lista de tuplas (individuo, fitness) ordenada por fitness descendente
        """
        fitness_results = []
        
        for individual in population:
            fitness = self.calculate_fitness(individual)
            fitness_results.append((individual, fitness))
        
        # Ordenar por fitness (mayor fitness primero - supervivencia del más apto)
        fitness_results.sort(key=lambda x: x[1], reverse=True)
        
        return fitness_results
    
    def selection(self, ranked_population: List[Tuple[List[int], float]], 
                  num_parents: int) -> List[List[int]]:
        """
        SELECCIÓN NATURAL - TORNEO/RULETA
        
        En biología: Los individuos más aptos tienen mayor probabilidad de reproducirse
        En TSP: Seleccionar padres basado en su fitness para crear descendencia
        
        Usa selección por torneo: cada padre se elige comparando individuos aleatorios
        
        Args:
            ranked_population: Población ordenada por fitness
            num_parents: Número de padres a seleccionar
            
        Returns:
            Lista de padres seleccionados
        """
        parents = []
        
        # ELITISMO: Los mejores individuos automáticamente pasan a la siguiente generación
        for i in range(self.elite_size):
            parents.append(ranked_population[i][0])
        
        # SELECCIÓN POR TORNEO para el resto
        tournament_size = 3
        for _ in range(num_parents - self.elite_size):
            # Seleccionar candidatos aleatorios para el torneo
            tournament = random.sample(ranked_population, tournament_size)
            # El ganador del torneo (mayor fitness) se convierte en padre
            winner = max(tournament, key=lambda x: x[1])
            parents.append(winner[0])
        
        print(f"   SELECCIÓN COMPLETADA: {len(parents)} padres seleccionados")
        print(f"   Elite preservada: {self.elite_size} mejores individuos")
        print(f"   Torneo para el resto: {num_parents - self.elite_size} individuos")
        
        return parents
    
    def order_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        CRUZAMIENTO GENÉTICO - ORDER CROSSOVER (OX)
        
        En biología: Dos padres crean descendencia combinando material genético
        En TSP: Crear nuevas rutas combinando segmentos de rutas padre
        
        OX es específico para TSP - mantiene el orden relativo de las ciudades
        
        Args:
            parent1, parent2: Rutas padre
            
        Returns:
            Tupla con dos hijos
        """
        size = len(parent1)
        
        # Seleccionar dos puntos de corte aleatorios
        start, end = sorted(random.sample(range(1, size), 2))  # Excluir ciudad 0
        
        # CREAR HIJO 1
        child1 = [None] * size
        child1[0] = 0  # Siempre empezar desde ciudad 0
        child1[start:end] = parent1[start:end]  # Heredar segmento del padre 1
        
        # Llenar posiciones restantes con genes del padre 2 en orden
        remaining_cities = [city for city in parent2 if city not in child1[start:end] and city != 0]
        j = 0
        for i in range(1, size):  # Empezar desde 1, ya que 0 está fija
            if child1[i] is None:
                child1[i] = remaining_cities[j]
                j += 1
        
        # CREAR HIJO 2 (proceso inverso)
        child2 = [None] * size
        child2[0] = 0
        child2[start:end] = parent2[start:end]
        
        remaining_cities = [city for city in parent1 if city not in child2[start:end] and city != 0]
        j = 0
        for i in range(1, size):
            if child2[i] is None:
                child2[i] = remaining_cities[j]
                j += 1
        
        return child1, child2
    
    def mutate(self, individual: List[int]) -> List[int]:
        """
        MUTACIÓN GENÉTICA
        
        En biología: Cambios aleatorios en el material genético
        En TSP: Intercambio aleatorio de ciudades para mantener diversidad
        
        Usa mutación por intercambio (swap mutation)
        
        Args:
            individual: Ruta a mutar
            
        Returns:
            Individuo mutado
        """
        mutated = individual.copy()
        
        if random.random() < self.mutation_rate:
            # Seleccionar dos posiciones aleatorias (excluyendo la ciudad inicial)
            idx1, idx2 = random.sample(range(1, len(individual)), 2)
            # Intercambiar ciudades (mutación)
            mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        
        return mutated
    
    def create_offspring(self, parents: List[List[int]]) -> List[List[int]]:
        """
        REPRODUCCIÓN - CREAR DESCENDENCIA
        
        En biología: Proceso de reproducción sexual donde los padres crean hijos
        En TSP: Combinar rutas padre para crear nuevas rutas hijo
        
        Args:
            parents: Lista de padres seleccionados
            
        Returns:
            Nueva población (descendencia)
        """
        offspring = []
        
        # Los elite pasan directamente (supervivencia garantizada)
        for i in range(self.elite_size):
            offspring.append(parents[i])
        
        # Crear resto de descendencia mediante cruzamiento
        for i in range(self.elite_size, self.population_size, 2):
            # Seleccionar dos padres aleatorios
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            
            # CRUZAMIENTO: crear dos hijos
            child1, child2 = self.order_crossover(parent1, parent2)
            
            # MUTACIÓN: aplicar mutación a los hijos
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            offspring.append(child1)
            if len(offspring) < self.population_size:
                offspring.append(child2)
        
        # Asegurar tamaño exacto de población
        while len(offspring) < self.population_size:
            parent = random.choice(parents)
            child = self.mutate(parent.copy())
            offspring.append(child)
        
        return offspring[:self.population_size]
    
    def get_route_distance(self, route: List[int]) -> float:
        """
        Calcular distancia total de una ruta
        
        Args:
            route: Lista de ciudades en orden de visita
            
        Returns:
            Distancia total de la ruta
        """
        total_distance = 0
        for i in range(len(route)):
            from_city = route[i]
            to_city = route[(i + 1) % len(route)]
            total_distance += self.distance_matrix[from_city][to_city]
        return total_distance
    
    def evolve(self) -> Tuple[List[int], float]:
        """
        PROCESO EVOLUTIVO COMPLETO
        
        En biología: Evolución de especies a través de generaciones
        En TSP: Mejoramiento iterativo de soluciones a través de generaciones
        
        Returns:
            Tupla con la mejor ruta encontrada y su distancia
        """
        print("\nINICIANDO EVOLUCIÓN GENÉTICA")
        
        # GÉNESIS: Crear población inicial
        population = self.create_initial_population()
        
        best_route = None
        best_distance = float('inf')
        
        # EVOLUCIÓN A TRAVÉS DE GENERACIONES
        for generation in range(self.generations):
            print(f"\n  GENERACIÓN {generation + 1}/{self.generations}")
            
            # 1. EVALUACIÓN: Calcular fitness de toda la población
            ranked_population = self.evaluate_population(population)
            
            # 2. REGISTRO DE ESTADÍSTICAS
            current_best_fitness = ranked_population[0][1]
            current_best_route = ranked_population[0][0]
            current_best_distance = self.get_route_distance(current_best_route)
            
            # Actualizar mejor solución global
            if current_best_distance < best_distance:
                best_route = current_best_route.copy()
                best_distance = current_best_distance
                print(f"      NUEVA MEJOR RUTA ENCONTRADA: Distancia = {best_distance:.2f}")
            
            # Guardar estadísticas
            self.fitness_history.append(current_best_fitness)
            self.best_distance_history.append(best_distance)
            
            # 3. SELECCIÓN: Elegir padres para reproducción
            num_parents = self.population_size // 2
            parents = self.selection(ranked_population, num_parents)
            
            # 4. REPRODUCCIÓN: Crear nueva generación
            population = self.create_offspring(parents)
            
            # Mostrar progreso cada 50 generaciones
            if (generation + 1) % 50 == 0:
                avg_fitness = np.mean([fit for _, fit in ranked_population])
                print(f"      Estadísticas Generación {generation + 1}:")
                print(f"      Mejor distancia: {current_best_distance:.2f}")
                print(f"      Fitness promedio: {avg_fitness:.6f}")
                print(f"      Diversidad: {len(set(str(ind) for ind in population))} rutas únicas")
        
        print("\nEVOLUCIÓN COMPLETADA")
        print(f"MEJOR RUTA ENCONTRADA:")
        print(f"   Ruta: {' -> '.join(map(str, best_route))} -> {best_route[0]}")
        print(f"   Distancia total: {best_distance:.2f}")
        
        return best_route, best_distance
    
    def plot_evolution(self, save_plot=True):
        """
        Visualizar el progreso evolutivo
        """
        # Configurar matplotlib para modo no interactivo
        plt.ioff()  # Desactivar modo interactivo
        
        plt.figure(figsize=(12, 5))
        
        # Gráfico 1: Evolución del fitness
        plt.subplot(1, 2, 1)
        plt.plot(self.fitness_history, 'b-', linewidth=2)
        plt.title('Evolución del Fitness\n(Supervivencia del Más Apto)', fontsize=12)
        plt.xlabel('Generación')
        plt.ylabel('Fitness (1/distancia)')
        plt.grid(True, alpha=0.3)
        
        # Gráfico 2: Evolución de la mejor distancia
        plt.subplot(1, 2, 2)
        plt.plot(self.best_distance_history, 'r-', linewidth=2)
        plt.title('Evolución de la Mejor Distancia\n(Presión Selectiva)', fontsize=12)
        plt.xlabel('Generación')
        plt.ylabel('Mejor Distancia')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            filename = 'tsp_evolution.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Gráfico guardado como: {filename}")
        else:
            try:
                plt.show()
            except Exception as e:
                print(f"No se puede mostrar el gráfico interactivo: {e}")
                filename = 'tsp_evolution.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Gráfico guardado como: {filename}")
        
        plt.close()  # Cerrar figura para liberar memoria


def create_sample_distance_matrix(num_cities: int) -> np.ndarray:
    """
    Crear matriz de distancias de ejemplo
    """
    # Generar coordenadas aleatorias para las ciudades
    np.random.seed(42)  # Para resultados reproducibles
    coords = np.random.uniform(0, 100, (num_cities, 2))
    
    # Calcular matriz de distancias euclidiana
    distance_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                distance_matrix[i][j] = np.sqrt(
                    (coords[i][0] - coords[j][0])**2 + 
                    (coords[i][1] - coords[j][1])**2
                )
    
    return distance_matrix


# EJEMPLO DE USO
if __name__ == "__main__":
    print("ALGORITMO GENÉTICO PARA TSP")
    print("Simulando evolución artificial para encontrar la mejor ruta\n")
    
    # Crear problema de ejemplo con 10 ciudades
    num_cities = 10
    distance_matrix = create_sample_distance_matrix(num_cities)
    
    print(f"PROBLEMA: Encontrar la mejor ruta visitando {num_cities} ciudades")
    print(f"ESPACIO DE BÚSQUEDA: {math.factorial(num_cities-1):,} rutas posibles")
    
    # Configurar y ejecutar algoritmo genético
    ga = GeneticTSP(
        distance_matrix=distance_matrix,
        population_size=25,      # Tamaño de la población
        mutation_rate=0.02,       # 2% de probabilidad de mutación
        elite_size=10,            # 20 mejores individuos pasan automáticamente
        generations=50         # 100 generaciones de evolución
    )
    
    # EVOLUCIONAR
    best_route, best_distance = ga.evolve()
    
    # Visualizar evolución (guardará la imagen como archivo)
    ga.plot_evolution(save_plot=True)
    
    print(f"\nRESULTADO FINAL:")
    print(f"Mejor ruta evolutiva: {best_route}")
    print(f"Distancia: {best_distance:.2f}")
    
    # Comparación con ruta aleatoria
    random_route = ga.create_individual()
    random_distance = ga.get_route_distance(random_route)
    improvement = ((random_distance - best_distance) / random_distance) * 100
    
    print(f"\nMEJORA vs SOLUCIÓN ALEATORIA:")
    print(f"Ruta aleatoria: {random_distance:.2f}")
    print(f"Mejora evolutiva: {improvement:.1f}%")