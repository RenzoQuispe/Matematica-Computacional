import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

"""
Implementación del Método de Penalización Interior con Barrera Logarítmica
"""

class Metodo_Penalizacion_Interior:

    def __init__(self, objective_func, objective_grad, constraints, constraints_grad, 
                 x0, mu0=1.0, beta=0.5, epsilon=0.01, grad_tol=0.1, max_outer=20, max_inner=50):
        """
        Parámetros:
        -----------
        objective_func : callable
            Función objetivo f(x)
        objective_grad : callable
            Gradiente de la función objetivo
        constraints : list of callables
            Lista de funciones de restricción g_i(x) <= 0
        constraints_grad : list of callables
            Lista de gradientes de las restricciones
        x0 : array-like
            Punto inicial estrictamente factible
        mu0 : float
            Parámetro inicial de barrera
        beta : float
            Factor de reducción (0 < beta < 1)
        epsilon : float
            Tolerancia para el parámetro mu
        grad_tol : float
            Tolerancia para el gradiente
        max_outer : int
            Máximo número de iteraciones externas
        max_inner : int
            Máximo número de iteraciones internas
        """
        self.f = objective_func
        self.grad_f = objective_grad
        self.g = constraints
        self.grad_g = constraints_grad
        self.x0 = np.array(x0)
        self.mu0 = mu0
        self.beta = beta
        self.epsilon = epsilon
        self.grad_tol = grad_tol
        self.max_outer = max_outer
        self.max_inner = max_inner
        
        # Para almacenar el historial
        self.history = {
            'outer_iterations': [],
            'mu_values': [],
            'solutions': [],
            'inner_iterations': [],
            'convergence': []
        }
    
    def is_feasible(self, x): # Verifica si el punto x es estrictamente factible
        for gi in self.g:
            if gi(x) >= 0:
                return False
        return True
    
    def barrier_function(self, x): # Función barrera logarítmica φ(x) = -Σ ln(-g_i(x))
        phi = 0
        for gi in self.g:
            gi_val = gi(x)
            if gi_val >= 0:
                return np.inf  # Penalización infinita si no es factible
            phi -= np.log(-gi_val)
        return phi
    
    def barrier_gradient(self, x): # Gradiente de la función barrera
        grad_phi = np.zeros_like(x)
        for i, (gi, grad_gi) in enumerate(zip(self.g, self.grad_g)):
            gi_val = gi(x)
            if gi_val >= 0:
                return np.full_like(x, np.inf)
            grad_phi += grad_gi(x) / (-gi_val)
        return grad_phi
    
    def augmented_objective(self, x, mu):
        # Función objetivo aumentada F_μ(x) = f(x) + μφ(x)
        return self.f(x) + mu * self.barrier_function(x)
    
    def augmented_gradient(self, x, mu):
        # Gradiente de la función objetivo aumentada
        return self.grad_f(x) + mu * self.barrier_gradient(x)
    
    def line_search_backtracking(self, x, direction, mu, alpha0=1.0, rho=0.5, c1=1e-4):
        # Búsqueda lineal con backtracking que mantiene factibilidad
        alpha = alpha0
        f_current = self.augmented_objective(x, mu)
        grad_current = self.augmented_gradient(x, mu)
        
        for _ in range(20):  # Máximo 20 iteraciones de búsqueda lineal
            x_new = x + alpha * direction
            
            # Verificar factibilidad
            if not self.is_feasible(x_new):
                alpha *= rho
                continue
            
            # Condición de Armijo
            f_new = self.augmented_objective(x_new, mu)
            if f_new <= f_current + c1 * alpha * np.dot(grad_current, direction):
                return alpha, x_new
            
            alpha *= rho
        
        return alpha, x + alpha * direction
    
    def gradient_descent_subproblem(self, x_start, mu, verbose=False):
        # Resuelve el subproblema usando gradiente descendente
        x = x_start.copy()
        inner_history = []
        
        for k in range(self.max_inner):
            # Calcular gradiente
            grad = self.augmented_gradient(x, mu)
            grad_norm = np.linalg.norm(grad)
            
            # Información de la iteración
            f_val = self.f(x)
            phi_val = self.barrier_function(x)
            F_val = self.augmented_objective(x, mu)
            
            inner_history.append({
                'iteration': k,
                'x': x.copy(),
                'f': f_val,
                'phi': phi_val,
                'F_mu': F_val,
                'grad_norm': grad_norm
            })
            
            if verbose and k < 3:  # Mostrar solo las primeras 3 iteraciones
                print(f"    Iteración interna {k+1}:")
                print(f"      x = ({x[0]:.3f}, {x[1]:.3f})")
                print(f"      f(x) = {f_val:.4f}")
                print(f"      φ(x) = {phi_val:.4f}")
                print(f"      F_μ(x) = {F_val:.4f}")
                print(f"      ||∇F_μ|| = {grad_norm:.4f}")
            
            # Criterio de parada
            if grad_norm < self.grad_tol:
                if verbose:
                    print(f"    Convergencia tras {k+1} iteraciones internas")
                break
            
            # Dirección de descenso
            direction = -grad / grad_norm
            
            # Búsqueda lineal
            alpha, x_new = self.line_search_backtracking(x, direction, mu)
            x = x_new
        
        return x, inner_history
    
    def solve(self, verbose=True): # Resuelve el problema de optimización
        if not self.is_feasible(self.x0):
            raise ValueError("El punto inicial no es estrictamente factible")
        
        x = self.x0.copy()
        mu = self.mu0
        
        if verbose:
            print("MÉTODO DE PENALIZACIÓN INTERIOR")
            print(f"Punto inicial: x₀ = ({x[0]:.3f}, {x[1]:.3f})")
            print(f"Parámetros: μ₀ = {self.mu0}, β = {self.beta}, ε = {self.epsilon}")
            print()
        
        for outer_iter in range(self.max_outer):
            if verbose:
                print(f"ITERACIÓN EXTERNA {outer_iter + 1}: μ = {mu:.6f}")
            
            # Resolver subproblema
            x_new, inner_hist = self.gradient_descent_subproblem(x, mu, verbose)
            
            # Almacenar información
            self.history['outer_iterations'].append(outer_iter + 1)
            self.history['mu_values'].append(mu)
            self.history['solutions'].append(x_new.copy())
            self.history['inner_iterations'].append(len(inner_hist))
            
            # Verificar convergencia
            if mu < self.epsilon:
                if verbose:
                    print(f"\n¡CONVERGENCIA! μ = {mu:.6f} < ε = {self.epsilon}")
                self.history['convergence'].append(True)
                break
            
            # Actualizar
            x = x_new
            mu *= self.beta
            
            if verbose:
                print(f"  Solución: x*({mu/self.beta:.6f}) = ({x[0]:.4f}, {x[1]:.4f})")
                print(f"  Valor objetivo: f = {self.f(x):.6f}")
                print()
        
        return x, self.history
    
    def plot_convergence(self, show_constraints=True): # Grafica la convergencia del método
        if not self.history['solutions']:
            print("No hay datos de convergencia para graficar")
            return
        
        # Configurar matplotlib para el entorno actual
        plt.style.use('default')
        plt.rcParams['figure.facecolor'] = 'white'
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Trayectoria de los puntos con restricciones
        solutions = np.array(self.history['solutions'])
        
        if show_constraints:
            # Crear una malla para visualizar las restricciones
            x1_range = np.linspace(0, 3, 100)
            x2_range = np.linspace(0, 3, 100)
            X1, X2 = np.meshgrid(x1_range, x2_range)
            
            # Restricción 1: x1 + x2 >= 1
            constraint1_line_x1 = np.linspace(0, 3, 100)
            constraint1_line_x2 = 1 - constraint1_line_x1
            
            # Restricción 2: x1² + x2² <= 4 (círculo)
            theta = np.linspace(0, 2*np.pi, 100)
            circle_x1 = 2 * np.cos(theta)
            circle_x2 = 2 * np.sin(theta)
            
            # Dibujar restricciones
            axes[0,0].plot(constraint1_line_x1, constraint1_line_x2, 'r--', 
                          linewidth=2, label='x₁ + x₂ = 1')
            axes[0,0].plot(circle_x1, circle_x2, 'b--', 
                          linewidth=2, label='x₁² + x₂² = 4')
            
            # Sombrear región factible
            valid_mask = (X1 + X2 >= 1) & (X1**2 + X2**2 <= 4)
            axes[0,0].contourf(X1, X2, valid_mask.astype(int), 
                              levels=[0.5, 1.5], colors=['lightgreen'], alpha=0.3)
        
        # Trayectoria
        axes[0,0].plot(solutions[:, 0], solutions[:, 1], 'ko-', 
                      markersize=6, linewidth=2, label='Trayectoria')
        axes[0,0].plot(solutions[0, 0], solutions[0, 1], 'go', 
                      markersize=12, label='Inicio', zorder=5)
        axes[0,0].plot(solutions[-1, 0], solutions[-1, 1], 'ro', 
                      markersize=12, label='Final', zorder=5)
        axes[0,0].plot(2, 1, 'b*', markersize=15, 
                      label='Óptimo sin restricciones', zorder=5)
        
        axes[0,0].set_xlabel('x₁', fontsize=12)
        axes[0,0].set_ylabel('x₂', fontsize=12)
        axes[0,0].set_title('Trayectoria de Convergencia', fontsize=14, fontweight='bold')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].legend(fontsize=10)
        axes[0,0].set_xlim(0, 2.5)
        axes[0,0].set_ylim(0, 2.5)
        
        # 2. Evolución del parámetro μ
        axes[0,1].semilogy(self.history['outer_iterations'], self.history['mu_values'], 
                          'b.-', markersize=8, linewidth=2)
        axes[0,1].axhline(y=self.epsilon, color='r', linestyle='--', linewidth=2,
                         label=f'ε = {self.epsilon}')
        axes[0,1].set_xlabel('Iteración Externa', fontsize=12)
        axes[0,1].set_ylabel('μ (escala log)', fontsize=12)
        axes[0,1].set_title('Evolución del Parámetro μ', fontsize=14, fontweight='bold')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].legend(fontsize=10)
        
        # 3. Valor de la función objetivo
        f_values = [self.f(sol) for sol in self.history['solutions']]
        axes[1,0].plot(self.history['outer_iterations'], f_values, 
                      'g.-', markersize=8, linewidth=2)
        axes[1,0].set_xlabel('Iteración Externa', fontsize=12)
        axes[1,0].set_ylabel('f(x)', fontsize=12)
        axes[1,0].set_title('Evolución de la Función Objetivo', fontsize=14, fontweight='bold')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Número de iteraciones internas
        bars = axes[1,1].bar(self.history['outer_iterations'], self.history['inner_iterations'],
                           color='orange', alpha=0.7, edgecolor='black')
        axes[1,1].set_xlabel('Iteración Externa', fontsize=12)
        axes[1,1].set_ylabel('Iteraciones Internas', fontsize=12)
        axes[1,1].set_title('Esfuerzo Computacional por Iteración', fontsize=14, fontweight='bold')
        axes[1,1].grid(True, alpha=0.3)
        
        # Añadir valores en las barras
        for bar, value in zip(bars, self.history['inner_iterations']):
            axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                          str(value), ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout(pad=3.0)
        
        # Guardar la figura
        plt.savefig('convergencia_metodo_interior.png', dpi=300, bbox_inches='tight')
        print("\nGráfica guardada como 'convergencia_metodo_interior.png'")
        
        # Mostrar la figura
        plt.show()
        
        return fig


# DEFINICIÓN DEL PROBLEMA DEL EJEMPLO

def objective_function(x):
    # f(x₁, x₂) = (x₁ - 2)² + (x₂ - 1)²
    return (x[0] - 2)**2 + (x[1] - 1)**2

def objective_gradient(x):
    # ∇f(x₁, x₂) = [2(x₁ - 2), 2(x₂ - 1)]
    return np.array([2*(x[0] - 2), 2*(x[1] - 1)])

def constraint1(x):
    # g₁(x₁, x₂) = -x₁ - x₂ + 1 ≤ 0
    return -x[0] - x[1] + 1

def constraint1_gradient(x):
    # ∇g₁(x₁, x₂) = [-1, -1]
    return np.array([-1, -1])

def constraint2(x):
    # g₂(x₁, x₂) = x₁² + x₂² - 4 ≤ 0
    return x[0]**2 + x[1]**2 - 4

def constraint2_gradient(x):
    # ∇g₂(x₁, x₂) = [2x₁, 2x₂]
    return np.array([2*x[0], 2*x[1]])


# EJECUCIÓN DEL EJEMPLO

if __name__ == "__main__":
    
    print("=" * 60)
    print("MÉTODO DE PENALIZACIÓN INTERIOR - EJEMPLO COMPUTACIONAL")
    print("=" * 60)
    
    # Definir el problema
    constraints = [constraint1, constraint2]
    constraints_gradients = [constraint1_gradient, constraint2_gradient]
    
    # Punto inicial (debe ser estrictamente factible)
    x0 = np.array([0.7, 0.8])
    
    # Verificar factibilidad del punto inicial
    print(f"\nVerificación del punto inicial x₀ = ({x0[0]}, {x0[1]}):")
    print(f"g₁(x₀) = {constraint1(x0):.4f} < 0: {'✓' if constraint1(x0) < 0 else '✗'}")
    print(f"g₂(x₀) = {constraint2(x0):.4f} < 0: {'✓' if constraint2(x0) < 0 else '✗'}")
    
    # Crear y resolver el problema
    solver = Metodo_Penalizacion_Interior(
        objective_func=objective_function,
        objective_grad=objective_gradient,
        constraints=constraints,
        constraints_grad=constraints_gradients,
        x0=x0,
        mu0=1.0,
        beta=0.5,
        epsilon=0.01,
        grad_tol=0.1
    )
    
    # Resolver
    solution, history = solver.solve(verbose=True)
    
    # Mostrar resultados finales
    print("\n" + "="*50)
    print("RESULTADOS FINALES")
    print("="*50)
    print(f"Solución óptima: x* = ({solution[0]:.6f}, {solution[1]:.6f})")
    print(f"Valor objetivo: f(x*) = {objective_function(solution):.8f}")
    
    print(f"\nVerificación de restricciones:")
    print(f"g₁(x*) = {constraint1(solution):.6f} ≤ 0: {'✓' if constraint1(solution) <= 0 else '✗'}")
    print(f"g₂(x*) = {constraint2(solution):.6f} ≤ 0: {'✓' if constraint2(solution) <= 0 else '✗'}")
    
    print(f"\nEstadísticas de convergencia:")
    print(f"Iteraciones externas: {len(history['outer_iterations'])}")
    print(f"Total iteraciones internas: {sum(history['inner_iterations'])}")
    print(f"Promedio iteraciones internas: {np.mean(history['inner_iterations']):.1f}")
    
    # Comparar con solución analítica
    print(f"\nComparación con el óptimo sin restricciones (2, 1):")
    unconstrained_opt = np.array([2, 1])
    print(f"Distancia al óptimo sin restricciones: {np.linalg.norm(solution - unconstrained_opt):.6f}")
    
    # Graficar convergencia
    print("\nGenerando gráficas de convergencia...")
    try:
        fig = solver.plot_convergence(show_constraints=True)
        print("Gráficas generadas exitosamente!")
    except Exception as e:
        print(f"Error al generar gráficas: {e}")
    
    # También crear una gráfica simple de la trayectoria
    try:
        plt.figure(figsize=(10, 8))
        
        # Datos de la trayectoria
        solutions = np.array(history['solutions'])
        
        # Dibujar restricciones
        x1_vals = np.linspace(0, 2.5, 100)
        x2_line = 1 - x1_vals  # x1 + x2 = 1
        
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x1 = 2 * np.cos(theta)
        circle_x2 = 2 * np.sin(theta)
        
        plt.plot(x1_vals, x2_line, 'r--', linewidth=2, label='x₁ + x₂ = 1')
        plt.plot(circle_x1, circle_x2, 'b--', linewidth=2, label='x₁² + x₂² = 4')
        
        # Trayectoria
        plt.plot(solutions[:, 0], solutions[:, 1], 'ko-', 
                markersize=8, linewidth=2, label='Trayectoria del método')
        plt.plot(solutions[0, 0], solutions[0, 1], 'go', 
                markersize=12, label='Punto inicial')
        plt.plot(solutions[-1, 0], solutions[-1, 1], 'ro', 
                markersize=12, label='Solución final')
        plt.plot(2, 1, 'b*', markersize=15, 
                label='Óptimo sin restricciones (2,1)')
        
        # Numeración de iteraciones
        for i, (x1, x2) in enumerate(solutions[::2]):  # Cada 2 puntos
            plt.annotate(f'{i*2+1}', (x1, x2), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
        plt.xlabel('x₁', fontsize=12)
        plt.ylabel('x₂', fontsize=12)
        plt.title('Trayectoria del Método de Penalización Interior', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 2.5)
        plt.ylim(0, 2.5)
        
        plt.tight_layout()
        plt.savefig('trayectoria_simple.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Gráfica simple guardada como 'trayectoria_simple.png'")
        
    except Exception as e:
        print(f"Error en gráfica simple: {e}")
    
    # Tabla de resumen de iteraciones
    print("\n" + "="*80)
    print("RESUMEN DE ITERACIONES")
    print("="*80)
    print(f"{'Iter':<4} {'μ':<12} {'x₁':<10} {'x₂':<10} {'f(x)':<12} {'Iter Int':<8}")
    print("-" * 80)
    for i, (mu, sol, inner_iter) in enumerate(zip(history['mu_values'], 
                                                  history['solutions'], 
                                                  history['inner_iterations'])):
        f_val = objective_function(sol)
        print(f"{i+1:<4} {mu:<12.6f} {sol[0]:<10.4f} {sol[1]:<10.4f} {f_val:<12.6f} {inner_iter:<8}")