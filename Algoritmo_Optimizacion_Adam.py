import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Función objetivo: f(θ) = (θ1 - 2)^2 + (θ2 + 1)^2
def f(theta):
    return (theta[0] - 2)**2 + (theta[1] + 1)**2

# Gradiente de f
def grad_f(theta):
    return np.array([
        2 * (theta[0] - 2),
        2 * (theta[1] + 1)
    ])

"""
Algoritmo Adam con criterios de convergencia.

Parámetros:
    f: función objetivo
    grad_f: gradiente de f
    theta0: vector inicial (numpy array)
    alpha: tasa de aprendizaje
    beta1, beta2, epsilon: hiperparámetros Adam
    max_iteraciones: número máximo de iteraciones
    tol_grad: tolerancia para la norma del gradiente
    tol_f: tolerancia para el cambio en la función objetivo
    verbose: imprimir información de convergencia
"""

def adam(f, grad_f, theta0, alpha=0.01, beta1=0.9, beta2=0.999,
                           epsilon=1e-8, max_iteraciones=1000, 
                           tol_grad=1e-6, tol_f=1e-8, verbose=True):
    theta = theta0.copy()
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    historial = []
    
    f_prev = f(theta)
    
    for t in range(1, max_iteraciones + 1):
        g = grad_f(theta)
        
        # Criterio de convergencia 1: Gradiente muy pequeño
        grad_norm = np.linalg.norm(g)
        if grad_norm < tol_grad:
            if verbose:
                print(f"Convergencia alcanzada en iteración {t}: ||grad|| = {grad_norm:.2e} < {tol_grad}")
            break
        
        # Momentos
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        
        # Corrección de sesgo
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        # Actualización
        theta_prev = theta.copy()
        theta = theta - alpha * m_hat / (np.sqrt(v_hat) + epsilon)
        
        f_current = f(theta)
        
        # Criterio de convergencia 2: Cambio en función objetivo muy pequeño
        if t > 1 and abs(f_current - f_prev) < tol_f:
            if verbose:
                print(f"Convergencia alcanzada en iteración {t}: |Δf| = {abs(f_current - f_prev):.2e} < {tol_f}")
            break
        
        # Criterio de convergencia 3: Cambio en parámetros muy pequeño
        param_change = np.linalg.norm(theta - theta_prev)
        if param_change < tol_grad:
            if verbose:
                print(f"Convergencia alcanzada en iteración {t}: ||Δθ|| = {param_change:.2e} < {tol_grad}")
            break
        
        # Guardar resultados de esta iteración
        historial.append({
            "t": t,
            "theta": theta.copy(),
            "grad": g.copy(),
            "grad_norm": grad_norm,
            "m": m.copy(),
            "v": v.copy(),
            "m_hat": m_hat.copy(),
            "v_hat": v_hat.copy(),
            "f(theta)": f_current,
            "param_change": param_change
        })
        
        f_prev = f_current
    
    if t == max_iteraciones and verbose:
        print(f"Se alcanzó el máximo de iteraciones ({max_iteraciones}) sin convergencia completa")
    
    return historial, theta

# Mostrar solo las primeras y últimas iteraciones para cada prueba
def mostrar_resumen(historial, nombre):
    print(f"\n{nombre} - Resumen de iteraciones:")
    if not historial:
        print("  Sin iteraciones registradas")
        return
        
    print("Primeras 3 iteraciones:")
    for i in range(min(3, len(historial))):
        paso = historial[i]
        theta_str = f"[{paso['theta'][0]:.4f}, {paso['theta'][1]:.4f}]"
        print(f"  Iter {paso['t']}: θ={theta_str}, f(θ)={paso['f(theta)']:.6f}, ||grad||={paso['grad_norm']:.2e}")
    
    if len(historial) > 6:
        print("  ...")
        print("Últimas 3 iteraciones:")
        for i in range(max(0, len(historial)-3), len(historial)):
            paso = historial[i]
            theta_str = f"[{paso['theta'][0]:.4f}, {paso['theta'][1]:.4f}]"
            print(f"  Iter {paso['t']}: θ={theta_str}, f(θ)={paso['f(theta)']:.6f}, ||grad||={paso['grad_norm']:.2e}")

# Prueba con diferentes configuraciones
print("="*60)
print("PRUEBA 1: alpha=0.1, beta1=0.9, beta2=0.999")
print("="*60)

theta0 = np.array([0.0, 0.0])
historial1, theta_final1 = adam(
    f, grad_f, theta0, 
    alpha=0.1,
    max_iteraciones=250,
    verbose=True
)

print(f"Resultado final: θ = {theta_final1}")
print(f"f(θ) = {f(theta_final1):.8f}")
print(f"Iteraciones realizadas: {len(historial1)}")

print("\n" + "="*60)
print("PRUEBA 2: alpha=0.05, beta1=0.9, beta2=0.999")
print("="*60)

historial2, theta_final2 = adam(
    f, grad_f, theta0,
    alpha=0.05,
    max_iteraciones=250,
    verbose=True
)

print(f"Resultado final: θ = {theta_final2}")
print(f"f(θ) = {f(theta_final2):.8f}")
print(f"Iteraciones realizadas: {len(historial2)}")

print("\n" + "="*60)
print("PRUEBA 3: alpha=0.02, beta1=0.9, beta2=0.999")
print("="*60)

historial3, theta_final3 = adam(
    f, grad_f, theta0,
    alpha=0.02,
    tol_grad=1e-8,
    tol_f=1e-10,
    max_iteraciones=250,
    verbose=True
)

print(f"Resultado final: θ = {theta_final3}")
print(f"f(θ) = {f(theta_final3):.8f}")
print(f"Iteraciones realizadas: {len(historial3)}")

print("\n" + "="*60)
print("PRUEBA 4: alpha=0.001, beta1=0.9, beta2=0.999")
print("="*60)

historial4, theta_final4 = adam(
    f, grad_f, theta0,
    alpha=0.001,
    max_iteraciones=250,
    verbose=True
)

print(f"Resultado final: θ = {theta_final4}")
print(f"f(θ) = {f(theta_final4):.8f}")
print(f"Iteraciones realizadas: {len(historial4)}")

# Mostrar resúmenes
mostrar_resumen(historial1, "PRUEBA 1 (α=0.1)")
mostrar_resumen(historial2, "PRUEBA 2 (α=0.05)")
mostrar_resumen(historial3, "PRUEBA 3 (α=0.02)")
mostrar_resumen(historial4, "PRUEBA 4 (α=0.001)")

# Análisis del problema
print("\n" + "="*60)
print("ANÁLISIS COMPARATIVO")
print("="*60)
print("SOLUCIÓN ÓPTIMA TEÓRICA: θ* = [2, -1], f(θ*) = 0")
print("\nRESULTADOS:")
print(f"   Prueba 1 (α=0.1):   {len(historial1):3d} iters → θ={theta_final1} → f={f(theta_final1):.2e}")
print(f"   Prueba 2 (α=0.05):  {len(historial2):3d} iters → θ={theta_final2} → f={f(theta_final2):.2e}")  
print(f"   Prueba 3 (α=0.02):  {len(historial3):3d} iters → θ={theta_final3} → f={f(theta_final3):.2e}")
print(f"   Prueba 4 (α=0.001): {len(historial4):3d} iters → θ={theta_final4} → f={f(theta_final4):.2e}")

# Crear gráfica comparativa
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
if historial1:
    plt.semilogy([paso['f(theta)'] for paso in historial1], 'r-', label='α=0.1', linewidth=2)
if historial2:
    plt.semilogy([paso['f(theta)'] for paso in historial2], 'b-', label='α=0.05', linewidth=2)
if historial3:
    plt.semilogy([paso['f(theta)'] for paso in historial3], 'g-', label='α=0.02', linewidth=2)
if historial4:
    plt.semilogy([paso['f(theta)'] for paso in historial4], 'm-', label='α=0.001', linewidth=2)
plt.xlabel('Iteración')
plt.ylabel('f(θ) (escala log)')
plt.title('Convergencia de la Función Objetivo')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
if historial1:
    plt.semilogy([paso['grad_norm'] for paso in historial1], 'r-', label='α=0.1', linewidth=2)
if historial2:
    plt.semilogy([paso['grad_norm'] for paso in historial2], 'b-', label='α=0.05', linewidth=2)
if historial3:
    plt.semilogy([paso['grad_norm'] for paso in historial3], 'g-', label='α=0.02', linewidth=2)
if historial4:
    plt.semilogy([paso['grad_norm'] for paso in historial4], 'm-', label='α=0.001', linewidth=2)
plt.xlabel('Iteración')
plt.ylabel('||∇f|| (escala log)')
plt.title('Convergencia del Gradiente')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
# Mostrar trayectorias en el espacio de parámetros
if historial1:
    thetas1 = np.array([paso['theta'] for paso in historial1])
    plt.plot(thetas1[:, 0], thetas1[:, 1], 'r-o', markersize=2, label='α=0.1', linewidth=2)
if historial2:
    thetas2 = np.array([paso['theta'] for paso in historial2])
    plt.plot(thetas2[:, 0], thetas2[:, 1], 'b-o', markersize=2, label='α=0.05', linewidth=2)
if historial3:
    thetas3 = np.array([paso['theta'] for paso in historial3])
    plt.plot(thetas3[:, 0], thetas3[:, 1], 'g-o', markersize=2, label='α=0.02', linewidth=2)

# Mostrar punto inicial y óptimo
plt.plot(0, 0, 'ko', markersize=8, label='Inicio [0,0]')
plt.plot(2, -1, 'r*', markersize=15, label='Óptimo [2,-1]')
plt.xlabel('θ₁')
plt.ylabel('θ₂')
plt.title('Trayectorias en el Espacio de Parámetros')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('adam_convergencia.png', dpi=300, bbox_inches='tight')
print("\nGráfica guardada como 'adam_convergencia.png'")
plt.close() 