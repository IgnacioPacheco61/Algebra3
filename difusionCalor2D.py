# TP Simulación Física en Videojuegos: Difusión de Calor 2D

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.sparse import diags, identity, kron
from scipy.sparse.linalg import spsolve

# Construcción matriz A para método implícito
def construir_matriz(nx, ny, dx, dy, dt, alpha):
    Nix, Niy = nx - 2, ny - 2
    Ix = identity(Nix)
    Iy = identity(Niy)

    main_diag_x = 2 * (1/dx*2 + 1/dy*2) * np.ones(Nix)
    off_diag_x = -1/dx**2 * np.ones(Nix - 1)
    Tx = diags([off_diag_x, main_diag_x, off_diag_x], [-1, 0, 1])

    off_diag_y = -1/dy**2 * np.ones(Niy - 1)
    Ty = diags([off_diag_y, off_diag_y], [-1, 1], shape=(Niy, Niy))

    L = kron(Iy, Tx) + kron(Ty, Ix)
    A = identity(Nix*Niy) - dt * alpha * L
    return A

# Inicializar temperatura y condiciones de frontera
def inicializar_T(nx, ny):
    T = np.ones((ny, nx)) * 25
    T[:, 0] = 100    # borde izquierdo
    T[:, -1] = 50    # borde derecho
    T[0, :] = 0      # borde superior
    T[-1, :] = 75    # borde inferior
    # Fuente interna caliente
    T[ny//2 - 1:ny//2 + 2, nx//2 - 1:nx//2 + 2] = 200
    return T

# TODO: Métodos a desarrollar
def optimizado(A, b):
    """Resolución de un sistema de ecuaciones lineales optimizado
        cuando la matriz A es tridiagonal.
    """
    A = A.toarray()
    n = len(b)
    a = np.zeros(n) #subdiagonal
    b_diag = np.zeros(n) #diagonal
    c = np.zeros(n) #superdiagonal

    #Extraer diagonales
    for i in range(n):
        b_diag[i] = A[i,i]
        if i > 0:
            a[i] = A[i, i - 1]
        if i < n - 1:
            c[i] = A[i, i + 1]
    
    #Forwad
    for i in range(1, n):
        m = a[i] /b_diag[i - 1]
        b_diag -= m * c[i - 1]
        b[i] -= m * b[i - 1]
    
    #Backward
    x = np.zeros(n)
    x[-1] = b[-1] / b_diag[-1]
    for i in range( n - 2 , -1, -1):
        x[i] = (b[i] - c[i] * x[i + 1]) / b_diag[i]
    
    return x

def gauss_pivoteo(A, b):
    """Método de Gauss con pivoteo parcial"""
    A = A.toarray().copy()
    b = b.copy()
    n = len(b)

    #Eliminacion hacia adelante
    for i in range(n):
        #Pivoteo parcial
        max_row = np.argmax(np.abs(A[i:, i])) + i
        if A[max_row, i] == 0:
            raise ValueError("Matriz singular")
        if max_row != i:
            A[[i, max_row]] = A[[max_row, i]]
            b[[i, max_row]] = b[[max_row, i]]
        
        for j in range(i + 1, n):
            m = A[j, i] / A[i, i]       
            A[j, i:] -= m * A[i, i:]
            b[j] -= m * b[i]

        
    #Sustitucion hacia atras
    x = np.zeros_like(b)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i,i]

    return x

# Un paso de simulación con el método implícito y método de solución
def paso_simulacion(T, A, nx, ny, dx, dy, dt, alpha, metodo_solucion):
    b = T[1:-1, 1:-1].copy()
    # Incorporar condiciones de borde en b
    b[:, 0] += dt * alpha * T[1:-1, 0] / dx**2
    b[:, -1] += dt * alpha * T[1:-1, -1] / dx**2
    b[0, :] += dt * alpha * T[0, 1:-1] / dy**2
    b[-1, :] += dt * alpha * T[-1, 1:-1] / dy**2
    b = b.flatten()

    if metodo_solucion == 'directo':
        T_vec = spsolve(A, b)
    elif metodo_solucion == 'optimizado':
        T_vec = optimizado(A, b)
    elif metodo_solucion == 'gauss_pivoteo':
        T_vec  = gauss_pivoteo(A, b)
    else:
        raise ValueError("Método de solución no reconocido")

    T_new = T.copy()
    T_new[1:-1, 1:-1] = T_vec.reshape((ny - 2, nx - 2))
    # Mantener la fuente de calor interna fija
    T_new[ny//2 - 1:ny//2 + 2, nx//2 - 1:nx//2 + 2] = 200
    return T_new

# Simular múltiples pasos, medir tiempos
def simular(nx, ny, dt, alpha, pasos, metodo_solucion):
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    A = construir_matriz(nx, ny, dx, dy, dt, alpha)
    T = inicializar_T(nx, ny)

    tiempos = []
    for _ in range(pasos):
        start = time.time()
        T = paso_simulacion(T, A, nx, ny, dx, dy, dt, alpha, metodo_solucion)
        end = time.time()
        tiempos.append(end - start)

    tiempo_promedio = np.mean(tiempos)
    return T, tiempo_promedio

# Error RMS relativo
def error_rms(T_ref, T):
    return np.sqrt(np.mean((T_ref - T)*2)) / np.sqrt(np.mean(T_ref*2))

# --------- Experimentos ---------

resoluciones = [20, 30, 50, 70, 100, 500]
dt = 0.1
alpha = 0.01
pasos = 10

# Guardar resultados para graficar
res_list = []
tiempos_dict = {"directo": [], "optimizado": [], "gauss_pivoteo": []}
errores_dict = {"directo": [], "optimizado": [], "gauss_pivoteo": []}

for n in resoluciones:
    print(f"\nResolución {n}x{n}")
    res_list.append(n)
    # Referencia con spsolve
    T_ref, t_ref = simular(n, n, dt, alpha, pasos, 'directo')
    tiempos_dict["directo"].append(t_ref)
    errores_dict["directo"].append(0)  # Error 0 contra sí mismo

    # Método optimizado (Thomas)
    try:
        T_opt, t_opt = simular(n, n, dt, alpha, pasos, 'optimizado')
        err_opt = error_rms(T_ref, T_opt)
        tiempos_dict["optimizado"].append(t_opt)
        errores_dict["optimizado"].append(err_opt)
        print(f"Optimizado: {t_opt:.4f} s - Error RMS: {err_opt:.2e}")
    except Exception as e:
        print(f"Error en optimizado para {n}x{n}: {e}")
        tiempos_dict["optimizado"].append(np.nan)
        errores_dict["optimizado"].append(np.nan)

    # Método Gauss solo para 20x20
    if n == 20:
        try:
            T_gauss, t_gauss = simular(n, n, dt, alpha, pasos, 'gauss_pivoteo')
            err_gauss = error_rms(T_ref, T_gauss)
            tiempos_dict["gauss_pivoteo"].append(t_gauss)
            errores_dict["gauss_pivoteo"].append(err_gauss)
            print(f"Gauss: {t_gauss:.4f} s - Error RMS: {err_gauss:.2e}")
        except Exception as e:
            print(f"Error en gauss para {n}x{n}: {e}")
            tiempos_dict["gauss_pivoteo"].append(np.nan)
            errores_dict["gauss_pivoteo"].append(np.nan)
    else:
        tiempos_dict["gauss_pivoteo"].append(np.nan)
        errores_dict["gauss_pivoteo"].append(np.nan)

# --- Gráficos ---

plt.figure(figsize=(10,5))
plt.plot(res_list, tiempos_dict["directo"], 'o-', label='spsolve (directo)')
plt.plot(res_list, tiempos_dict["optimizado"], 's-', label='Optimizado (Thomas)')
plt.plot(res_list, tiempos_dict["gauss_pivoteo"], 'd-', label='Gauss pivoteo parcial')
plt.yscale('log')
plt.xlabel('Tamaño de la cuadrícula (n x n)')
plt.ylabel('Tiempo promedio por paso [s] (escala log)')
plt.title('Tiempo de ejecución promedio por método')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(res_list, errores_dict["optimizado"], 's-', label='Optimizado (Thomas)')
plt.plot(res_list, errores_dict["gauss_pivoteo"], 'd-', label='Gauss pivoteo parcial')
plt.xlabel('Tamaño de la cuadrícula (n x n)')
plt.ylabel('Error RMS relativo respecto a spsolve')
plt.title('Error numérico vs tamaño del sistema')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()