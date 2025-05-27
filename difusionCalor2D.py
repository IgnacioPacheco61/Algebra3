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
    main_diag_x = 2 * (1/dx**2 + 1/dy**2) * np.ones(Nix)
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
    T[:, 0] = 100
    T[:, -1] = 50
    T[0, :] = 0
    T[-1, :] = 75
    T[ny//2 - 1:ny//2 + 2, nx//2 - 1:nx//2 + 2] = 200
    return T

# TODO: Métodos a desarrollar
def optimizado(A, b):
    n = len(b)
    a = np.zeros(n)
    c = np.zeros(n)
    d = b.copy()
    b_diag = np.zeros(n)
    for i in range(n):
        b_diag[i] = A[i, i]
        if i > 0:
            a[i] = A[i, i - 1]
        if i < n - 1:
            c[i] = A[i, i + 1]
    for i in range(1, n):
        m = a[i] / b_diag[i - 1]
        b_diag[i] -= m * c[i - 1]
        d[i] -= m * d[i - 1]
    x = np.zeros(n)
    x[-1] = d[-1] / b_diag[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b_diag[i]
    return x

def gauss_pivoteo(A, b):
    A = A.toarray().copy()
    b = b.copy()
    n = len(b)
    for i in range(n):
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
    x = np.zeros_like(b)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i,i]
    return x

# Un paso de simulación con el método implícito y método de solución
def paso_simulacion(T, A, nx, ny, dx, dy, dt, alpha, metodo_solucion):
    b = T[1:-1, 1:-1].copy()
    b[:, 0] += dt * alpha * T[1:-1, 0] / dx**2
    b[:, -1] += dt * alpha * T[1:-1, -1] / dx**2
    b[0, :] += dt * alpha * T[0, 1:-1] / dy**2
    b[-1, :] += dt * alpha * T[-1, 1:-1] / dy**2
    b = b.flatten()
    if metodo_solucion == 'directo':
        T_vec = spsolve(A, b)
    elif metodo_solucion == 'gauss_pivoteo':
        T_vec = gauss_pivoteo(A, b)
    elif metodo_solucion == 'optimizado':
        T_vec = optimizado(A, b)
    else:
        raise ValueError("Método de solución no reconocido")
    T_new = T.copy()
    T_new[1:-1, 1:-1] = T_vec.reshape((ny - 2, nx - 2))
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
    return T, np.mean(tiempos)

# Error RMS relativo
def error_rms(T_ref, T):
    return np.sqrt(np.mean((T - T_ref)**2)) / np.sqrt(np.mean(T_ref**2))



# --------- Experimentos combinados ---------

resoluciones_bajas = [20, 30, 50]
resoluciones_altas = [70, 100, 500, 1000]
dt = 0.1
alpha = 0.01
pasos = 10

tiempos_gauss = []
errores_gauss = []

tiempos_opt = []
errores_opt = []

tiempos_directo = []

print("Ejecutando métodos para resoluciones bajas (Gauss + Optimizado + Directo)\n")
for res in resoluciones_bajas:
    nx = ny = res
    print(f"Resolución {res}x{res}")
    T_ref, t_ref = simular(nx, ny, dt, alpha, pasos, metodo_solucion='directo')
    tiempos_directo.append(t_ref)

    try:
        T_gauss, t_gauss = simular(nx, ny, dt, alpha, pasos, metodo_solucion='gauss_pivoteo')
        err_gauss = error_rms(T_ref, T_gauss)
        tiempos_gauss.append(t_gauss)
        errores_gauss.append(err_gauss)
        print(f"  Gauss: {t_gauss:.4f} s - Error RMS: {err_gauss:.2e}")
    except Exception as e:
        print(f"  Gauss ERROR: {e}")
        tiempos_gauss.append(None)
        errores_gauss.append(None)

    try:
        T_opt, t_opt = simular(nx, ny, dt, alpha, pasos, metodo_solucion='optimizado')
        err_opt = error_rms(T_ref, T_opt)
        tiempos_opt.append(t_opt)
        errores_opt.append(err_opt)
        print(f"  Optimizado: {t_opt:.4f} s - Error RMS: {err_opt:.2e}")
    except Exception as e:
        print(f"  Optimizado ERROR: {e}")
        tiempos_opt.append(None)
        errores_opt.append(None)

print("\nSolo spsolve para resoluciones altas (por eficiencia):\n")
for res in resoluciones_altas:
    nx = ny = res
    print(f"Resolución {res}x{res}")
    try:
        T_ref, t_ref = simular(nx, ny, dt, alpha, pasos, metodo_solucion='directo')
        tiempos_directo.append(t_ref)
        print(f"  spsolve: {t_ref:.4f} s")
    except Exception as e:
        print(f"  Directo ERROR: {e}")
        tiempos_directo.append(None)

# Concatenar listas
resoluciones_totales = resoluciones_bajas + resoluciones_altas

# --- Graficar ---

def filtrar(l):
    return [x if x is not None else np.nan for x in l]

plt.figure(figsize=(10,5))
plt.plot(resoluciones_totales, filtrar(tiempos_directo), 'o-', label='spsolve (directo)')
plt.plot(resoluciones_bajas, filtrar(tiempos_opt), 's-', label='Optimizado (Thomas)')
plt.plot(resoluciones_bajas, filtrar(tiempos_gauss), 'd-', label='Gauss pivoteo parcial')
plt.yscale('log')
plt.xlabel('Tamaño de la cuadrícula (n x n)')
plt.ylabel('Tiempo promedio por paso [s] (escala log)')
plt.title('Tiempo de ejecución promedio por método')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(resoluciones_bajas, filtrar(errores_opt), 's-', label='Optimizado (Thomas)')
plt.plot(resoluciones_bajas, filtrar(errores_gauss), 'd-', label='Gauss pivoteo parcial')
plt.xlabel('Tamaño de la cuadrícula (n x n)')
plt.ylabel('Error RMS respecto a spsolve')
plt.title('Error numérico vs tamaño del sistema')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
