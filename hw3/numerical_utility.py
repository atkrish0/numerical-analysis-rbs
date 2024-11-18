import numpy as np

def base_points(n, a, b):
    return [a + ((b - a) * i / n) for i in range(n + 1)]


def chebyshev_points(n, a, b):
    return [((a + b) / 2) + ((b - a) / 2)* np.cos((2 * i + 1) * np.pi / (2 * (n + 1))) for i in range(n + 1)]


def cubic_spline_interpolation(x, y):
    n = len(x) - 1
    h = np.diff(x)

    A = np.zeros((n + 1, n + 1))
    B = np.zeros(n + 1)

    A[0, 0] = 1
    A[n, n] = 1

    for i in range(1, n):
        A[i, i-1] = h[i-1]
        A[i, i] = 2 * (h[i-1] + h[i])
        A[i, i+1] = h[i]
        B[i] = 3 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])
    
    c = np.linalg.solve(A, B)

    b = [(y[i+1] - y[i]) / h[i] - h[i] * (2*c[i] + c[i+1]) / 3 for i in range(n)]
    d = [(c[i+1] - c[i]) / (3*h[i]) for i in range(n)]
    a = y[:-1]
    
    return a, b, c[:-1], d


def evaluate_spline(x_val, x, a, b, c, d):
    for i in range(len(x) - 1):
        if x[i] <= x_val <= x[i + 1]:
            dx = x_val - x[i]
            return a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3
        

def clamped_cubic_spline(x, y, der_a, der_b):
    n = len(x) - 1
    h = np.diff(x)

    A = np.zeros((n + 1, n + 1))
    B = np.zeros(n + 1)

    A[0, 0] = 2 * h[0]
    A[0, 1] = h[0]
    B[0] = 3 * ((y[1] - y[0]) / h[0] - der_a)

    A[n, n - 1] = h[-1]
    A[n, n] = 2 * h[-1]
    B[n] = 3 * (der_b - (y[n] - y[n - 1]) / h[-1])

    a, b, c, d = cubic_spline_interpolation(x, y)
    
    return a, b, c, d


def lagrange_interpolation(xy_points, x, n):
    sorted_points = sorted(xy_points,key=lambda x: x[0])
    result = 0.0
    for i in range(n + 1):
        z = sorted_points[i][1]
        for j in range(n + 1):
            if j!= i:
                z *= (x - sorted_points[j][0]) / (sorted_points[i][0] - sorted_points[j][0])
        result += z
    return result

def divided_differences(x, y):
    n = len(x)
    coef = np.array(y, float)

    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coef[i] = (coef[i] - coef[i - 1]) / (x[i] - x[i - j])
    return coef


def newton_interpolation(x_data, y_data, x):
    coef = divided_differences(x_data, y_data)  
    n = len(coef)
    result = coef[-1]  

    for i in range(n - 2, -1, -1):
        result = result * (x - x_data[i]) + coef[i]
    return result


def hermite_interpolation(x_values, y_values, y_derivatives):
    n = len(x_values)
    z = np.zeros(2 * n)
    Q = np.zeros((2 * n, 2 * n))

    for i in range(n):
        z[2 * i] = x_values[i]
        z[2 * i + 1] = x_values[i]
        Q[2 * i, 0] = y_values[i]
        Q[2 * i + 1, 0] = y_values[i]
        Q[2 * i + 1, 1] = y_derivatives[i]
        if i != 0:
            Q[2 * i, 1] = (Q[2 * i, 0] - Q[2 * i - 1, 0]) / (z[2 * i] - z[2 * i - 1])

    for j in range(2, 2 * n):
        for i in range(j, 2 * n):
            Q[i, j] = (Q[i, j - 1] - Q[i - 1, j - 1]) / (z[i] - z[i - j])

    return z, Q[0]


def evaluate_hermite(z, coef, x):
    n = len(coef)
    result = coef[0]
    product_term = 1
    for i in range(1, n):
        product_term *= (x - z[i - 1])
        result += coef[i] * product_term
    return result

