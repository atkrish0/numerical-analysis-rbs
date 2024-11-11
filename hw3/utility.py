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