import numpy as np
import pandas as pd
import yfinance as yf

# ===============================================================================================

def check_square(matrix):
    return matrix.shape[0] == matrix.shape[1]

def check_symmetric(matrix):
    if not check_square(matrix):
        return False
    return np.array_equal(matrix, matrix.T)

def check_singular(matrix):
    if not check_square(matrix):
        return False
    det = np.linalg.det(matrix)
    return np.isclose(det, 0)

def check_posdef(matrix):
    if not check_symmetric(matrix):
        return False
    eigenvalues = np.linalg.eigvals(matrix)
    return all(eigenvalue > 0 for eigenvalue in eigenvalues)

# ===============================================================================================

def lu_conditions(matrix):
    if not check_square(matrix):
        return False
    if check_singular(matrix):
        return False
    return True

def lu_factorization(matrix):
    n = len(matrix)
    L = np.eye(n)
    U = matrix.copy()

    for i in range(n):
        for j in range(i + 1, n):
            k = U[j, i] / U[i, i]
            L[j, i] = k
            U[j, i:] = U[j, i:] - k * U[i, i:]

    return L, U

# ===============================================================================================

def ldlt_conditions(matrix):
    if not check_square(matrix):
        return False
    if not check_symmetric(matrix):
        return False
    if check_singular(matrix):
        return False
    return True

def ldlt_factorization(matrix):
    n = len(matrix)
    L = np.eye(n)
    D = np.zeros(n)

    for i in range(n):
        D[i] = matrix[i, i] - sum(L[i, k] ** 2 * D[k] for k in range(i))       
        for j in range(i + 1, n):
            L[j, i] = (matrix[j, i] - sum(L[j, k] * L[i, k] * D[k] for k in range(i))) / D[i]
    
    D_matrix = np.diag(D)
    return L, D_matrix

# ===============================================================================================

def cholesky_conditions(matrix):
    if not check_square(matrix):
        return False
    if not check_symmetric(matrix):
        return False
    if not check_posdef(matrix):
        return False
    return True

def cholesky_factorization(matrix):
    n = len(matrix)
    L = np.zeros_like(matrix)
    for i in range(n):
        L[i, i] = np.sqrt(matrix[i, i] - sum(L[i, k] ** 2 for k in range(i)))
        for j in range(i + 1, n):
            L[j, i] = (matrix[j, i] - sum(L[j, k] * L[i, k] for k in range(i))) / L[i, i]
    return L

# ===============================================================================================

# L * y = b
def forward_sub(L, b):
    n = len(b)
    y = np.zeros_like(b, dtype=float)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    return y

# U * x = y
def backward_sub(U, y):
    n = len(y)
    x = np.zeros_like(y, dtype=float)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    return x

# A * x = b via LU factorization
def solve_system(A, b):
    if not lu_conditions(A):
        raise ValueError("LU factorization not possible for the given matrix.")
    L, U = lu_factorization(A)
    y = forward_sub(L, b)
    x = backward_sub(U, y)
    return x

# A * x = b via Gaussian elimination
def gaussian_elimination(A, b):
    A = A.copy()
    b = b.copy()
    n = len(b)

    for i in range(n):
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]
    x = np.zeros_like(b, dtype=float)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
    return x

# ===============================================================================================

def svd(matrix):
    AtA = matrix.T @ matrix
    AAt = matrix @ matrix.T

    eigvals_AtA, V = np.linalg.eig(AtA)
    singular_values = np.sqrt(eigvals_AtA)
    eigvals_AAt, U_raw = np.linalg.eig(AAt)

    sorted_indices = np.argsort(singular_values)[::-1]
    singular_values = singular_values[sorted_indices]
    V = V[:, sorted_indices]
    U_raw = U_raw[:, sorted_indices]

    S = np.diag(singular_values)
    U = (matrix @ V) @ np.linalg.pinv(S)
    Vt = V.T

    return U, S, Vt

# ===============================================================================================

def pca(matrix):
    n = matrix.shape[0]
    mean_centered = matrix - np.mean(matrix, axis=0)
    cov_mat = (1 / n) * mean_centered.T @ mean_centered
    eigvals, eigvecs = np.linalg.eig(cov_mat)

    sorted_indices = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]

    transformed_data = mean_centered @ eigvecs

    return eigvals, eigvecs, transformed_data

