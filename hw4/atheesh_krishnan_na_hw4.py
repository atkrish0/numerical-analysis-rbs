# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import yfinance as yf

from hw4_utility import *

# ===============================================================================================

# UTILITY FILE (hw4_utility.py)

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

# ===============================================================================================

"""# Q1"""

m1 = np.array([[4, 0, 0, 0],[6, 7, 0, 0],[9, 11, 1, 0],[5, 4, 1, 1]])
m2 = np.array([[2, 3, 1, 2],[-2, 4, -1, 5],[3, 7, 1.5, 1],[6, -9, 3, 7]])

for i, matrix in enumerate([m1, m2], start=1):
    print(f"Matrix {i}:")
    print(f"Symmetric:", check_symmetric(matrix))
    print(f"Singular:", check_singular(matrix))
    print(f"Positive Definite:", check_posdef(matrix))
    print("--------------------")

"""# Q2

### LU Factorization
"""

A = np.array([[4, 1, 1, 1],[1, 3, -1, 1],[1, -1, 2, 0],[1, 1, 0, 2]], dtype=float)

if lu_conditions(A):
    print("LU factorization conditions satisfied.")
    L, U = lu_factorization(A)
    print("\nMatrix A:")
    print(A)
    print("\nLower Triangular Matrix L:")
    print(L)
    print("\nUpper Triangular Matrix U:")
    print(U)
    print("\n(LU):")
    print(L @ U)
    if np.allclose(A, L @ U):
        print("\nLU factorization verified.")
    else:
        print("\nLU factorization failed.")
else:
    print("LU factorization not possibl.e")

"""### LDL^T Factorization"""

A = np.array([[4, 1, 1, 1],[1, 3, -1, 1],[1, -1, 2, 0],[1, 1, 0, 2]], dtype=float)

if ldlt_conditions(A):
    print("LDL^T conditions satisfied.")
    L, D = ldlt_factorization(A)
    print("\nMatrix A:")
    print(A)
    print("\nLower Triangular Matrix L:")
    print(L)
    print("\nDiagonal Matrix D:")
    print(D)
    print("\n(LDL^T):")
    print(L @ D @ L.T)
    if np.allclose(A, L @ D @ L.T):
        print("\nLDL^T factorization verified.")
    else:
        print("\nLDL^T factorization failed.")
else:
    print("LDL^T factorization not possible.")

"""### Cholesky Decomposition"""

A = np.array([[4, 1, 1, 1],[1, 3, -1, 1],[1, -1, 2, 0],[1, 1, 0, 2]], dtype=float)

if cholesky_conditions(A):
    print("Cholesky conditions satisfied.")
    L = cholesky_factorization(A)
    print("\nMatrix A:")
    print(A)
    print("\nLower Triangular Matrix L:")
    print(L)
    print("\nTranspose of Lower Triangular Matrix L.T:")
    print(L.T)
    print("\n(L @ L.T):")
    print(L @ L.T)
    if np.allclose(A, L @ L.T):
        print("\nCholesky factorization verified.")
    else:
        print("\nCholesky factorization failed.")
else:
    print("Cholesky factorization not possible.")

"""# Q3"""

A = np.array([[1, 1, 0, 3],[2, 1, -1, 1],[3, -1, -1, 2],[-1, 2, 3, -1]], dtype=float)
B = np.array([8, 7, 14, -7], dtype=float)
B_systems = [B.copy()]

for i in range(len(B)):
    B2 = B.copy()
    B2[i] *= 1.01
    B_systems.append(B2)

print("Solutions via LU FACTORIZATION:")
print("================")
for i, b in enumerate(B_systems, start=1):
    try:
        X = solve_system(A, b)
        print(f"SYSTEM {i}:")
        print(f"RHS Vector B:{b}")
        print(f"Solution X:{X}\n")
    except ValueError as e:
        print(f"SYSTEM {i}: {e}\n")

print("Solutions via GAUSSIAN ELIMINATION:")
print("======================================")
for i, b in enumerate(B_systems, start=1):
    X = gaussian_elimination(A, b)
    print(f"SYSTEM {i}:")
    print(f"RHS Vector B:{b}")
    print(f"Solution X:{X}\n")

n = A.shape[0]

decomposition_ops = (2 / 3) * n**3
substitution_ops = 10 * n**2
lu_ops = decomposition_ops + substitution_ops

gaussian_ops = (10 / 3) * n**3

print(f"LU Decomposition Total Operations: {lu_ops:.2f}")
print(f"Gaussian Elimination Total Operations: {gaussian_ops:.2f}\n")

if lu_ops < gaussian_ops:
    print("LU decomposition is more efficient for solving multiple systems with the same matrix A.")
else:
    print("Gaussian elimination is more efficient for solving single systems.")

"""# Q4"""

def get_stock_data(tickers, start_date, end_date):
    data = {ticker: yf.download(ticker, start=start_date, end=end_date)['Adj Close'] for ticker in tickers}
    df = pd.concat(data, axis=1, join="inner")
    df.columns = tickers
    df = df.dropna(how="any")
    return df

def analyze_stock_pair(main_ticker, pair_ticker, start_date, end_date):
    tickers = [main_ticker, pair_ticker]
    prices = get_stock_data(tickers, start_date, end_date)
    returns = prices.pct_change().dropna()
    cov_matrix = returns.cov()

    if cholesky_conditions(cov_matrix):
        L = cholesky_factorization(cov_matrix.to_numpy())
        return cov_matrix, True, L
    else:
        return cov_matrix, False, None

start_date = "2024-01-02"
end_date = "2024-11-29"

main_ticker = "JPM"
pair_tickers = ["BAC", "XOM", "AAPL", "CVX"]

for pair_ticker in pair_tickers:
    print(f"\nPair: {main_ticker} and {pair_ticker}\n")
    cov_matrix, cholesky_possible, L = analyze_stock_pair(main_ticker, pair_ticker, start_date, end_date)
    print(f"Covariance Matrix:\n{cov_matrix}\n")
    print(f"Cholesky Decomposition Possible: {cholesky_possible}\n")

    if L is not None:
        print(f"Cholesky Factorization (L):\n{L}")
        print(f"\nReconstructed Covariance Matrix (L * L^T):\n{L @ L.T}")
        print(f"\nOriginal Covariance Matrix:\n{cov_matrix}")
        if np.allclose(cov_matrix.to_numpy(), L @ L.T):
            print("\nVerification Passed.")
        else:
            print("\nVerification Failed.")
        break

"""# Q5"""

A = np.array([[-4, 6],[3, 8]], dtype=float)
U, S, Vt = svd(A)

print("Original Matrix A:")
print(A)
print("\nU (Left Singular Vectors):")
print(U)
print("\nS (Singular Values):")
print(S)
print("\nVt (Right Singular Vectors - Transposed):")
print(Vt)

print("\n(U * S * Vt):")
print(U @ S @ Vt)

if np.allclose(A, U @ S @ Vt):
    print("\nVerification Passed.")
else:
    print("\nVerification Failed.")

"""# Q6"""

A = np.array([[1, -1],[0,  1],[-1, 0]])

eigenvalues, eigenvectors, transformed_data = pca(A)

print("Original Matrix A:")
print(A)
print("\nEigenvalues:")
print(eigenvalues)
print("\nEigenvectors:")
print(eigenvectors)
print("\nPCA Extraction:")
print(transformed_data)