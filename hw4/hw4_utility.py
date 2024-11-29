import numpy as np

def check_square(matrix):
    return matrix.shape[0] == matrix.shape[1]

def check_symmetric(matrix):
    if not check_square(matrix):
        return False
    return np.array_equal(matrix, matrix.T)

def check_singular(matrix):
    det = np.linalg.det(matrix)
    return np.isclose(det, 0)

def check_posdef(matrix):
    if not check_symmetric(matrix):
        return False
    eigenvalues = np.linalg.eigvals(matrix)
    return all(eigenvalue > 0 for eigenvalue in eigenvalues)

# ===============================================================================================

def check_matrix(matrix):
    det = np.linalg.det(matrix)
    check_symmetric = np.array_equal(matrix, matrix.T)
    check_singular = np.isclose(det, 0)
    try:
        eigenvalues = np.linalg.eigvals(matrix)
        check_posdef = np.all(eigenvalues > 0)
    except np.linalg.LinAlgError:
        check_posdef = False
        
    return check_symmetric, check_singular, check_posdef

def check_matrix2(matrix):
    if (matrix == matrix.T).all():
        check_symmetric = True
    else:
        check_symmetric = False

    det = np.linalg.det(matrix)
    if det == 0:
        check_singular = True
    else:
        check_singular = False

    try:
        eigenvalues = np.linalg.eigvals(matrix)
        check_posdef = True
        for eigenvalue in eigenvalues:
            if eigenvalue <= 0:
                check_posdef = False
                break
    except np.linalg.LinAlgError:  
        check_posdef = False

    return check_symmetric, check_singular, check_posdef

# ===============================================================================================

def lu_conditions():
    return None

def lu_factorization(A):
    n = len(A)
    L = np.eye(n)  # Initialize L as the identity matrix
    U = A.copy()   # Start with U as a copy of A

    for i in range(n):
        # For each column, eliminate elements below the pivot
        for j in range(i + 1, n):
            # Multiplier for elimination
            multiplier = U[j, i] / U[i, i]
            L[j, i] = multiplier  # Store multiplier in L
            # Update the row in U
            U[j, i:] = U[j, i:] - multiplier * U[i, i:]

    return L, U

def ldlt_conditions():
    return None

def ldlt_factorization(A):
    n = A.shape[0]
    
    # Check if the matrix is symmetric
    if not np.allclose(A, A.T):
        raise ValueError("Matrix is not symmetric. LDLᵀ factorization is not possible.")
    
    L = np.eye(n)  # Initialize L as the identity matrix
    D = np.zeros((n, n))  # Initialize D as a zero matrix
    
    for i in range(n):
        # Compute D[i, i]
        D[i, i] = A[i, i] - sum(L[i, k] ** 2 * D[k, k] for k in range(i))
        
        for j in range(i + 1, n):
            # Compute L[j, i]
            L[j, i] = (A[j, i] - sum(L[j, k] * L[i, k] * D[k, k] for k in range(i))) / D[i, i]
    
    return L, D

def cholesky_conditions():
    return None

def cholesky_factorization(A):
    return None

# ===============================================================================================



