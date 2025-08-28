# pytest -v

import numpy as np
import pytest
import scipy.sparse as sp
from MatLite import Matrix

def test_constructor_from_list():
    A = Matrix([[1, 2], [3, 4]])
    assert A.shape == (2, 2)
    assert np.allclose(A.data, np.array([[1, 2], [3, 4]]))

def test_constructor_from_ndarray():
    arr = np.array([1, 2, 3])
    A = Matrix(arr)
    assert A.shape == (1, 3)

def test_addition_and_subtraction():
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[5, 6], [7, 8]])
    C = A + B
    D = B - A
    assert np.allclose(C.data, np.array([[6, 8], [10, 12]]))
    assert np.allclose(D.data, np.array([[4, 4], [4, 4]]))

def test_scalar_multiplication_and_division():
    A = Matrix([[1, 2], [3, 4]])
    B = A * 2
    C = A / 2
    assert np.allclose(B.data, np.array([[2, 4], [6, 8]]))
    assert np.allclose(C.data, np.array([[0.5, 1], [1.5, 2]]))

def test_matrix_multiplication():
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[2], [1]])
    C = A * B
    assert np.allclose(C.data, np.array([[4], [10]]))

def test_power_and_matpower():
    A = Matrix([[2, 0], [0, 2]])
    Apow = A ** 2
    Amatpow = A ^ 3
    assert np.allclose(Apow.data, np.array([[4, 0], [0, 4]]))
    assert np.allclose(Amatpow.data, np.array([[8, 0], [0, 8]]))

def test_transpose_and_Hermitian():
    A = Matrix([[1, 2j], [3, 4]])
    assert np.allclose(A.T.data, np.array([[1, 3], [2j, 4]]))
    assert np.allclose(A.H.data, np.array([[1, 3], [-2j, 4]]))

def test_abs_and_norm():
    A = Matrix([[3, -4]])
    assert np.allclose(A.abs().data, np.array([[3, 4]]))
    assert np.isclose(Matrix.norm(A), 5.0)

def test_trace_and_det():
    A = Matrix([[1, 2], [3, 4]])
    assert np.isclose(A.trace(), 5)
    assert np.isclose(A.det(), -2)

def test_rank():
    A = Matrix([[1, 2], [2, 4]])
    B = Matrix([[1, 0], [0, 1]])
    assert A.rank == 1
    assert B.rank == 2

def test_max_min_sum():
    A = Matrix([[1, 2], [3, 4]])
    assert np.allclose(A.max(0).data, np.array([[3, 4]]))
    assert np.allclose(A.min(0).data, np.array([[1, 2]]))
    assert np.allclose(A.sum(0).data, np.array([[4, 6]]))

def test_diag():
    v = Matrix([1, 2, 3])
    D = Matrix.diag(v)
    assert np.allclose(D.data, np.diag([1, 2, 3]))
    A = Matrix([[1, 2], [3, 4]])
    d = Matrix.diag(A)
    assert np.allclose(d.data.ravel(), np.array([1, 4]))

def test_static_zeros_ones_eye():
    A = Matrix.zeros(2, 3)
    B = Matrix.ones(2, 2)
    C = Matrix.eye(3)
    assert np.all(A.data == 0)
    assert np.all(B.data == 1)
    assert np.allclose(C.data, np.eye(3))

def test_real_imag():
    A = Matrix([[1+2j, 3-4j]])
    assert np.allclose(A.real.data, np.array([[1, 3]]))
    assert np.allclose(A.imag.data, np.array([[2, -4]]))

def test_backslash():
    A = Matrix([[3., 2.], [1., 2.]])
    B = Matrix([[5.], [5.]])
    x = A.backslash(B)
    assert np.allclose(A.data @ x.data, B.data)

def test_is_cow_active():
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix(A)  # partage données
    assert B.is_cow_active()["is_view"]

def test_sparse_constructor_and_shape():
    A_sparse = sp.csr_matrix([[1, 0], [0, 2]])
    A = Matrix(A_sparse)
    assert A.shape == (2, 2)
    assert sp.issparse(A.data)

def test_sparse_eye_and_zeros():
    A = Matrix.eye(3, sparse=True)
    B = Matrix.zeros(3, sparse=True)
    assert sp.issparse(A.data)
    assert sp.issparse(B.data)
    assert np.allclose(A.data.toarray(), np.eye(3))
    assert np.all(B.data.toarray() == 0)

def test_sparse_addition_and_subtraction():
    A = Matrix.eye(2, sparse=True)
    B = Matrix.ones(2, sparse=True)
    C = A + B
    D = B - A
    assert sp.issparse(C.data)
    assert np.allclose(C.data.toarray(), np.eye(2) + np.ones((2,2)))
    assert np.allclose(D.data.toarray(), np.ones((2,2)) - np.eye(2))

def test_sparse_scalar_multiplication_and_division():
    A = Matrix.eye(2, sparse=True)
    B = A * 2
    C = A / 2
    assert np.allclose(B.data.toarray(), 2*np.eye(2))
    assert np.allclose(C.data.toarray(), 0.5*np.eye(2))

def test_sparse_abs_and_norm():
    A = Matrix(sp.csr_matrix([[-3, 0], [0, 4]]))
    AbsA = abs(A)
    assert np.allclose(AbsA.data.toarray(), np.array([[3, 0], [0, 4]]))
    assert np.isclose(Matrix.norm(A, 'fro'), 5.0)

def test_sparse_diag_extraction_and_construction():
    v = Matrix([1, 2, 3])
    D = Matrix.diag(v, k=0)
    assert sp.issparse(D.data) or isinstance(D.data, np.ndarray)

    A = Matrix(sp.csr_matrix([[1, 2], [3, 4]]))
    d = Matrix.diag(A)
    assert np.allclose(d.data.ravel(), np.array([1, 4]))

def test_sparse_transpose_and_Hermitian():
    A = Matrix(sp.csr_matrix([[1, 2], [3, 4]]))
    assert np.allclose(A.T.data.toarray(), np.array([[1, 3], [2, 4]]))
    B = Matrix(sp.csr_matrix([[1, 2j], [3, 4]]))
    H = B.H
    assert np.allclose(H.data.toarray(), np.array([[1, 3], [-2j, 4]]))

def test_sparse_trace_and_rank():
    A = Matrix(sp.csr_matrix([[1, 0], [0, 2]]))
    assert np.isclose(A.trace(), 3)
    assert A.rank == 2

def test_sparse_backslash():
    A = Matrix(sp.csr_matrix([[3., 2.], [1., 2.]]))
    B = Matrix([[5.], [5.]])
    x = A.backslash(B)
    assert np.allclose(A.data @ x.data[0], B.data)
    
def test_sparse_sum_max_min():
    A = Matrix(sp.csr_matrix([[1, 2], [3, 4]]))
    assert np.allclose(A.sum(0).data, np.array([[4, 6]]))
    assert np.allclose(A.max(0).data, np.array([[3, 4]]))
    assert np.allclose(A.min(1).data, np.array([[1], [3]]))

def test_large_dense_operations():
    """Test sur des matrices denses 1000x1000"""
    n = 1000
    A = Matrix.rand(n, n, random_state=42)
    B = Matrix.rand(n, n, random_state=24)
    
    # Vérifie que la multiplication conserve la taille
    C = A * B
    assert C.shape == (n, n)

    # Norme de Frobenius positive
    norm_val = Matrix.norm(C, 'fro')
    assert norm_val > 0

def test_large_sparse_operations():
    """Test sur des matrices creuses 5000x5000"""
    n = 5000
    A = Matrix.rand(n, n, sparse=True, density=0.001, random_state=42)
    B = Matrix.eye(n, sparse=True)

    # Produit avec identité
    C = A * B
    assert sp.issparse(C.data)
    assert C.shape == (n, n)

    # Trace = somme des diagonales
    tr = A.trace()
    assert isinstance(tr, float)

def test_large_backslash():
    """Test de résolution Ax=b sur grande matrice SPD"""
    n = 200
    # Matrice symétrique définie positive
    A = Matrix.rand(n, n, random_state=42)
    A = A + A.H + n * Matrix.eye(n)
    b = Matrix.rand(n, 1, random_state=24)

    x = A.backslash(b)
    # Vérifie que Ax ≈ b
    residu = Matrix.norm(A * x - b)
    assert residu < 1e-8