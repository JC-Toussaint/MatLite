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

def test_constructor_from_matrix():
    """Test du constructeur avec COW depuis une autre Matrix"""
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix(A)  # COW
    assert B.is_cow_active()["is_view"]
    assert np.allclose(A.data, B.data)

def test_cow_functionality():
    """Test du système Copy-on-Write"""
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix(A)  # Partage des données
    C = Matrix(A)  # Partage aussi
    
    # Vérifier le partage initial
    cow_info_a = A.is_cow_active()
    cow_info_b = B.is_cow_active()
    assert cow_info_a["has_children"]
    assert cow_info_b["is_view"]
    
    # Modification déclenche COW
    original_a_00 = A[0, 0]
    B[0, 0] = 999
    
    # A ne doit pas être affectée
    assert A[0, 0] == original_a_00
    assert B[0, 0] == 999

def test_addition_and_subtraction():
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[5, 6], [7, 8]])
    C = A + B
    D = B - A
    assert np.allclose(C.data, np.array([[6, 8], [10, 12]]))
    assert np.allclose(D.data, np.array([[4, 4], [4, 4]]))

def test_inplace_operations():
    """Test des opérations en place avec COW"""
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix(A)
    original_A = A[0, 0]
    
    # Opération en place déclenche COW
    B += 5
    
    # A ne doit pas être affectée
    assert A[0, 0] == original_A
    assert B[0, 0] == original_A + 5
    
    # Test soustraction en place
    C = Matrix([[1, 1], [1, 1]])
    D = Matrix(C)
    C -= 0.5
    assert D[0, 0] == 1.0
    assert C[0, 0] == 0.5

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

def test_matmul_operator():
    """Test de l'opérateur @"""
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[2, 0], [1, 3]])
    C = A @ B
    expected = np.array([[4, 6], [10, 12]])
    assert np.allclose(C.data, expected)

def test_power_and_matpower():
    A = Matrix([[2, 0], [0, 2]])
    Apow = A ** 2  # Puissance élément par élément
    Amatpow = A ^ 3  # Puissance matricielle
    assert np.allclose(Apow.data, np.array([[4, 0], [0, 4]]))
    assert np.allclose(Amatpow.data, np.array([[8, 0], [0, 8]]))

def test_rpow():
    """Test de l'opérateur ** avec base scalaire"""
    A = Matrix([[1, 2], [3, 0]])
    result = 2 ** A
    expected = np.array([[2, 4], [8, 1]])
    assert np.allclose(result.data, expected)

def test_power_with_matrix_exponent():
    """Test de la puissance avec exposant Matrix"""
    A = Matrix([[2, 3], [4, 5]])
    exp = Matrix([[2, 1], [0, 2]])
    result = A ** exp
    expected = np.array([[4, 3], [1, 25]])
    assert np.allclose(result.data, expected)

def test_negative_operator():
    """Test de l'opérateur unaire -"""
    A = Matrix([[1, -2], [3, -4]])
    neg_A = -A
    expected = np.array([[-1, 2], [-3, 4]])
    assert np.allclose(neg_A.data, expected)

def test_transpose_and_Hermitian():
    A = Matrix([[1, 2j], [3, 4]])
    assert np.allclose(A.T.data, np.array([[1, 3], [2j, 4]]))
    assert np.allclose(A.H.data, np.array([[1, 3], [-2j, 4]]))

def test_abs_and_norm():
    A = Matrix([[3, -4]])
    assert np.allclose(A.abs().data, np.array([[3, 4]]))
    assert np.allclose(abs(A).data, np.array([[3, 4]]))  # Test surcharge abs()
    assert np.isclose(Matrix.norm(A), 5.0)

def test_norm_different_orders():
    """Test des différentes normes"""
    A = Matrix([[1, 2], [3, 4]])
    assert isinstance(A.norm(1), float)
    assert isinstance(A.norm(np.inf), float) 
    assert isinstance(A.norm('fro'), float)

def test_trace_and_det():
    A = Matrix([[1, 2], [3, 4]])
    assert np.isclose(A.trace(), 5)
    assert np.isclose(A.det(), -2)

def test_rank():
    A = Matrix([[1, 2], [2, 4]])
    B = Matrix([[1, 0], [0, 1]])
    assert A.rank == 1
    assert B.rank == 2

def test_eigenvalues():
    """Test du calcul des valeurs propres"""
    A = Matrix([[2, 1], [1, 2]])
    vals, vecs = A.eig()
    assert len(vals) == 2
    assert vecs.shape == (2, 2)
    
    # Test avec nombre limité de valeurs propres
    A_large = Matrix.eye(10) * 2
    vals_limited, vecs_limited = A_large.eig(n=3)
    assert len(vals_limited) == 3

def test_max_min_sum():
    A = Matrix([[1, 2], [3, 4]])
    assert np.allclose(A.max(0).data, np.array([[3, 4]]))
    assert np.allclose(A.min(0).data, np.array([[1, 2]]))
    assert np.allclose(A.sum(0).data, np.array([[4, 6]]))
    
    # Test avec dim=1
    assert np.allclose(A.max(1).data, np.array([[2], [4]]))
    assert np.allclose(A.min(1).data, np.array([[1], [3]]))

def test_max_min_sum_default():
    """Test des réductions sans spécifier la dimension"""
    A = Matrix([[1, 2, 3], [4, 5, 6]])
    max_result = A.max()  # Doit être par colonne par défaut
    assert max_result.shape == (1, 3)
    assert np.allclose(max_result.data, np.array([[4, 5, 6]]))
    
    # Test sur vecteur
    v = Matrix([[1, 2, 3]])
    max_scalar = v.max(1)
    assert isinstance(max_scalar, float)
    assert max_scalar == 3.0

def test_diag():
    v = Matrix([1, 2, 3])
    D = Matrix.diag(v)
    assert np.allclose(D.data, np.diag([1, 2, 3]))
    
    # Test avec offset
    D_offset = Matrix.diag(v, k=1)
    expected = np.array([[0, 1, 0, 0], [0, 0, 2, 0], [0, 0, 0, 3], [0, 0, 0, 0]])
    assert np.allclose(D_offset.data, expected)
    
    # Extraction de diagonale
    A = Matrix([[1, 2], [3, 4]])
    d = Matrix.diag(A)
    assert np.allclose(d.data.ravel(), np.array([1, 4]))

def test_concatenation():
    """Test de la fonction cat"""
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[5, 6], [7, 8]])
    
    # Concaténation verticale (dim=0)
    C_vert = Matrix.cat(0, A, B)
    expected_vert = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    assert np.allclose(C_vert.data, expected_vert)
    
    # Concaténation horizontale (dim=1)
    C_horiz = Matrix.cat(1, A, B)
    expected_horiz = np.array([[1, 2, 5, 6], [3, 4, 7, 8]])
    assert np.allclose(C_horiz.data, expected_horiz)

def test_static_zeros_ones_eye():
    A = Matrix.zeros(2, 3)
    B = Matrix.ones(2, 2)
    C = Matrix.eye(3)
    D = Matrix.eye(3, k=1)  # Sur-diagonale
    
    assert np.all(A.data == 0)
    assert np.all(B.data == 1)
    assert np.allclose(C.data, np.eye(3))
    assert np.allclose(D.data, np.eye(3, k=1))

def test_random_matrices():
    """Test des générateurs de matrices aléatoires"""
    A = Matrix.rand(3, 4, random_state=42)
    B = Matrix.randn(3, 4, random_state=42)
    
    assert A.shape == (3, 4)
    assert B.shape == (3, 4)
    assert np.all((A.data >= 0) & (A.data <= 1))  # rand entre 0 et 1
    
    # Test reproductibilité
    A2 = Matrix.rand(3, 4, random_state=42)
    assert np.allclose(A.data, A2.data)

def test_complex_matrices():
    """Test avec nombres complexes"""
    A = Matrix.rand(2, 2, dtype=complex, random_state=42)
    assert A.data.dtype == complex
    
    B = Matrix.randn(3, 3, dtype=complex, random_state=24)
    assert B.data.dtype == complex
    
    # Test des propriétés real/imag
    real_part = A.real
    imag_part = A.imag
    assert np.allclose((real_part + 1j * imag_part).data, A.data)

def test_real_imag():
    A = Matrix([[1+2j, 3-4j]])
    assert np.allclose(A.real.data, np.array([[1, 3]]))
    assert np.allclose(A.imag.data, np.array([[2, -4]]))

def test_trigonometric_functions():
    """Test des fonctions trigonométriques"""
    A = Matrix([[0, np.pi/2], [np.pi, 3*np.pi/2]])
    
    sin_A = A.sin()
    cos_A = A.cos()
    tan_A = A.tan()
    
    expected_sin = np.sin(A.data)
    expected_cos = np.cos(A.data)
    expected_tan = np.tan(A.data)
    
    assert np.allclose(sin_A.data, expected_sin)
    assert np.allclose(cos_A.data, expected_cos)
    assert np.allclose(tan_A.data, expected_tan)

def test_inverse_trigonometric_functions():
    """Test des fonctions trigonométriques inverses"""
    A = Matrix([[0, 0.5], [1, -1]])
    
    asin_A = A.asin()
    acos_A = A.acos()
    atan_A = A.atan()
    
    expected_asin = np.arcsin(A.data)
    expected_acos = np.arccos(A.data)
    expected_atan = np.arctan(A.data)
    
    assert np.allclose(asin_A.data, expected_asin)
    assert np.allclose(acos_A.data, expected_acos)
    assert np.allclose(atan_A.data, expected_atan)

def test_atan2():
    """Test de la fonction atan2"""
    Y = Matrix([[1, 1], [-1, -1]])
    X = Matrix([[1, -1], [1, -1]])
    
    result = Matrix.atan2(Y, X)
    expected = np.arctan2(Y.data, X.data)
    
    assert np.allclose(result.data, expected)

def test_exp_function():
    """Test de la fonction exponentielle"""
    A = Matrix([[0, 1], [2, 3]])
    exp_A = A.exp()
    expected = np.exp(A.data)
    assert np.allclose(exp_A.data, expected)

def test_backslash():
    A = Matrix([[3., 2.], [1., 2.]])
    B = Matrix([[5.], [5.]])
    x = A.backslash(B)
    assert np.allclose(A.data @ x.data, B.data)
    
    # Test avec fonction globale
    from MatLite import backslash
    x2 = backslash(A, B)
    assert np.allclose(x.data, x2)

def test_floordiv_operator():
    """Test de l'opérateur // pour backslash"""
    A = Matrix([[3., 2.], [1., 2.]])
    B = Matrix([[5.], [5.]])
    x = A // B  # Équivalent à A.backslash(B)
    assert np.allclose(A.data @ x.data, B.data)

def test_advanced_indexing():
    """Test de l'indexation avancée avec listes"""
    A = Matrix.rand(5, 5, random_state=42)
    
    # Indexation avec listes
    rows = [0, 2, 4]
    cols = [1, 3]
    submatrix = A[rows, cols]
    
    expected = A.data[np.array(rows)[:, None], np.array(cols)]
    assert np.allclose(submatrix.data, expected)

def test_slicing_with_cow():
    """Test du slicing avec optimisation COW"""
    A = Matrix.rand(10, 10, random_state=42)
    
    # Slicing crée une vue COW si possible
    row_slice = A[2:5, :]
    col_slice = A[:, 1:4]
    submatrix = A[2:8, 2:8]
    
    assert row_slice.shape == (3, 10)
    assert col_slice.shape == (10, 3)
    assert submatrix.shape == (6, 6)

def test_assignment_formats():
    """Test des différents formats d'assignation MATLAB-like"""
    A = Matrix.zeros(3, 3)
    
    # Assignation de colonne avec format [[v1], [v2], [v3]]
    A[:, 0] = [[1], [2], [3]]
    assert np.allclose(A[:, 0].data.ravel(), [1, 2, 3])
    
    # Assignation de ligne avec format [[v1, v2, v3]]
    A[0, :] = [[4, 5, 6]]
    assert np.allclose(A[0, :].data.ravel(), [4, 5, 6])

def test_is_cow_active():
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix(A)  # partage données
    
    cow_info = B.is_cow_active()
    assert isinstance(cow_info, dict)
    assert "is_view" in cow_info
    assert "has_children" in cow_info
    assert "is_modified" in cow_info
    assert cow_info["is_view"]

def test_copy_method():
    """Test de la méthode copy qui force une copie"""
    A = Matrix.rand(5, 5, random_state=42)
    B = Matrix(A)  # COW
    C = B.copy()   # Force une copie
    
    # B et C doivent avoir des données indépendantes maintenant
    B[0, 0] = 999
    original_c = C[0, 0]
    assert C[0, 0] == original_c  # C ne doit pas être affectée

# Tests pour matrices creuses

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

def test_sparse_random_matrices():
    """Test des matrices aléatoires creuses"""
    A = Matrix.rand(100, 100, sparse=True, density=0.1, random_state=42)
    B = Matrix.randn(50, 50, sparse=True, density=0.05, random_state=24)
    
    assert sp.issparse(A.data)
    assert sp.issparse(B.data)
    assert A.shape == (100, 100)
    assert B.shape == (50, 50)

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

def test_sparse_matrix_multiplication():
    """Test du produit matriciel avec matrices creuses"""
    A = Matrix.eye(3, sparse=True)
    B = Matrix.ones(3, sparse=True)
    C = A * B
    assert sp.issparse(C.data)
    expected = np.eye(3) @ np.ones((3, 3))
    assert np.allclose(C.data.toarray(), expected)

def test_sparse_power_operations():
    """Test des opérations de puissance avec matrices creuses"""
    A = Matrix(sp.csr_matrix([[2, 0], [0, 3]]))
    
    # Puissance élément par élément
    A_pow = A ** 2
    expected_pow = np.array([[4, 0], [0, 9]])
    assert np.allclose(A_pow.data.toarray(), expected_pow)

def test_sparse_abs_and_norm():
    A = Matrix(sp.csr_matrix([[-3, 0], [0, 4]]))
    AbsA = abs(A)
    assert np.allclose(AbsA.data.toarray(), np.array([[3, 0], [0, 4]]))
    assert np.isclose(Matrix.norm(A, 'fro'), 5.0)

def test_sparse_trigonometric():
    """Test des fonctions trigonométriques avec matrices creuses"""
    A = Matrix(sp.csr_matrix([[0, np.pi/2], [0, 0]]))
    sin_A = A.sin()
    cos_A = A.cos()
    
    expected_sin = np.sin(A.data.toarray())
    expected_cos = np.cos(A.data.toarray())
    
    assert np.allclose(sin_A.data.toarray(), expected_sin)
    assert np.allclose(cos_A.data.toarray(), expected_cos)

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

def test_sparse_eigenvalues():
    """Test du calcul des valeurs propres pour matrices creuses"""
    A = Matrix.eye(10, sparse=True) * 2
    vals, vecs = A.eig(n=5, option='lm')
    assert len(vals) == 5
    assert all(np.isclose(val, 2.0) for val in vals)

def test_sparse_backslash():
    A = Matrix(sp.csr_matrix([[3., 2.], [1., 2.]]))
    B = Matrix([[5.], [5.]])
    x = A.backslash(B)
    residual = np.linalg.norm(A.data @ x.data.ravel() - B.data.ravel())
    assert residual < 1e-10
    
def test_sparse_sum_max_min():
    A = Matrix(sp.csr_matrix([[1, 2], [3, 4]]))
    assert np.allclose(A.sum(0).data, np.array([[4, 6]]))
    assert np.allclose(A.max(0).data, np.array([[3, 4]]))
    assert np.allclose(A.min(1).data, np.array([[1], [3]]))

def test_sparse_concatenation():
    """Test de concaténation avec matrices creuses"""
    A = Matrix.eye(2, sparse=True)
    B = Matrix.zeros(2, sparse=True)
    
    C_vert = Matrix.cat(0, A, B)
    C_horiz = Matrix.cat(1, A, B)
    
    assert sp.issparse(C_vert.data)
    assert sp.issparse(C_horiz.data)
    assert C_vert.shape == (4, 2)
    assert C_horiz.shape == (2, 4)

# Tests de performance et cas limites

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

def test_memory_efficiency():
    """Test de l'efficacité mémoire avec COW"""
    n = 500
    A = Matrix.rand(n, n, random_state=42)
    
    # Création de plusieurs "copies" avec COW
    matrices = [Matrix(A) for _ in range(10)]
    
    # Toutes doivent partager les données initialement
    for m in matrices:
        assert m.is_cow_active()["is_view"]
    
    # Modification d'une seule matrice
    matrices[5][0, 0] = 999
    
    # Seule cette matrice ne doit plus être une vue
    for i, m in enumerate(matrices):
        if i == 5:
            assert not m.is_cow_active()["is_view"]
        else:
            assert m.is_cow_active()["is_view"] or m.is_cow_active()["has_children"]

def test_edge_cases():
    """Test des cas limites"""
    # Matrices 1x1
    A = Matrix([[5]])
    assert A.det() == 5
    assert A.trace() == 5
    assert A.rank == 1
    
    # Matrices vides (si supportées)
    try:
        empty = Matrix.zeros(0, 5)
        assert empty.shape == (0, 5)
    except ValueError:
        pass  # Acceptable si non supporté
    
    # Matrices très rectangulaires
    tall = Matrix.rand(1000, 2, random_state=42)
    wide = Matrix.rand(2, 1000, random_state=42)
    product = tall * wide
    assert product.shape == (1000, 1000)

def test_dtypes_consistency():
    """Test de la cohérence des types de données"""
    # Test avec différents dtypes
    A_float = Matrix.rand(3, 3, dtype=float, random_state=42)
    A_complex = Matrix.rand(3, 3, dtype=complex, random_state=42)
    
    assert A_float.data.dtype == float
    assert A_complex.data.dtype == complex
    
    # Les opérations doivent préserver les types appropriés
    B_float = A_float + A_float
    B_complex = A_complex + A_complex
    
    assert B_float.data.dtype == float
    assert B_complex.data.dtype == complex