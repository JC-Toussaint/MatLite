import numpy as np
from numbers import Number
from scipy.sparse import issparse, spmatrix, csr_matrix
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

def backslash(A, b):
    """
    Résout le système linéaire A x = b
    A : np.ndarray (pleine) ou scipy.sparse.spmatrix (creuse), carrée N x N
    b : vecteur (N,), colonne (N,1), liste, Matrix
    Retourne un vecteur colonne (N,1).
    """
    # Conversion de b si nécessaire
    if isinstance(b, Matrix):
        b = b.data
    elif isinstance(b, list):
        b = np.array(b)

    if A.shape[0] != A.shape[1]:
        raise ValueError("La matrice A doit être carrée.")
    if b.shape[0] != A.shape[0]:
        raise ValueError("Dimensions incompatibles entre A et b.")

    # Si b est (N,1), on le réduit en (N,)
    if b.ndim == 2 and b.shape[1] == 1:
        b = b.ravel()

    # Résolution
    if issparse(A):
        x = spsolve(A, b)
    else:
        x = np.linalg.solve(A, b)

    return x.reshape(-1, 1)

def _is_scalar(x):
    return np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0)

class Matrix:
    def __init__(self, data):
        """
        Constructeur pour Matrix.
        - Liste de listes -> np.array(data) (forme n x m)
        - Liste simple -> np.array([data]) (1 x n)
        - np.ndarray 1D -> reshape en (1 x n)
        - np.ndarray 2D -> gardé tel quel
        - Matrice creuse -> gardée telle quelle
        """
        if isinstance(data, sp.spmatrix):
            self.data = data
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                self.data = data.reshape(1, -1)  # vecteur ligne
            elif data.ndim == 2:
                self.data = data
            else:
                raise ValueError("Les matrices doivent être 1D ou 2D.")
        elif isinstance(data, list):
            if len(data) == 0:
                raise ValueError("Liste vide non supportée")
            if isinstance(data[0], list):
                self.data = np.array(data)
            else:
                self.data = np.array([data])  # vecteur ligne
        else:
            raise TypeError("Data doit être un np.ndarray, une matrice creuse, ou une liste.")

    @property
    def shape(self):
        return self.data.shape

    def __getitem__(self, key):
        """
        Accès aux éléments avec support du slicing avancé :
        - A[i, j] : accès à un élément
        - A[i] : accès à une ligne (pour matrices) ou élément (pour vecteurs)
        - A[i:j, k:l] : sous-matrice avec slicing
        - A[:, j] : colonne j
        - A[i, :] : ligne i
        - A[rows, cols] : indexation avancée avec listes/arrays
        """
        if isinstance(key, tuple):
            # Cas A[row_spec, col_spec]
            row_key, col_key = key
            
            # Gestion de l'indexation avancée (listes, arrays)
            if isinstance(row_key, (list, np.ndarray)) or isinstance(col_key, (list, np.ndarray)):
                return self._advanced_indexing(row_key, col_key)
            
            # Slicing standard
            if isinstance(self.data, np.ndarray):
                result = self.data[row_key, col_key]
            elif sp.issparse(self.data):
                result = self.data[row_key, col_key]
                if hasattr(result, 'toarray') and result.shape == (1, 1):
                    # Si c'est un seul élément, retourner le scalaire
                    return result.toarray()[0, 0]
            else:
                raise TypeError("Type de matrice non supporté")
                
            # Si c'est un scalaire, le retourner directement
            if np.isscalar(result):
                return result
            
            # Sinon, encapsuler dans une Matrix
            return Matrix(result)
            
        else:
            # Cas A[key] - un seul indice
            rows, cols = self.data.shape
            
            # Pour les vecteurs, accès direct à l'élément
            if rows == 1:  # vecteur ligne
                if isinstance(self.data, np.ndarray):
                    result = self.data[0, key]
                else:  # sparse
                    result = self.data[0, key]
                    if hasattr(result, 'toarray'):
                        result = result.toarray()[0, 0] if result.shape == (1, 1) else result
                return result if np.isscalar(result) else Matrix(result)
                
            elif cols == 1:  # vecteur colonne
                if isinstance(self.data, np.ndarray):
                    result = self.data[key, 0]
                else:  # sparse
                    result = self.data[key, 0]
                    if hasattr(result, 'toarray'):
                        result = result.toarray()[0, 0] if result.shape == (1, 1) else result
                return result if np.isscalar(result) else Matrix(result)
                
            else:  # matrice générale - accès à la ligne key
                if isinstance(key, slice):
                    # Slicing de lignes
                    if isinstance(self.data, np.ndarray):
                        result = self.data[key, :]
                    else:  # sparse
                        result = self.data[key, :]
                    return Matrix(result)
                else:
                    # Accès à une ligne spécifique
                    if isinstance(self.data, np.ndarray):
                        result = self.data[key, :]
                    else:  # sparse
                        result = self.data[key, :]
                    return Matrix(result)

    def _advanced_indexing(self, row_key, col_key):
        """
        Gestion de l'indexation avancée avec des listes ou arrays.
        Exemple: A[[0, 2], [1, 3]] pour extraire les éléments (0,1), (0,3), (2,1), (2,3)
        """
        # Conversion en arrays numpy si nécessaire
        if isinstance(row_key, list):
            row_key = np.array(row_key)
        if isinstance(col_key, list):
            col_key = np.array(col_key)
            
        if isinstance(self.data, np.ndarray):
            # NumPy supporte l'indexation avancée nativement
            result = self.data[row_key, col_key]
        elif sp.issparse(self.data):
            # Pour les matrices creuses, on utilise l'indexation de scipy
            result = self.data[row_key, col_key]
        else:
            raise TypeError("Type de matrice non supporté pour l'indexation avancée")
            
        return Matrix(result) if not np.isscalar(result) else result

    def __setitem__(self, key, value):
        """
        Affectation d'éléments avec support du slicing avancé :
        - A[i, j] = val : affectation d'un élément
        - A[i] = val : affectation d'une ligne (matrices) ou élément (vecteurs)
        - A[i:j, k:l] = val : affectation de sous-matrice
        - A[:, j] = val : affectation d'une colonne
        - A[i, :] = val : affectation d'une ligne
        - A[rows, cols] = val : indexation avancée
        """
        # Conversion et préparation de la valeur
        if isinstance(value, Matrix):
            value = value.data
        
        # Conversion des formats courants pour compatibilité MATLAB
        value = self._prepare_value_for_assignment(value, key)
            
        if isinstance(self.data, np.ndarray):
            if isinstance(key, tuple):
                # Cas A[row_spec, col_spec] = value
                row_key, col_key = key
                self.data[row_key, col_key] = value
            else:
                # Cas A[key] = value
                rows, cols = self.data.shape
                if rows == 1:   # vecteur ligne
                    self.data[0, key] = value
                elif cols == 1: # vecteur colonne
                    self.data[key, 0] = value
                else:           # matrice générale - affectation de ligne(s)
                    self.data[key, :] = value
                    
        elif sp.issparse(self.data):
            # Pour les matrices creuses, convertir temporairement au format approprié
            original_format = self.data.getformat()
            
            # LIL format est plus efficace pour les modifications
            if original_format != 'lil':
                tmp = self.data.tolil()
            else:
                tmp = self.data
                
            if isinstance(key, tuple):
                row_key, col_key = key
                tmp[row_key, col_key] = value
            else:
                rows, cols = tmp.shape
                if rows == 1:
                    tmp[0, key] = value
                elif cols == 1:
                    tmp[key, 0] = value
                else:
                    tmp[key, :] = value
            
            # Reconvertir au format original si nécessaire
            if original_format != 'lil':
                self.data = tmp.asformat(original_format)
            else:
                self.data = tmp
        else:
            raise TypeError("Type de matrice non supporté pour __setitem__")

    def _prepare_value_for_assignment(self, value, key):
        """
        Prépare la valeur pour l'affectation en gérant les différents formats.
        Convertit les formats MATLAB courants vers les formats NumPy appropriés.
        """
        # Si c'est déjà un scalaire, pas de conversion nécessaire
        if np.isscalar(value):
            return value
            
        # Conversion en array NumPy
        if isinstance(value, list):
            value = np.array(value)
            
        # Si c'est un array NumPy, analyser sa forme
        if isinstance(value, np.ndarray):
            # Pour l'affectation d'une colonne ou ligne, aplatir si nécessaire
            if isinstance(key, tuple):
                row_key, col_key = key
                
                # Cas spécial: affectation d'une colonne A[:, j] = [[v1], [v2], [v3]]
                if col_key == slice(None) or np.isscalar(col_key):
                    if value.ndim == 2 and value.shape[1] == 1:
                        # Colonne [[v1], [v2], [v3]] -> [v1, v2, v3]
                        value = value.ravel()
                    elif value.ndim == 2 and value.shape[0] == 1:
                        # Ligne [[v1, v2, v3]] -> [v1, v2, v3] 
                        value = value.ravel()
                        
                # Cas spécial: affectation d'une ligne A[i, :] = [[v1, v2, v3]]
                elif row_key == slice(None) or np.isscalar(row_key):
                    if value.ndim == 2 and value.shape[0] == 1:
                        # Ligne [[v1, v2, v3]] -> [v1, v2, v3]
                        value = value.ravel()
                    elif value.ndim == 2 and value.shape[1] == 1:
                        # Colonne [[v1], [v2], [v3]] -> [v1, v2, v3]
                        value = value.ravel()
            else:
                # Affectation avec un seul indice A[key] = value
                if value.ndim == 2:
                    # Aplatir les matrices 1D en vecteurs
                    if value.shape[0] == 1 or value.shape[1] == 1:
                        value = value.ravel()
                        
        return value

    # --- Produits matriciels ---
    def __matmul__(self, other):
        return self._matmul(other)

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            # Multiplication à droite par un scalaire
            return Matrix(self.data * other)
        elif isinstance(other, Matrix):
            # Vérification stricte des dimensions (nb colonnes A == nb lignes B)
            if self.data.shape[1] != other.data.shape[0]:
                raise ValueError(f"Dimensions incompatibles pour le produit matriciel : "
                                f"{self.data.shape} * {other.data.shape}")
            return Matrix(self.data @ other.data)
        else:
            raise TypeError("Multiplication non supportée pour ce type.")

    def _matmul(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("Le produit n'est défini qu'entre deux Matrix.")
        A, B = self.data, other.data
        if isinstance(A, np.ndarray) and isinstance(B, np.ndarray):
            return Matrix(A @ B)
        if isinstance(A, np.ndarray) and issparse(B):
            return Matrix(A @ B.toarray())
        if issparse(A) and isinstance(B, np.ndarray):
            return Matrix(A @ B)
        if issparse(A) and issparse(B):
            return Matrix(A @ B)
        raise TypeError("Type de matrice non supporté.")
    
    def __mul__(self, other):
        # Produit avec scalaire
        if np.isscalar(other):
            return Matrix(self.data * other)

        # Produit matriciel
        if isinstance(other, Matrix):
            if sp.issparse(self.data) or sp.issparse(other.data):
                result = self.data @ other.data
            else:
                if self.data.shape[1] != other.data.shape[0]:
                    raise ValueError(f"Dimensions incompatibles pour multiplication: "
                                     f"{self.data.shape} * {other.data.shape}")
                result = self.data @ other.data

            # si résultat est 1x1, on retourne un scalaire
            if isinstance(result, np.ndarray) and result.shape == (1, 1):
                return result[0, 0]
            return Matrix(result)

    # --- Produit scalaire gauche ---
    def __rmul__(self, other):
        if _is_scalar(other):
            return Matrix(other * self.data)
        raise TypeError("Seul un scalaire peut multiplier Matrix à gauche.")

    # --- Division scalaire droite ---
    def __truediv__(self, other):
        if not _is_scalar(other):
            raise TypeError("La division est définie uniquement par un scalaire (A / alpha).")
        other_val = float(other)
        if other_val == 0:
            raise ZeroDivisionError("Division par zéro.")
        return Matrix(self.data / other)

    # --- Addition ---
    def __add__(self, other):
        if isinstance(other, Matrix):
            if sp.issparse(self.data) or sp.issparse(other.data):
                return Matrix(self.data + other.data)
            else:
                return Matrix(self.data + other.data)
        elif _is_scalar(other):
            return Matrix(self.data + other)
        else:
            return NotImplemented

    def __radd__(self, other):
        # Cas scalaire + Matrix
        return self.__add__(other)

    def __iadd__(self, other):
        # Addition en place
        if isinstance(other, Matrix):
            self.data = self.data + other.data
        elif _is_scalar(other):
            self.data = self.data + other
        else:
            return NotImplemented
        return self

    # --- Soustraction ---
    def __sub__(self, other):
        if isinstance(other, Matrix):
            if sp.issparse(self.data) or sp.issparse(other.data):
                return Matrix(self.data - other.data)
            else:
                return Matrix(self.data - other.data)
        elif _is_scalar(other):
            return Matrix(self.data - other)
        else:
            return NotImplemented

    def __rsub__(self, other):
        # Cas scalaire - Matrix
        if _is_scalar(other):
            return Matrix(other - self.data)
        elif isinstance(other, Matrix):
            return Matrix(other.data - self.data)
        return NotImplemented

    def __isub__(self, other):
        # Soustraction en place
        if isinstance(other, Matrix):
            self.data = self.data - other.data
        elif _is_scalar(other):
            self.data = self.data - other
        else:
            return NotImplemented
        return self

    def __neg__(self):
        """
        Implémente l'opérateur unaire - pour la matrice, retournant -A.
        """
        if sp.issparse(self.data):
            # Si la matrice est creuse, appliquer la négation à chaque élément
            return Matrix(-self.data)
        else:
            # Si la matrice est dense, appliquer la négation normalement
            return Matrix(-self.data)
        
    @staticmethod
    def diag(obj, k=0):
        """
        Reproduit le comportement de Matlab :
        
        - diag(V, k) avec V vecteur (N,) ou (N,1) :
            retourne une matrice carrée de taille (N+abs(k)) 
            avec V placé sur la k-ième diagonale.
        
        - diag(X, k) avec X matrice (m x n) :
            retourne un vecteur colonne contenant les éléments 
            de la k-ième diagonale de X.
        
        Convention :
        k=0 : diagonale principale
        k>0 : diagonale au-dessus
        k<0 : diagonale en dessous
        """
        if isinstance(obj, Matrix):
            data = obj.data
        else:
            data = obj

        if isinstance(data, np.ndarray):
            # Cas Vecteur -> matrice diagonale
            if data.ndim == 1 or (data.ndim == 2 and (data.shape[0] == 1 or data.shape[1] == 1)):
                v = data.ravel()
                D = np.diag(v, k=k)
                return Matrix(D)
            # Cas matrice -> extraction de diagonale
            elif data.ndim == 2:
                d = np.diag(data, k=k)
                return Matrix(d.reshape(-1, 1))
            else:
                raise ValueError("Entrée non supportée pour diag.")
        
        elif sp.issparse(data):
            # Cas vecteur sparse -> matrice diagonale
            if data.shape[0] == 1 or data.shape[1] == 1:
                v = np.array(data.todense()).ravel()
                D = sp.diags(v, offsets=k, format="csr")
                return Matrix(D)
            # Cas matrice sparse -> extraction de diagonale
            else:
                d = data.diagonal(k=k)
                return Matrix(d.reshape(-1, 1))
        
        else:
            raise TypeError("Type non supporté pour diag.")

    @staticmethod
    def abs(self):
        """
        Calcule la valeur absolue (module) de chaque élément de la matrice.
        
        Pour les nombres réels : |x|
        Pour les nombres complexes : |a + bi| = sqrt(a² + b²)
        
        Reproduit le comportement de abs(X) dans MATLAB.
        
        Returns:
            Matrix: Nouvelle matrice contenant les valeurs absolues
        """
        if isinstance(self.data, np.ndarray):
            return Matrix(np.abs(self.data))
        elif sp.issparse(self.data):
            # Pour les matrices creuses, on utilise la méthode abs() de scipy.sparse
            return Matrix(abs(self.data))
        else:
            raise TypeError("Type de matrice non supporté pour abs().")

    def __abs__(self):
        """
        Surcharge de l'opérateur abs() intégré de Python.
        Permet d'utiliser abs(A) au lieu de A.abs()
        """
        return self.abs()
    
    @staticmethod
    def max(self, dim=None):
        """
        Trouve les éléments maximum d'une matrice/vecteur, à la manière de MATLAB.

        - M = X.max() :
            * si X est un vecteur -> maximum de tous les éléments (scalaire)
            * si X est une matrice -> vecteur ligne contenant le maximum par colonne
            * pour N-D arrays -> maximum le long de la première dimension non-singleton

        - M = X.max(dim) :
            * maximum le long de la dimension `dim` (0 = lignes, 1 = colonnes, ...)
        
        Returns:
            Matrix ou scalaire: Le ou les éléments maximum
        """
        data = self.data

        # --- Cas dense ---
        if isinstance(data, np.ndarray):
            if dim is None:
                # Détection MATLAB-like : première dimension non-singleton
                if data.ndim == 1:
                    return float(np.max(data))
                else:
                    # trouve la première dim > 1
                    dim = next((i for i, s in enumerate(data.shape) if s > 1), 0)

            m = np.max(data, axis=dim, keepdims=False)

            # vecteur ligne si maximum par colonne
            if data.ndim == 2 and dim == 0:
                return Matrix(np.array(m, ndmin=2))  # row vector
            elif data.ndim == 2 and dim == 1:
                return Matrix(np.array(m).reshape(-1, 1))  # column vector
            else:
                # scalaire ou tableau général
                if np.isscalar(m):
                    return float(m)
                return Matrix(np.array(m))

        # --- Cas sparse ---
        elif sp.issparse(data):
            if dim is None:
                # par défaut maximum par colonne (comme MATLAB)
                dim = 0

            m = data.max(axis=dim)

            if dim == 0:
                # Convertir en array dense pour le résultat
                if hasattr(m, 'toarray'):
                    m = m.toarray()
                return Matrix(np.array(m).ravel()[np.newaxis, :])  # row vector
            elif dim == 1:
                if hasattr(m, 'toarray'):
                    m = m.toarray()
                return Matrix(np.array(m).ravel()[:, np.newaxis])  # column vector
            else:
                raise ValueError("Dimension non supportée pour une matrice creuse.")

        else:
            raise TypeError("Type de matrice non supporté pour max().")

    @staticmethod
    def min(self, dim=None):
        """
        Trouve les éléments minimum d'une matrice/vecteur, à la manière de MATLAB.

        - M = X.min() :
            * si X est un vecteur -> minimum de tous les éléments (scalaire)
            * si X est une matrice -> vecteur ligne contenant le minimum par colonne
            * pour N-D arrays -> minimum le long de la première dimension non-singleton

        - M = X.min(dim) :
            * minimum le long de la dimension `dim` (0 = lignes, 1 = colonnes, ...)
        
        Returns:
            Matrix ou scalaire: Le ou les éléments minimum
        """
        data = self.data

        # --- Cas dense ---
        if isinstance(data, np.ndarray):
            if dim is None:
                # Détection MATLAB-like : première dimension non-singleton
                if data.ndim == 1:
                    return float(np.min(data))
                else:
                    # trouve la première dim > 1
                    dim = next((i for i, s in enumerate(data.shape) if s > 1), 0)

            m = np.min(data, axis=dim, keepdims=False)

            # vecteur ligne si minimum par colonne
            if data.ndim == 2 and dim == 0:
                return Matrix(np.array(m, ndmin=2))  # row vector
            elif data.ndim == 2 and dim == 1:
                return Matrix(np.array(m).reshape(-1, 1))  # column vector
            else:
                # scalaire ou tableau général
                if np.isscalar(m):
                    return float(m)
                return Matrix(np.array(m))

        # --- Cas sparse ---
        elif sp.issparse(data):
            if dim is None:
                # par défaut minimum par colonne (comme MATLAB)
                dim = 0

            m = data.min(axis=dim)

            if dim == 0:
                # Convertir en array dense pour le résultat
                if hasattr(m, 'toarray'):
                    m = m.toarray()
                return Matrix(np.array(m).ravel()[np.newaxis, :])  # row vector
            elif dim == 1:
                if hasattr(m, 'toarray'):
                    m = m.toarray()
                return Matrix(np.array(m).ravel()[:, np.newaxis])  # column vector
            else:
                raise ValueError("Dimension non supportée pour une matrice creuse.")

        else:
            raise TypeError("Type de matrice non supporté pour min().")
        
    @staticmethod
    def sum(self, dim=None):
        """
        Somme des éléments d'une matrice/vecteur, à la manière de MATLAB.

        - S = X.sum() :
            * si X est un vecteur -> somme de tous les éléments (scalaire)
            * si X est une matrice -> vecteur ligne contenant la somme par colonne
            * pour N-D arrays -> somme le long de la première dimension non-singleton

        - S = X.sum(dim) :
            * somme le long de la dimension `dim` (0 = lignes, 1 = colonnes, ...)
        """
        data = self.data

        # --- Cas dense ---
        if isinstance(data, np.ndarray):
            if dim is None:
                # Détection MATLAB-like : première dimension non-singleton
                if data.ndim == 1:
                    return float(np.sum(data))
                else:
                    # trouve la première dim > 1
                    dim = next((i for i, s in enumerate(data.shape) if s > 1), 0)

            s = np.sum(data, axis=dim, keepdims=False)

            # vecteur ligne si somme par colonne
            if data.ndim == 2 and dim == 0:
                return Matrix(np.array(s, ndmin=2))  # row vector
            elif data.ndim == 2 and dim == 1:
                return Matrix(np.array(s).reshape(-1, 1))  # column vector
            else:
                # scalaire ou tableau général
                return Matrix(np.array(s))

        # --- Cas sparse ---
        elif sp.issparse(data):
            if dim is None:
                # par défaut somme par colonne (comme MATLAB)
                dim = 0

            s = data.sum(axis=dim)

            if dim == 0:
                return Matrix(np.array(s).ravel()[np.newaxis, :])  # row vector
            elif dim == 1:
                return Matrix(np.array(s).ravel()[:, np.newaxis])  # column vector
            else:
                raise ValueError("Dimension non supportée pour une matrice creuse.")

        else:
            raise TypeError("Type de matrice non supporté pour sum().")

    @staticmethod
    def trace(self):
        """
        Somme des éléments diagonaux de la matrice.
        Équivaut à la somme des valeurs propres.
        """
        if isinstance(self.data, np.ndarray):
            return float(np.trace(self.data))
        elif sp.issparse(self.data):
            return float(self.data.diagonal().sum())
        else:
            raise TypeError("Type de matrice non supporté pour trace.")
        
    @staticmethod
    def det(self):
        """
        Déterminant de la matrice.
        Vérifie que la matrice est carrée.
        """
        m, n = self.data.shape
        if m != n:
            raise ValueError("Le déterminant n'est défini que pour les matrices carrées.")

        if isinstance(self.data, np.ndarray):
            return float(np.linalg.det(self.data))
        elif sp.issparse(self.data):
            # Pas de support direct du déterminant pour sparse
            # On convertit en dense (si taille raisonnable)
            return float(np.linalg.det(self.data.toarray()))
        else:
            raise TypeError("Type de matrice non supporté pour det.")

    @property
    def rank(self):
        """
        Calcule le rang de la matrice.
        Le rang est le nombre de lignes (ou colonnes) linéairement indépendantes.
        
        Pour les matrices denses : utilise numpy.linalg.matrix_rank
        Pour les matrices creuses : utilise la décomposition SVD sparse
        
        Returns:
            int: Le rang de la matrice
        """
        if isinstance(self.data, np.ndarray):
            return int(np.linalg.matrix_rank(self.data))
        
        elif sp.issparse(self.data):
            # Pour les matrices creuses, on utilise SVD
            m, n = self.data.shape
            
            # Cas particuliers
            if m == 0 or n == 0:
                return 0
            
            if m == 1 or n == 1:
                # Vecteur : rang = 1 si non nul, 0 sinon
                return 1 if self.data.nnz > 0 else 0
            
            # Pour les petites matrices, on peut convertir en dense
            if m * n <= 10000:  # seuil arbitraire
                return int(np.linalg.matrix_rank(self.data.toarray()))
            
            # Pour les grandes matrices creuses, utiliser SVD sparse
            try:
                k = min(m, n, self.data.nnz) - 1
                if k <= 0:
                    return 1 if self.data.nnz > 0 else 0
                
                # Calcul des valeurs singulières
                singular_values = svds(self.data, k=k, return_singular_vectors=False)
                
                # Compter les valeurs singulières non-nulles (avec tolérance)
                tol = max(m, n) * np.finfo(float).eps * np.max(singular_values)
                rank_sparse = np.sum(singular_values > tol)
                
                # Il peut y avoir une valeur singulière de plus
                return min(rank_sparse + 1, min(m, n))
                
            except:
                # En cas d'échec, conversion en dense si possible
                if m * n <= 100000:
                    return int(np.linalg.matrix_rank(self.data.toarray()))
                else:
                    raise RuntimeError("Impossible de calculer le rang pour cette matrice creuse")
        
        else:
            raise TypeError("Type de matrice non supporté pour rank.")
   
    @staticmethod
    def norm(self, ord=2):
        """
        Calcule la norme d'un vecteur ou d'une matrice, à la manière de MATLAB.

        norm(X,2)   : 2-norme (valeur singulière max pour matrice, euclidienne pour vecteur)
        norm(X)     : équivalent à norm(X,2)
        norm(X,1)   : norme 1 (somme des valeurs absolues pour vecteur,
                                max des sommes colonnes pour matrice)
        norm(X,Inf) : norme infinie (max des valeurs absolues pour vecteur,
                                      max des sommes lignes pour matrice)
        norm(X,'fro'): norme de Frobenius (racine de la somme des carrés)
        """
        data = self.data

        # --- Dense ---
        if isinstance(data, np.ndarray):
            return float(np.linalg.norm(data, ord=ord))

        # --- Creuse ---
        elif sp.issparse(data):
            if ord in [None, 2]:
                # Norme 2 pour matrice creuse = plus grande valeur singulière
                # SciPy ne supporte pas directement, mais on peut approximer
                try:
                    # Calcul de la plus grande valeur singulière via svds
                    from scipy.sparse.linalg import svds
                    u, s, vt = svds(data, k=1)
                    return float(s.max())
                except Exception:
                    # fallback : conversion en dense si petite
                    if np.prod(data.shape) <= 4096:  # seuil arbitraire
                        return float(np.linalg.norm(data.toarray(), ord=2))
                    raise RuntimeError("Impossible de calculer la norme 2 pour cette matrice creuse.")
            elif ord == 1:
                return float(abs(data).sum(axis=0).max())
            elif ord == np.inf:
                return float(abs(data).sum(axis=1).max())
            elif ord == 'fro':
                return float(np.sqrt((abs(data).power(2)).sum()))
            else:
                raise ValueError("Norme non supportée pour matrices creuses avec ord=" + str(ord))

        else:
            raise TypeError("Type de matrice non supporté pour norm().")

    @staticmethod
    def eig(self, n=None, option='lm'):
        """
        Calcule les valeurs propres et vecteurs propres de la matrice.

        Args:
            n (int | None):
                - None : toutes les valeurs propres (dense uniquement)
                - sinon : nombre de valeurs propres à extraire
            option (str):
                - 'lm' : valeurs propres de plus grande amplitude (largest magnitude)
                - 'sm' : valeurs propres de plus petite amplitude (smallest magnitude)

        Returns:
            (vals, vecs):
                vals : np.ndarray de taille (n,)
                vecs : np.ndarray de taille (N, n), chaque colonne est un vecteur propre
        """
        m, k = self.data.shape
        if m != k:
            raise ValueError("La matrice doit être carrée pour calculer des valeurs propres.")

        # --- Matrice dense ---
        if isinstance(self.data, np.ndarray):
            vals, vecs = np.linalg.eig(self.data)

            # Toutes les valeurs propres demandées
            if n is None or n >= len(vals):
                return vals, vecs

            # Sélection selon l'option
            if option == 'lm':
                idx = np.argsort(-np.abs(vals))
            elif option == 'sm':
                idx = np.argsort(np.abs(vals))
            else:
                raise ValueError("option doit être 'lm' ou 'sm'.")

            idx = idx[:n]
            return vals[idx], vecs[:, idx]

        # --- Matrice creuse ---
        elif sp.issparse(self.data):
            if n is None:
                raise ValueError("Pour les matrices creuses, n doit être spécifié.")

            # Détection hermitienne/symétrique
            is_hermitian = (self.data - self.data.getH()).nnz == 0

            if is_hermitian:
                # eigsh = plus stable pour hermitiennes
                if option == 'lm':
                    vals, vecs = spla.eigsh(self.data, k=n, which='LM')
                elif option == 'sm':
                    vals, vecs = spla.eigsh(self.data, k=n, which='SM')
                else:
                    raise ValueError("option doit être 'lm' ou 'sm'.")
            else:
                if option == 'lm':
                    vals, vecs = spla.eigs(self.data, k=n, which='LM')
                elif option == 'sm':
                    vals, vecs = spla.eigs(self.data, k=n, which='SM')
                else:
                    raise ValueError("option doit être 'lm' ou 'sm'.")

            return vals, Matrix(vecs)

        else:
            raise TypeError("Type de matrice non supporté pour eig().")
  
    @staticmethod
    def zeros(m, n=None, dtype=float, sparse=False):
        """
        Crée une matrice de zéros, à la manière de MATLAB.
        
        Args:
            m (int): Nombre de lignes
            n (int, optional): Nombre de colonnes. Si None, crée une matrice carrée m×m
            dtype (type, optional): Type de données (float, int, complex, etc.)
            sparse (bool, optional): Si True, crée une matrice creuse
            
        Returns:
            Matrix: Matrice de zéros
            
        Examples:
            Matrix.zeros(3)        # matrice 3×3 de zéros
            Matrix.zeros(2, 4)     # matrice 2×4 de zéros
            Matrix.zeros(3, 3, dtype=int)  # matrice 3×3 d'entiers zéros
            Matrix.zeros(100, 100, sparse=True)  # matrice creuse 100×100
        """
        if n is None:
            n = m
            
        if sparse:
            # Matrice creuse de zéros (CSR format par défaut)
            import scipy.sparse as sp
            data = sp.csr_matrix((m, n), dtype=dtype)
            return Matrix(data)
        else:
            # Matrice dense de zéros
            data = np.zeros((m, n), dtype=dtype)
            return Matrix(data)

    @staticmethod
    def ones(m, n=None, dtype=float, sparse=False):
        """
        Crée une matrice de uns, à la manière de MATLAB.
        
        Args:
            m (int): Nombre de lignes
            n (int, optional): Nombre de colonnes. Si None, crée une matrice carrée m×m
            dtype (type, optional): Type de données (float, int, complex, etc.)
            sparse (bool, optional): Si True, crée une matrice creuse (déconseillé pour ones)
            
        Returns:
            Matrix: Matrice de uns
            
        Examples:
            Matrix.ones(3)         # matrice 3×3 de uns
            Matrix.ones(2, 4)      # matrice 2×4 de uns
            Matrix.ones(3, 3, dtype=int)  # matrice 3×3 d'entiers uns
        """
        if n is None:
            n = m
            
        if sparse:
            # Matrice creuse de uns (moins efficace, mais possible)
            import scipy.sparse as sp
            data = sp.csr_matrix(np.ones((m, n), dtype=dtype))
            return Matrix(data)
        else:
            # Matrice dense de uns
            data = np.ones((m, n), dtype=dtype)
            return Matrix(data)

    @staticmethod
    def eye(m, n=None, k=0, dtype=float, sparse=False):
        """
        Crée une matrice identité ou avec des uns sur une diagonale, à la manière de MATLAB.
        
        Args:
            m (int): Nombre de lignes
            n (int, optional): Nombre de colonnes. Si None, crée une matrice carrée m×m
            k (int, optional): Décalage de la diagonale (0=principale, >0=au-dessus, <0=en-dessous)
            dtype (type, optional): Type de données (float, int, complex, etc.)
            sparse (bool, optional): Si True, crée une matrice creuse
            
        Returns:
            Matrix: Matrice identité ou diagonale
            
        Examples:
            Matrix.eye(3)          # matrice identité 3×3
            Matrix.eye(3, 4)       # matrice 3×4 avec des 1 sur la diagonale principale
            Matrix.eye(4, k=1)     # matrice 4×4 avec des 1 sur la sur-diagonale
            Matrix.eye(4, k=-1)    # matrice 4×4 avec des 1 sur la sous-diagonale
            Matrix.eye(1000, sparse=True)  # matrice identité creuse 1000×1000
        """
        if n is None:
            n = m
            
        if sparse:
            # Matrice creuse identité ou diagonale
            import scipy.sparse as sp
            if k == 0 and m == n:
                # Cas spécial: matrice identité carrée
                data = sp.eye(m, dtype=dtype, format='csr')
            else:
                # Cas général: diagonale décalée
                data = sp.diags([1], offsets=[k], shape=(m, n), dtype=dtype, format='csr')
            return Matrix(data)
        else:
            # Matrice dense identité ou diagonale
            data = np.eye(m, n, k=k, dtype=dtype)
            return Matrix(data)

    @staticmethod
    def rand(m, n=None, dtype=float, sparse=False, density=0.1, random_state=None):
        """
        Crée une matrice de nombres aléatoires uniformes entre 0 et 1, à la manière de MATLAB.
        
        Args:
            m (int): Nombre de lignes
            n (int, optional): Nombre de colonnes. Si None, crée une matrice carrée m×m
            dtype (type, optional): Type de données (float, complex, etc.)
            sparse (bool, optional): Si True, crée une matrice creuse
            density (float, optional): Densité pour matrices creuses (proportion d'éléments non-nuls)
            random_state (int, optional): Graine pour la génération aléatoire
            
        Returns:
            Matrix: Matrice de nombres aléatoires
            
        Examples:
            Matrix.rand(3)         # matrice 3×3 aléatoire
            Matrix.rand(2, 4)      # matrice 2×4 aléatoire
            Matrix.rand(100, 100, sparse=True, density=0.05)  # matrice creuse avec 5% d'éléments
        """
        if n is None:
            n = m
            
        if random_state is not None:
            np.random.seed(random_state)
            
        if sparse:
            # Matrice creuse aléatoire
            import scipy.sparse as sp
            data = sp.random(m, n, density=density, format='csr', dtype=dtype, random_state=random_state)
            return Matrix(data)
        else:
            # Matrice dense aléatoire
            if dtype == complex:
                # Nombres complexes: partie réelle + imaginaire aléatoires
                real_part = np.random.rand(m, n)
                imag_part = np.random.rand(m, n)
                data = real_part + 1j * imag_part
            else:
                data = np.random.rand(m, n).astype(dtype)
            return Matrix(data)

    @staticmethod
    def randn(m, n=None, dtype=float, sparse=False, density=0.1, random_state=None):
        """
        Crée une matrice de nombres aléatoires suivant une distribution normale (0, 1).
        
        Args:
            m (int): Nombre de lignes
            n (int, optional): Nombre de colonnes. Si None, crée une matrice carrée m×m
            dtype (type, optional): Type de données (float, complex, etc.)
            sparse (bool, optional): Si True, crée une matrice creuse avec distribution normale
            density (float, optional): Densité pour matrices creuses
            random_state (int, optional): Graine pour la génération aléatoire
            
        Returns:
            Matrix: Matrice de nombres aléatoires gaussiens
            
        Examples:
            Matrix.randn(3)        # matrice 3×3 gaussienne
            Matrix.randn(2, 4)     # matrice 2×4 gaussienne
        """
        if n is None:
            n = m
            
        if random_state is not None:
            np.random.seed(random_state)
            
        if sparse:
            # Pour les matrices creuses, on génère d'abord une matrice dense puis on la rend creuse
            import scipy.sparse as sp
            if dtype == complex:
                real_part = np.random.randn(m, n)
                imag_part = np.random.randn(m, n)
                dense_data = real_part + 1j * imag_part
            else:
                dense_data = np.random.randn(m, n).astype(dtype)
            
            # Masque aléatoire pour la sparsité
            mask = np.random.rand(m, n) < density
            dense_data = dense_data * mask
            data = sp.csr_matrix(dense_data)
            return Matrix(data)
        else:
            # Matrice dense gaussienne
            if dtype == complex:
                real_part = np.random.randn(m, n)
                imag_part = np.random.randn(m, n)
                data = real_part + 1j * imag_part
            else:
                data = np.random.randn(m, n).astype(dtype)
                print(data)
                print('='*20)
            return Matrix(data)

    @staticmethod
    def cos(self):
        """
        Applique la fonction cosinus élément par élément à la matrice.
        Équivaut à MATLAB : cos(A)
        """
        data = self.data

        if isinstance(data, np.ndarray):
            return Matrix(np.cos(data))
        elif sp.issparse(data):
            # appliquer cos élément par élément sur les valeurs non nulles
            data_coo = data.tocoo()
            new_data = np.cos(data_coo.data)
            from scipy.sparse import coo_matrix
            return Matrix(coo_matrix((new_data, (data_coo.row, data_coo.col)), shape=data.shape).asformat(data.getformat()))
        else:
            raise TypeError("Type de matrice non supporté pour cos().")
     
    @staticmethod   
    def sin(self):
        """
        Applique la fonction sininus élément par élément à la matrice.
        Équivaut à MATLAB : sin(A)
        """
        data = self.data

        if isinstance(data, np.ndarray):
            return Matrix(np.sin(data))
        elif sp.issparse(data):
            # appliquer sin élément par élément sur les valeurs non nulles
            data_coo = data.tocoo()
            new_data = np.sin(data_coo.data)
            from scipy.sparse import coo_matrix
            return Matrix(coo_matrix((new_data, (data_coo.row, data_coo.col)), shape=data.shape).asformat(data.getformat()))
        else:
            raise TypeError("Type de matrice non supporté pour sin().")
 
    @staticmethod
    def tan(self):
        """
        Applique la fonction taninus élément par élément à la matrice.
        Équivaut à MATLAB : tan(A)
        """
        data = self.data

        if isinstance(data, np.ndarray):
            return Matrix(np.tan(data))
        elif sp.issparse(data):
            # appliquer tan élément par élément sur les valeurs non nulles
            data_coo = data.tocoo()
            new_data = np.tan(data_coo.data)
            from scipy.sparse import coo_matrix
            return Matrix(coo_matrix((new_data, (data_coo.row, data_coo.col)), shape=data.shape).asformat(data.getformat()))
        else:
            raise TypeError("Type de matrice non supporté pour tan().")
    
    @staticmethod           
    def exp(self):
        """
        Applique la fonction exp élément par élément à la matrice.
        Équivaut à MATLAB : exp(A)
        """
        data = self.data

        if iexpstance(data, np.ndarray):
            return Matrix(np.exp(data))
        elif sp.issparse(data):
            # appliquer exp élément par élément sur les valeurs non nulles
            data_coo = data.tocoo()
            new_data = np.exp(data_coo.data)
            from scipy.sparse import coo_matrix
            return Matrix(coo_matrix((new_data, (data_coo.row, data_coo.col)), shape=data.shape).asformat(data.getformat()))
        else:
            raise TypeError("Type de matrice non supporté pour exp().")
  
    @staticmethod
    def acos(self):
        """
        Applique la fonction arccos (acos) élément par élément à la matrice.
        Équivaut à MATLAB : acos(A)
        """
        data = self.data

        if isinstance(data, np.ndarray):
            return Matrix(np.arccos(data))
        elif sp.issparse(data):
            # appliquer acos uniquement sur les éléments non nuls
            data_coo = data.tocoo()
            new_data = np.arccos(data_coo.data)
            from scipy.sparse import coo_matrix
            return Matrix(coo_matrix((new_data, (data_coo.row, data_coo.col)), shape=data.shape).asformat(data.getformat()))
        else:
            raise TypeError("Type de matrice non supporté pour acos().")

    @staticmethod
    def asin(self):
        """
        Applique la fonction arcsin (asin) élément par élément à la matrice.
        Équivaut à MATLAB : asin(A)
        """
        data = self.data

        if isinstance(data, np.ndarray):
            return Matrix(np.arcsin(data))
        elif sp.issparse(data):
            # appliquer asin uniquement sur les éléments non nuls
            data_coo = data.tocoo()
            new_data = np.arcsin(data_coo.data)
            from scipy.sparse import coo_matrix
            return Matrix(coo_matrix((new_data, (data_coo.row, data_coo.col)), shape=data.shape).asformat(data.getformat()))
        else:
            raise TypeError("Type de matrice non supporté pour asin().")

    @staticmethod
    def atan(self):
        """
        Applique la fonction arctan (atan) élément par élément à la matrice.
        Équivaut à MATLAB : atan(A)
        """
        data = self.data

        if isinstance(data, np.ndarray):
            return Matrix(np.arctan(data))
        elif sp.issparse(data):
            # appliquer atan uniquement sur les éléments non nuls
            data_coo = data.tocoo()
            new_data = np.arctan(data_coo.data)
            from scipy.sparse import coo_matrix
            return Matrix(coo_matrix((new_data, (data_coo.row, data_coo.col)), 
                                     shape=data.shape).asformat(data.getformat()))
        else:
            raise TypeError("Type de matrice non supporté pour atan().")
  
    @staticmethod
    def atan2(Y, X):
        """
        Applique atan2(Y,X) élément par élément (comme MATLAB).
        
        Args:
            Y, X : Matrix | np.ndarray | sparse | scalar
        
        Returns:
            Matrix
        """
        # Extraire les données sous-jacentes
        Yd = Y.data if isinstance(Y, Matrix) else Y
        Xd = X.data if isinstance(X, Matrix) else X

        # Cas numpy (dense)
        if isinstance(Yd, np.ndarray) or np.isscalar(Yd):
            return Matrix(np.arctan2(Yd, Xd))

        # Cas sparse
        elif sp.issparse(Yd) and sp.issparse(Xd):
            # on suppose qu'ils ont les mêmes positions de non-nuls
            Ycoo = Yd.tocoo()
            Xcoo = Xd.tocoo()

            if not (np.array_equal(Ycoo.row, Xcoo.row) and np.array_equal(Ycoo.col, Xcoo.col)):
                raise ValueError("Pour les matrices creuses, les schémas de non-nuls doivent coïncider.")

            new_data = np.arctan2(Ycoo.data, Xcoo.data)
            from scipy.sparse import coo_matrix
            return Matrix(coo_matrix((new_data, (Ycoo.row, Ycoo.col)), shape=Yd.shape).asformat(Yd.getformat()))

        else:
            raise TypeError("Types non supportés pour atan2.")
                               
    # --- Transposée Hermitienne ---
    @property
    def H(self):
        if isinstance(self.data, np.ndarray):
            return Matrix(self.data.conj().T)
        elif issparse(self.data):
            return Matrix(self.data.getH())
        else:
            raise TypeError("Type de matrice non supporté pour H.")

    def backslash(self, b):
        """
        Résout Ax = b pour x.
        """
        if not (isinstance(b, Matrix)):
            raise TypeError("b doit être un objet Matrix")

        if sp.issparse(self.data):
            x = spla.spsolve(self.data, b.data)
        else:
            x = np.linalg.solve(self.data, b.data)

        return Matrix(x)

    @property
    def real(self):
        """Partie réelle de la matrice"""
        return Matrix(self.data.real)

    @property
    def imag(self):
        """Partie imaginaire de la matrice"""
        return Matrix(self.data.imag)
    
    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return f"Matrix({repr(self.data)})"
      
# -----------------------
# Exemple d'utilisation
# -----------------------
if __name__ == "__main__":
    # Création depuis np.ndarray
    A = Matrix(np.array([[3., 2.], [1., 2.]]))
    # Création depuis liste de liste
    B = Matrix([[1.], [2.]])

    print("A =\n", A)
    print("B =\n", B)

    # Produit matriciel
    print("A * B =\n", (A * B))
    print("norm(A) ", Matrix.norm(A, np.inf))
    norm_inf = np.max(np.sum(np.abs(A.data), axis=1))
    print(norm_inf) 
    
    R = Matrix.rand(5, n=1, dtype=float, sparse=True, density=0.1, random_state=None).H
    print(R.data)
    print(R[0])
    print(Matrix.atan2(A[:, 0], A[:, 1]))
    
    R = Matrix.rand(10, n=10, dtype=float, sparse=False, density=0.1, random_state=None)
    print(R.shape)
    plt.plot(R.data[:, 0], R.data[:, 1])
    plt.grid(True)
    plt.show()
    
    print(R)
    print('-'*20)
    print(Matrix.sum(R, 0))
    