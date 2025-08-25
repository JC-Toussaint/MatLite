import numpy as np
from numbers import Number
from scipy.sparse import issparse, spmatrix, csr_matrix
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import weakref
from typing import Optional, Union, Any
import copy

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

class COWData:
    """
    Classe pour gérer les données avec Copy-on-Write.
    Permet de partager les données entre plusieurs Matrix tant qu'aucune modification n'est effectuée.
    """
    
    def __init__(self, data):
        """
        Initialise un conteneur COW avec les données.
        
        Args:
            data: Les données à encapsuler (np.ndarray ou scipy.sparse)
        """
        self._data = data
        self._is_view = False
        self._parent_ref = None  # Référence faible vers le parent
        self._children = weakref.WeakSet()  # Références faibles vers les enfants
        self._is_modified = False
    
    @property
    def data(self):
        """Accès en lecture aux données."""
        return self._data
    
    @property
    def shape(self):
        """Forme des données."""
        return self._data.shape
    
    def _ensure_writable(self):
        """
        S'assure que les données sont modifiables.
        Effectue une copie si nécessaire (Copy-on-Write).
        """
        if self._is_view or len(self._children) > 0:
            # Copie nécessaire : soit c'est une vue, soit il y a des enfants
            if isinstance(self._data, np.ndarray):
                self._data = self._data.copy()
            elif sp.issparse(self._data):
                self._data = self._data.copy()
            else:
                # Fallback pour d'autres types
                self._data = copy.deepcopy(self._data)
            
            # Réinitialiser les références
            self._is_view = False
            self._parent_ref = None
            self._children.clear()
        
        self._is_modified = True
        return self._data
    
    def create_view(self):
        """
        Crée une nouvelle vue COW des mêmes données.
        
        Returns:
            COWData: Nouvelle instance partageant les mêmes données
        """
        new_cow = COWData(self._data)
        new_cow._is_view = True
        new_cow._parent_ref = weakref.ref(self)
        self._children.add(new_cow)
        return new_cow
    
    def slice_view(self, key):
        """
        Crée une vue COW pour un slice des données.
        
        Args:
            key: Clé de slicing
            
        Returns:
            COWData: Nouvelle instance avec les données slicées
        """
        if isinstance(self._data, np.ndarray):
            sliced_data = self._data[key]
            # Si c'est une vue NumPy, on la garde comme vue
            if sliced_data.base is not None:
                new_cow = COWData(sliced_data)
                new_cow._is_view = True
                new_cow._parent_ref = weakref.ref(self)
                self._children.add(new_cow)
                return new_cow
            else:
                # C'est déjà une copie
                return COWData(sliced_data)
        elif sp.issparse(self._data):
            # Pour les matrices creuses, le slicing crée généralement une copie
            return COWData(self._data[key])
        else:
            return COWData(self._data[key])
    
    def __setitem__(self, key, value):
        """Modification avec COW."""
        writable_data = self._ensure_writable()
        writable_data[key] = value
    
    def __getitem__(self, key):
        """Accès en lecture."""
        return self._data[key]

class Matrix:
    def __init__(self, data):
        """
        Constructeur pour Matrix avec support Copy-on-Write.
        - Liste de listes -> np.array(data) (forme n x m)
        - Liste simple -> np.array([data]) (1 x n)
        - np.ndarray 1D -> reshape en (1 x n)
        - np.ndarray 2D -> gardé tel quel
        - Matrice creuse -> gardée telle quelle
        - Matrix -> partage les données (COW)
        """
        if isinstance(data, Matrix):
            # Copy-on-Write : partager les données de l'autre Matrix
            self._cow_data = data._cow_data.create_view()
        elif isinstance(data, COWData):
            self._cow_data = data.create_view()
        elif isinstance(data, sp.spmatrix):
            self._cow_data = COWData(data)
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                self._cow_data = COWData(data.reshape(1, -1))  # vecteur ligne
            elif data.ndim == 2:
                self._cow_data = COWData(data)
            else:
                raise ValueError("Les matrices doivent être 1D ou 2D.")
        elif isinstance(data, list):
            if len(data) == 0:
                raise ValueError("Liste vide non supportée")
            if isinstance(data[0], list):
                self._cow_data = COWData(np.array(data))
            else:
                self._cow_data = COWData(np.array([data]))  # vecteur ligne
        else:
            raise TypeError("Data doit être un np.ndarray, une matrice creuse, une Matrix, ou une liste.")

    @property
    def data(self):
        """Accès en lecture aux données sous-jacentes."""
        return self._cow_data.data

    @property
    def shape(self):
        return self._cow_data.shape

    def copy(self):
        """
        Crée une copie indépendante de la Matrix.
        Utile quand on veut forcer une copie même avec COW.
        """
        if isinstance(self._cow_data.data, np.ndarray):
            return Matrix(self._cow_data.data.copy())
        elif sp.issparse(self._cow_data.data):
            return Matrix(self._cow_data.data.copy())
        else:
            return Matrix(copy.deepcopy(self._cow_data.data))

    def __getitem__(self, key):
        """
        Accès aux éléments avec support du slicing avancé et COW.
        """
        if isinstance(key, tuple):
            # Cas A[row_spec, col_spec]
            row_key, col_key = key
            
            # Gestion de l'indexation avancée (listes, arrays)
            if isinstance(row_key, (list, np.ndarray)) or isinstance(col_key, (list, np.ndarray)):
                return self._advanced_indexing(row_key, col_key)
            
            # Slicing standard
            data = self._cow_data.data
            if isinstance(data, np.ndarray):
                result = data[row_key, col_key]
            elif sp.issparse(data):
                result = data[row_key, col_key]
                if hasattr(result, 'toarray') and result.shape == (1, 1):
                    # Si c'est un seul élément, retourner le scalaire
                    return result.toarray()[0, 0]
            else:
                raise TypeError("Type de matrice non supporté")
                
            # Si c'est un scalaire, le retourner directement
            if np.isscalar(result):
                return result
            
            # Créer une vue COW pour les slices
            if isinstance(result, np.ndarray) and result.base is not None:
                # C'est une vue NumPy
                cow_view = self._cow_data.slice_view((row_key, col_key))
                return Matrix(cow_view)
            else:
                # C'est une copie ou matrice creuse
                return Matrix(result)
            
        else:
            # Cas A[key] - un seul indice
            data = self._cow_data.data
            rows, cols = data.shape
            
            # Pour les vecteurs, accès direct à l'élément
            if rows == 1:  # vecteur ligne
                if isinstance(data, np.ndarray):
                    result = data[0, key]
                else:  # sparse
                    result = data[0, key]
                    if hasattr(result, 'toarray'):
                        result = result.toarray()[0, 0] if result.shape == (1, 1) else result
                
                if np.isscalar(result):
                    return result
                else:
                    cow_view = self._cow_data.slice_view((0, key))
                    return Matrix(cow_view)
                
            elif cols == 1:  # vecteur colonne
                if isinstance(data, np.ndarray):
                    result = data[key, 0]
                else:  # sparse
                    result = data[key, 0]
                    if hasattr(result, 'toarray'):
                        result = result.toarray()[0, 0] if result.shape == (1, 1) else result
                
                if np.isscalar(result):
                    return result
                else:
                    cow_view = self._cow_data.slice_view((key, 0))
                    return Matrix(cow_view)
                
            else:  # matrice générale - accès à la ligne key
                if isinstance(key, slice):
                    # Slicing de lignes
                    cow_view = self._cow_data.slice_view((key, slice(None)))
                    return Matrix(cow_view)
                else:
                    # Accès à une ligne spécifique
                    cow_view = self._cow_data.slice_view((key, slice(None)))
                    return Matrix(cow_view)

    def _advanced_indexing(self, row_key, col_key):
        """
        Gestion de l'indexation avancée avec des listes ou arrays.
        L'indexation avancée crée toujours une copie, pas de COW ici.
        """
        # Conversion en arrays numpy si nécessaire
        if isinstance(row_key, list):
            row_key = np.array(row_key)
        if isinstance(col_key, list):
            col_key = np.array(col_key)
            
        data = self._cow_data.data
        if isinstance(data, np.ndarray):
            # NumPy supporte l'indexation avancée nativement
            result = data[row_key, col_key]
        elif sp.issparse(data):
            # Pour les matrices creuses, on utilise l'indexation de scipy
            result = data[row_key, col_key]
        else:
            raise TypeError("Type de matrice non supporté pour l'indexation avancée")
            
        return Matrix(result) if not np.isscalar(result) else result

    def __setitem__(self, key, value):
        """
        Affectation d'éléments avec Copy-on-Write.
        Déclenche automatiquement une copie si nécessaire.
        """
        # Conversion et préparation de la valeur
        if isinstance(value, Matrix):
            value = value.data
        
        # Conversion des formats courants pour compatibilité MATLAB
        value = self._prepare_value_for_assignment(value, key)
        
        # Utiliser COW pour l'affectation
        self._cow_data[key] = value

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
        # Addition en place - déclenche COW
        if isinstance(other, Matrix):
            new_data = self._cow_data._ensure_writable() + other.data
            self._cow_data._data = new_data
        elif _is_scalar(other):
            new_data = self._cow_data._ensure_writable() + other
            self._cow_data._data = new_data
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
        # Soustraction en place - déclenche COW
        if isinstance(other, Matrix):
            new_data = self._cow_data._ensure_writable() - other.data
            self._cow_data._data = new_data
        elif _is_scalar(other):
            new_data = self._cow_data._ensure_writable() - other
            self._cow_data._data = new_data
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

    def __xor__(self, n):
        """
        Surcharge de l'opérateur ^ pour calculer la puissance matricielle A^n.
        
        Calcule le produit matriciel de A répété n fois : A * A * ... * A (n fois)
        
        Args:
            n (int): Exposant (doit être un entier positif ou nul)
            
        Returns:
            Matrix: Résultat de A^n
            
        Raises:
            ValueError: Si la matrice n'est pas carrée ou si n < 0
            TypeError: Si n n'est pas un entier
            
        Examples:
            A = Matrix([[2, 1], [1, 1]])
            A2 = A ^ 2  # Équivaut à A * A
            A3 = A ^ 3  # Équivaut à A * A * A
        """
        # Vérification du type de l'exposant
        if not isinstance(n, int):
            raise TypeError("L'exposant doit être un entier.")
        
        if n < 0:
            raise ValueError("L'exposant doit être positif ou nul.")
        
        # Vérification que la matrice est carrée
        rows, cols = self.shape
        if rows != cols:
            raise ValueError("La puissance matricielle n'est définie que pour les matrices carrées.")
        
        # Cas particuliers
        if n == 0:
            # A^0 = I (matrice identité)
            if isinstance(self.data, np.ndarray):
                return Matrix.eye(rows, dtype=self.data.dtype)
            elif sp.issparse(self.data):
                return Matrix.eye(rows, dtype=self.data.dtype, sparse=True)
        
        if n == 1:
            # A^1 = A (copie avec COW)
            return Matrix(self)
        
        # Algorithme d'exponentiation rapide pour optimiser les grandes puissances
        result = Matrix.eye(rows, dtype=self.data.dtype, 
                        sparse=sp.issparse(self.data))
        base = Matrix(self)  # Copie COW de la matrice de base
        
        while n > 0:
            if n % 2 == 1:
                result = result * base
            base = base * base
            n //= 2
        
        return result
 
    def __pow__(self, exponent):
        """
        Surcharge de l'opérateur ** pour calculer la puissance élément par élément A**n.
        
        Calcule chaque élément de la matrice élevé à la puissance n : [A[i,j]^n]
        Équivaut à l'opération MATLAB : A.^n
        
        Args:
            exponent (scalar | Matrix): Exposant 
                - Si scalaire : applique la même puissance à tous les éléments
                - Si Matrix : puissance élément par élément (broadcasting supporté)
            
        Returns:
            Matrix: Résultat avec chaque élément élevé à la puissance correspondante
            
        Examples:
            A = Matrix([[2, 3], [4, 5]])
            A2 = A ** 2        # [[4, 9], [16, 25]]
            A_half = A ** 0.5  # [[√2, √3], [2, √5]]
            
            # Avec une matrice d'exposants
            exp = Matrix([[2, 3], [1, 2]])
            result = A ** exp  # [[2^2, 3^3], [4^1, 5^2]] = [[4, 27], [4, 25]]
        """
        data = self.data
        
        # Cas exposant scalaire
        if np.isscalar(exponent):
            if isinstance(data, np.ndarray):
                return Matrix(np.power(data, exponent))
            elif sp.issparse(data):
                # Pour les matrices creuses, appliquer power aux éléments non-nuls
                data_coo = data.tocoo()
                new_data = np.power(data_coo.data, exponent)
                from scipy.sparse import coo_matrix
                result = coo_matrix((new_data, (data_coo.row, data_coo.col)), 
                                shape=data.shape).asformat(data.getformat())
                return Matrix(result)
            else:
                raise TypeError("Type de matrice non supporté pour __pow__.")
        
        # Cas exposant Matrix
        elif isinstance(exponent, Matrix):
            exp_data = exponent.data
            
            # Matrices denses
            if isinstance(data, np.ndarray) and isinstance(exp_data, np.ndarray):
                return Matrix(np.power(data, exp_data))
            
            # Matrices creuses
            elif sp.issparse(data) and sp.issparse(exp_data):
                # Vérifier que les patterns de sparsité sont compatibles
                data_coo = data.tocoo()
                exp_coo = exp_data.tocoo()
                
                # Pour simplifier, on convertit en dense si les formes sont petites
                if np.prod(data.shape) <= 10000:  # seuil arbitraire
                    return Matrix(np.power(data.toarray(), exp_data.toarray()))
                else:
                    # Pour les grandes matrices, on suppose des patterns identiques
                    if not (np.array_equal(data_coo.row, exp_coo.row) and 
                        np.array_equal(data_coo.col, exp_coo.col)):
                        raise ValueError("Les patterns de sparsité doivent coïncider pour ** avec matrices creuses.")
                    
                    new_data = np.power(data_coo.data, exp_coo.data)
                    from scipy.sparse import coo_matrix
                    result = coo_matrix((new_data, (data_coo.row, data_coo.col)), 
                                    shape=data.shape).asformat(data.getformat())
                    return Matrix(result)
            
            # Cas mixte (dense/sparse)
            elif isinstance(data, np.ndarray) and sp.issparse(exp_data):
                return Matrix(np.power(data, exp_data.toarray()))
            elif sp.issparse(data) and isinstance(exp_data, np.ndarray):
                return Matrix(np.power(data.toarray(), exp_data))
            
            else:
                raise TypeError("Types de matrices non compatibles pour __pow__.")
        
        # Cas exposant array numpy ou liste
        elif isinstance(exponent, (np.ndarray, list)):
            exp_array = np.array(exponent) if isinstance(exponent, list) else exponent
            
            if isinstance(data, np.ndarray):
                return Matrix(np.power(data, exp_array))
            elif sp.issparse(data):
                return Matrix(np.power(data.toarray(), exp_array))
            else:
                raise TypeError("Type de matrice non supporté pour __pow__.")
        
        else:
            raise TypeError(f"Type d'exposant non supporté: {type(exponent)}")

    def __rpow__(self, base):
        """
        Surcharge de l'opérateur ** pour le cas base ** Matrix.
        
        Calcule base élevé à chaque élément de la matrice : [base^A[i,j]]
        
        Args:
            base (scalar): Base de la puissance
            
        Returns:
            Matrix: Résultat avec base élevé à chaque élément de A
            
        Example:
            A = Matrix([[1, 2], [3, 4]])
            result = 2 ** A  # [[2^1, 2^2], [2^3, 2^4]] = [[2, 4], [8, 16]]
        """
        if not np.isscalar(base):
            raise TypeError("La base doit être un scalaire pour base ** Matrix.")
        
        data = self.data
        
        if isinstance(data, np.ndarray):
            return Matrix(np.power(base, data))
        elif sp.issparse(data):
            # Pour les matrices creuses, appliquer à tous les éléments
            # Note: cela peut créer une matrice dense si base != 0
            return Matrix(np.power(base, data.toarray()))
        else:
            raise TypeError("Type de matrice non supporté pour __rpow__.")
           
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
    
    def max(self, dim=None):
        """
        Trouve les éléments maximum d'une matrice/vecteur, à la manière de MATLAB.
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

    def min(self, dim=None):
        """
        Trouve les éléments minimum d'une matrice/vecteur, à la manière de MATLAB.
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
        
    def sum(self, dim=None):
        """
        Somme des éléments d'une matrice/vecteur, à la manière de MATLAB.
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
                from scipy.sparse.linalg import svds
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
                vecs : Matrix de taille (N, n), chaque colonne est un vecteur propre
        """
        m, k = self.data.shape
        if m != k:
            raise ValueError("La matrice doit être carrée pour calculer des valeurs propres.")

        # --- Matrice dense ---
        if isinstance(self.data, np.ndarray):
            vals, vecs = np.linalg.eig(self.data)

            # Toutes les valeurs propres demandées
            if n is None or n >= len(vals):
                return vals, Matrix(vecs)

            # Sélection selon l'option
            if option == 'lm':
                idx = np.argsort(-np.abs(vals))
            elif option == 'sm':
                idx = np.argsort(np.abs(vals))
            else:
                raise ValueError("option doit être 'lm' ou 'sm'.")

            idx = idx[:n]
            return vals[idx], Matrix(vecs[:, idx])

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
        """
        if n is None:
            n = m
            
        if sparse:
            # Matrice creuse de zéros (CSR format par défaut)
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
        """
        if n is None:
            n = m
            
        if sparse:
            # Matrice creuse de uns (moins efficace, mais possible)
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
        """
        if n is None:
            n = m
            
        if sparse:
            # Matrice creuse identité ou diagonale
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
        """
        if n is None:
            n = m
            
        if random_state is not None:
            np.random.seed(random_state)
            
        if sparse:
            # Matrice creuse aléatoire
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
        """
        if n is None:
            n = m
            
        if random_state is not None:
            np.random.seed(random_state)
            
        if sparse:
            # Pour les matrices creuses, on génère d'abord une matrice dense puis on la rend creuse
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
            return Matrix(coo_matrix((new_data, (data_coo.row, data_coo.col)), 
                                     shape=data.shape).asformat(data.getformat()))
        else:
            raise TypeError("Type de matrice non supporté pour cos().")
     
    @staticmethod
    def sin(self):
        """
        Applique la fonction sinus élément par élément à la matrice.
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
            return Matrix(coo_matrix((new_data, (data_coo.row, data_coo.col)), 
                                     shape=data.shape).asformat(data.getformat()))
        else:
            raise TypeError("Type de matrice non supporté pour sin().")

    @staticmethod
    def tan(self):
        """
        Applique la fonction tangente élément par élément à la matrice.
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
            return Matrix(coo_matrix((new_data, (data_coo.row, data_coo.col)), 
                                     shape=data.shape).asformat(data.getformat()))
        else:
            raise TypeError("Type de matrice non supporté pour tan().")
    
    @staticmethod
    def exp(self):
        """
        Applique la fonction exp élément par élément à la matrice.
        Équivaut à MATLAB : exp(A)
        """
        data = self.data

        if isinstance(data, np.ndarray):
            return Matrix(np.exp(data))
        elif sp.issparse(data):
            # appliquer exp élément par élément sur les valeurs non nulles
            data_coo = data.tocoo()
            new_data = np.exp(data_coo.data)
            from scipy.sparse import coo_matrix
            return Matrix(coo_matrix((new_data, (data_coo.row, data_coo.col)), 
                                     shape=data.shape).asformat(data.getformat()))
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
            return Matrix(coo_matrix((new_data, (data_coo.row, data_coo.col)), 
                                     shape=data.shape).asformat(data.getformat()))
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
            return Matrix(coo_matrix((new_data, (data_coo.row, data_coo.col)), 
                                     shape=data.shape).asformat(data.getformat()))
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
            return Matrix(coo_matrix((new_data, (Ycoo.row, Ycoo.col)), 
                                     shape=Yd.shape).asformat(Yd.getformat()))

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
        if not isinstance(b, Matrix):
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
    
    def is_cow_active(self):
        """
        Vérifie si le Copy-on-Write est actif pour cette matrice.
        
        Returns:
            dict: Informations sur l'état COW
        """
        return {
            'is_view': self._cow_data._is_view,
            'has_children': len(self._cow_data._children) > 0,
            'is_modified': self._cow_data._is_modified,
            'children_count': len(self._cow_data._children),
            'has_parent': self._cow_data._parent_ref is not None
        }
    
    def copy(self):
        """
        Force une copie des données, même si COW est actif.
        Utile pour optimiser quand on sait qu'on va faire beaucoup de modifications.
        """
        self._cow_data._ensure_writable()
        return self
    
    def __str__(self):
        return str(self.data)

    def __repr__(self):
        cow_info = self.is_cow_active()
        status = "COW" if cow_info['is_view'] or cow_info['has_children'] else "own"
        return f"Matrix({status}):\n{repr(self.data)}"


# -----------------------
# Fonctions utilitaires pour COW
# -----------------------
def memory_usage_info(matrices):
    """
    Affiche des informations sur l'utilisation mémoire des matrices avec COW.
    
    Args:
        matrices (list): Liste de matrices à analyser
    """
    print("=== Analyse de l'utilisation mémoire COW ===")
    
    total_matrices = len(matrices)
    total_data_objects = len(set(id(m._cow_data._data) for m in matrices))
    memory_saved = total_matrices - total_data_objects
    
    print(f"Nombre total de matrices: {total_matrices}")
    print(f"Nombre d'objets de données distincts: {total_data_objects}")
    print(f"Économie mémoire (objets partagés): {memory_saved}")
    print(f"Ratio de partage: {memory_saved/total_matrices*100:.1f}%")
    
    for i, m in enumerate(matrices):
        info = m.is_cow_active()
        status = "VIEW" if info['is_view'] else ("SHARED" if info['has_children'] else "OWN")
        print(f"Matrix[{i}]: {status}, enfants: {info['children_count']}, modifiée: {info['is_modified']}")


# -----------------------
# Exemple d'utilisation avec COW
# -----------------------
def demonstrate_cow():
    """
    Démontre les avantages du Copy-on-Write.
    """
    print("=== Démonstration Copy-on-Write ===")
    
    # Création d'une matrice de base
    A = Matrix.rand(1000, 1000)
    print(f"Matrice A créée: {A.shape}")
    
    # Création de copies avec COW
    B = Matrix(A)  # Partage les données
    C = Matrix(A)  # Partage aussi les données
    D = A[100:200, 100:200]  # Vue sur une sous-région
    
    print("\nÉtat après création des copies:")
    memory_usage_info([A, B, C, D])
    
    # Modification de B - déclenche COW
    print("\nModification de B[0, 0] = 999...")
    B[0, 0] = 999
    
    print("État après modification de B:")
    memory_usage_info([A, B, C, D])
    
    # Vérification que A et C ne sont pas affectées
    print(f"\nA[0, 0] = {A[0, 0]} (inchangé)")
    print(f"B[0, 0] = {B[0, 0]} (modifié)")
    print(f"C[0, 0] = {C[0, 0]} (inchangé)")
    
    return A, B, C, D

def cg(A, b, tol=1.0e-6):
    sz = A.shape
    assert sz[0] == sz[1]
    x = Matrix.rand(sz[0], 1, random_state = 42)
    r = b-A*x
    p = r
    k=0
    while (Matrix.norm(r) > tol * Matrix.norm(b)):
        k += 1
        a = p.H* r /(p.H*A*p)
        x += a*p
        r = b-A*x
        beta = -p.H*A*r /(p.H*A*p)
        p = r + beta*p
    return x, k

def test_COW():
    N = 10
    p = Matrix.rand(1, N, random_state = 42)
    r = Matrix(p)
    print(p)
    print(r)
    p += 2*r
    print('-'*5)
    print(p)
    print(r)
                
if __name__ == "__main__":
    # Test de base avec COW
    print("=== Tests de base MatLite avec COW ===")
    
    # Création depuis np.ndarray
    A = Matrix(np.array([[3., 2.], [1., 2.]]))
    # Création depuis liste de liste
    B = Matrix([[1.], [2.]])
    
    print("A =\n", A)
    print("B =\n", B)
    print("Info COW A:", A.is_cow_active())
    
    # Test de copie COW
    C = Matrix(A)  # Partage les données
    print("Info COW C (copie de A):", C.is_cow_active())
    
    # Produit matriciel (crée une nouvelle matrice)
    result = A * B
    print("A * B =\n", result)
    
    # Test de modification avec COW
    print("\n=== Test modification avec COW ===")
    print("Avant modification C[0,0]:")
    print("A[0,0] =", A[0, 0])
    print("C[0,0] =", C[0, 0])
    
    C[0, 0] = 99  # Déclenche COW
    print("Après C[0,0] = 99:")
    print("A[0,0] =", A[0, 0], "(inchangé)")
    print("C[0,0] =", C[0, 0], "(modifié)")
    
    # Démonstration complète
    print("\n" + "="*50)
    demonstrate_cow()
    
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
    
    
    A = Matrix(np.array([[3., 2.],
                         [1., 2.]]))
    B = Matrix(np.array([[1.],
                         [2.]]))

    print(A*B)
    # Produits matriciels
    print("A * A =\n", (A * A))
    print("A @ A =\n", (A @ A))

    # Résolution de système
    B = Matrix(np.array([[5.], [5.]]))
    x = A.backslash(B)
    print("Solution système A x = B :\n", x)
    x = Matrix.backslash(A, B)
    print("Solution système A x = B :\n", x)
    
    # Création de matrices
    A = Matrix([[1, 2, 3], [4, 5, 6]])        # Depuis une liste
    B = Matrix(np.random.randn(3, 2))          # Depuis un array NumPy
    C = Matrix.eye(3)                          # Matrice identité 3x3
    D = Matrix.rand(5, 5, sparse=True)         # Matrice creuse aléatoire

    # Opérations matricielles
    result = A * B                             # Produit matriciel
    transposed = A.H                           # Transposée conjuguée
    norm_val = Matrix.norm(A)                  # Norme matricielle

    # Indexation MATLAB-like
    A[0, 1] = 10                              # Modification d'un élément
    row = A[0, :]                             # Extraction d'une ligne
    col = A[:, 1]                             # Extraction d'une colonne

    # Résolution de système linéaire Ax = B
    print(A*B)
    x = Matrix.backslash(A * A.H, A * B)    # Moindres carrés
    print(x)
    
    A = Matrix(np.array([[3., 1.],
                         [1., 2.]]))
    B = Matrix(np.array([[1.],
                         [2.]]))
    
    N = 100
    A = Matrix.rand(N, N, random_state = 42)
    A = A + A.H + N*Matrix.eye(N)
    B = Matrix.rand(N, 1, random_state = 42)
    x, iter = cg(A, B)
    print(f'iterations : {iter} residu : {Matrix.norm(B-A*x, np.inf)}')
    
    test_COW()
    print('-'*20)
    import time

    A = Matrix.rand(2000, 2000)  # ~32MB
        
    # Test COW
    start = time.time()
    B = Matrix(A)  # COW
    print(f"COW: {time.time() - start:.2g}s")  # ~0.000001s
    print("Info COW A :", A.is_cow_active())
    print("Info COW B :", B.is_cow_active())
    
    print('id : ', id(A.data), id(B.data))
    A[0, 0] = -100
    C = A
    print('id : ', id(A.data), id(B.data))
    B[0, 0] = -200
    print('id : ', id(A.data), id(B.data))
    B[0, 1] = -200
    print('id : ', id(A.data), id(B.data))
    
    print("Info COW A :", A.is_cow_active())
    print("Info COW B :", B.is_cow_active())
    print("Info COW C :", C.is_cow_active())
                 
    # Test copie immédiate
    start = time.time() 
    C = A.copy()  # Vraie copie
    print(f"Force copy: {time.time() - start:.2g}s")  # ~0.1s

    # Modification → COW se déclenche automatiquement
    B[0, 0] = 999  # Première modif → copie automatique
    
    print(Matrix.abs(A))
    print('-'*20)
    A = Matrix(np.array([[3., 1.],
                         [1., 2.]]))
    print(A**2)
    print(A^2)