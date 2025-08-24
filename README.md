# MatLite

Une bibliothèque Python offrant une interface MATLAB-like pour les opérations matricielles avec support natif des matrices denses et creuses, optimisée avec un système Copy-on-Write (COW) pour une gestion mémoire efficace.

## Auteur

**Jean-Christophe Toussaint**  
Grenoble-INP Phelma

## Description

MatLite fournit une classe `Matrix` qui encapsule les matrices NumPy et SciPy.sparse, offrant une syntaxe familière aux utilisateurs de MATLAB tout en bénéficiant de la performance des bibliothèques Python scientifiques et d'un système de Copy-on-Write pour une gestion mémoire optimisée.

### Caractéristiques principales

- ✅ Interface MATLAB-like intuitive
- ✅ Support des matrices denses (NumPy) et creuses (SciPy.sparse)
- ✅ **Système Copy-on-Write (COW)** pour optimisation mémoire
- ✅ Indexation et slicing avancés compatible MATLAB
- ✅ Opérations matricielles optimisées
- ✅ Fonctions mathématiques élémentaires complètes
- ✅ Générateurs de matrices (zeros, ones, eye, rand, randn)
- ✅ Résolution de systèmes linéaires avec `backslash`
- ✅ **Opérations en place avec COW automatique** (`+=`, `-=`)
- ✅ **Fonctions de diagnostic COW** pour l'analyse mémoire
- ✅ **Support complet des nombres complexes**

## Installation

```bash
git clone https://github.com/JC-Toussaint/MatLite.git
cd MatLite
pip install -r requirements.txt
```

### Dépendances

- NumPy
- SciPy
- Matplotlib (pour les exemples)

## Guide de démarrage rapide

```python
from MatLite import Matrix
import numpy as np

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

# Résolution de système linéaire Ax = b
x = Matrix.backslash(A.H * A, A.H * B)    # Moindres carrés
```

## 🚀 Nouveauté : Copy-on-Write (COW)

MatLite intègre un système Copy-on-Write avancé qui optimise automatiquement l'utilisation mémoire :

```python
# Création d'une grande matrice
A = Matrix.rand(2000, 2000)  # ~32MB

# Les copies partagent les données jusqu'à modification
B = Matrix(A)  # Instantané, pas de copie mémoire
C = Matrix(A)  # Partage aussi les données avec A

# La modification déclenche automatiquement une copie
B[0, 0] = 999  # Seule B est copiée, A et C restent partagées

# Analyse de l'utilisation mémoire
print("État COW de A:", A.is_cow_active())
print("État COW de B:", B.is_cow_active())

# Outils de diagnostic
from MatLite import memory_usage_info
memory_usage_info([A, B, C])
```

### Avantages du COW

- **Économie mémoire** : Les copies partagent les données tant qu'aucune modification n'est effectuée
- **Performance** : Copie instantanée des grandes matrices
- **Transparence** : Fonctionne automatiquement, aucune modification de code nécessaire
- **Sécurité** : Garantit l'isolation des données lors des modifications

## Documentation détaillée

### Création de matrices

```python
# Différentes façons de créer des matrices
A = Matrix([[1, 2], [3, 4]])              # Liste de listes → matrice 2x2
v = Matrix([1, 2, 3])                     # Liste → vecteur ligne 1x3
B = Matrix(np.array([[1], [2], [3]]))     # Array NumPy → vecteur colonne 3x1

# Générateurs de matrices
zeros = Matrix.zeros(3, 4)                # Matrice 3x4 de zéros
ones = Matrix.ones(2, 2)                  # Matrice 2x2 de uns  
eye = Matrix.eye(5, k=1)                  # Matrice 5x5, uns sur sur-diagonale
rand = Matrix.rand(3, 3)                  # Matrice 3x3 aléatoire [0,1]
randn = Matrix.randn(2, 4)                # Matrice 2x4 gaussienne N(0,1)

# Matrices creuses
sparse_eye = Matrix.eye(1000, sparse=True)        # Identité creuse 1000x1000
sparse_rand = Matrix.rand(100, 100, sparse=True, density=0.05)  # 5% de remplissage

# Matrices complexes
complex_rand = Matrix.rand(3, 3, dtype=complex)   # Matrice complexe aléatoire
complex_randn = Matrix.randn(3, 3, dtype=complex) # Matrice complexe gaussienne
```

### Indexation et slicing avancés

```python
A = Matrix.rand(5, 5)

# Accès aux éléments
element = A[2, 3]                         # Élément (2,3)
A[1, 1] = 42                             # Modification (déclenche COW si nécessaire)

# Slicing avec COW optimisé
row = A[2, :]                            # Ligne 2 complète (vue COW si possible)
col = A[:, 1]                            # Colonne 1 complète (vue COW si possible)
submatrix = A[1:4, 0:3]                  # Sous-matrice 3x3 (vue COW si possible)

# Indexation avancée
selected = A[[0, 2, 4], [1, 3]]          # Éléments spécifiques
A[:, 0] = [1, 2, 3, 4, 5]               # Affectation de colonne (COW automatique)

# Conversion intelligente des formats MATLAB
A[:, 0] = [[1], [2], [3], [4], [5]]     # Accepte les formats colonne MATLAB
A[0, :] = [[1, 2, 3, 4, 5]]             # Accepte les formats ligne MATLAB
```

### Opérations matricielles

```python
A = Matrix.randn(3, 3)
B = Matrix.randn(3, 3)
v = Matrix.randn(3, 1)

# Opérations de base
C = A + B                                # Addition
C = A - B                                # Soustraction  
C = A * B                                # Produit matriciel
C = A * 2.5                              # Produit par scalaire
C = A / 2.0                              # Division par scalaire

# Opérations en place avec COW
A += B                                   # Addition en place (COW automatique)
A -= 2.5                                 # Soustraction en place (COW automatique)

# Opérations avancées
At = A.H                                 # Transposée conjuguée
abs_A = abs(A)                           # Valeur absolue (surcharge de abs())
neg_A = -A                               # Négation unaire
norm_2 = Matrix.norm(A)                  # Norme 2
norm_inf = Matrix.norm(A, np.inf)        # Norme infinie
trace_val = A.trace()                    # Trace
det_val = A.det()                        # Déterminant
rank_val = A.rank                        # Rang (propriété)
```

### Fonctions mathématiques complètes

```python
A = Matrix.rand(3, 3) * 2 - 1  # Valeurs dans [-1, 1]

# Fonctions trigonométriques
sin_A = A.sin() ou sin_A = Matrix.sin(A)
cos_A = A.cos()  
tan_A = A.tan()

# Fonctions trigonométriques inverses
asin_A = A.asin() ou asin_A = Matrix.asin(A)
acos_A = A.acos()
atan_A = A.atan()

# Fonction atan2 (à deux arguments)
angle = Matrix.atan2(A[:, 0], A[:, 1])   # Angle en coordonnées polaires

# Autres fonctions
abs_A = A.abs()  ou abs_A = Matrix.abs(A) # Valeur absolue
exp_A = A.exp()                           # Exponentielle

# Matrices complexes
A_complex = Matrix.randn(3, 3, dtype=complex)
real_part = A_complex.real               # Partie réelle
imag_part = A_complex.imag               # Partie imaginaire
```

### Statistiques et réductions MATLAB-like

```python
A = Matrix.randn(4, 5)

# Réductions par défaut (première dimension non-singleton, comme MATLAB)
max_val = A.max()  ou max_val = Matrix.max(A) # Maximum par colonne → vecteur ligne 1x5
min_val = A.min()                             # Minimum par colonne → vecteur ligne 1x5  
sum_val = A.sum()                             # Somme par colonne → vecteur ligne 1x5

# Réductions avec dimension spécifiée
max_rows = A.max(dim=1)                  # Maximum par ligne → vecteur colonne 4x1
sum_cols = A.sum(dim=0)                  # Somme par colonne (explicite)

# Pour un vecteur
v = Matrix.randn(5, 1)
max_scalar = v.max()                     # Retourne un scalaire

# Support des matrices creuses
sparse_A = Matrix.rand(1000, 1000, sparse=True, density=0.01)
sparse_sum = sparse_A.sum()              # Calcul efficace pour matrices creuses
```

### Matrices diagonales et diag()

```python
# Création de matrice diagonale depuis un vecteur
v = Matrix([1, 2, 3, 4])
D = Matrix.diag(v)                       # Matrice 4x4 avec v sur la diagonale
D_up = Matrix.diag(v, k=1)              # v sur la sur-diagonale

# Extraction de diagonale depuis une matrice
A = Matrix.randn(5, 5)
main_diag = Matrix.diag(A)               # Diagonale principale → vecteur colonne
upper_diag = Matrix.diag(A, k=1)         # Sur-diagonale → vecteur colonne
lower_diag = Matrix.diag(A, k=-1)        # Sous-diagonale → vecteur colonne
```

### Valeurs et vecteurs propres

```python
A = Matrix.randn(5, 5)
A = A + A.H  # Rendre la matrice hermitienne

# Toutes les valeurs propres (matrices denses)
eigenvals, eigenvecs = A.eig()

# Quelques valeurs propres (matrices denses ou creuses)
vals, vecs = A.eig(n=3, option='lm')     # 3 plus grandes en module
vals, vecs = A.eig(n=2, option='sm')     # 2 plus petites en module

# Pour matrices creuses (optimisé automatiquement)
A_sparse = Matrix.rand(1000, 1000, sparse=True, density=0.01)
A_sparse = A_sparse + A_sparse.H
sparse_vals, sparse_vecs = A_sparse.eig(n=10, option='lm')
```

### Résolution de systèmes linéaires

```python
A = Matrix.randn(5, 5)
b = Matrix.randn(5, 1)

# Résolution directe Ax = b
x = Matrix.backslash(A, b)               # Utilise la fonction globale
x = A.backslash(b)                       # Méthode de la classe

# Support automatique dense/creuse
A_sparse = Matrix.rand(1000, 1000, sparse=True, density=0.01)
b_sparse = Matrix.randn(1000, 1)
x_sparse = Matrix.backslash(A_sparse, b_sparse)  # Utilise spsolve automatiquement

# Fonction backslash globale avec gestion complète des formats
from MatLite import backslash
x = backslash(A, b)                      # Gestion automatique des types
```

## Matrices creuses optimisées

MatLite gère intelligemment les matrices creuses SciPy avec optimisations spécifiques :

```python
# Création efficace de matrices creuses
A = Matrix.eye(1000, sparse=True)                    # Matrice identité creuse
B = Matrix.rand(1000, 1000, sparse=True, density=0.01)  # Matrice aléatoire creuse

# Les opérations préservent la sparsité quand c'est bénéfique
C = A + B                                           # Résultat creuse
D = A * B                                           # Produit creuse optimisé
norm_val = A.norm(2)                                # Calcul SVD sparse pour norme 2

# Fonctions mathématiques sur matrices creuses
sin_sparse = A.sin()                                # Appliqué aux éléments non-nuls
exp_sparse = B.exp()                                # Optimisé pour la sparsité

# Statistiques pour matrices creuses
max_sparse = B.max()                                # Maximum efficace
rank_sparse = B.rank                                # Rang avec SVD sparse si nécessaire
```

## 🔧 Outils de diagnostic COW

MatLite fournit des outils pour analyser et optimiser l'utilisation mémoire :

```python
from MatLite import memory_usage_info, demonstrate_cow

# Création de plusieurs matrices avec partage COW
A = Matrix.rand(1000, 1000)
B = Matrix(A)  # Partage avec A
C = Matrix(A)  # Partage avec A
D = A[100:200, 100:200]  # Vue sur sous-région

# Analyse détaillée de l'utilisation mémoire
memory_usage_info([A, B, C, D])

# État COW individuel
print("État COW de A:", A.is_cow_active())
# Retourne: {'is_view': False, 'has_children': True, 'is_modified': False, 
#           'children_count': 3, 'has_parent': False}

# Démonstration complète du COW
demonstrate_cow()  # Exemple interactif des avantages COW
```

## Exemples d'utilisation

### Algorithme du Gradient Conjugué avec COW

```python
def cg(A, b, tol=1.0e-6):
    """Gradient conjugué optimisé avec COW"""
    sz = A.shape
    assert sz[0] == sz[1]
    
    x = Matrix.rand(sz[0], 1, random_state=42)
    r = b - A * x  # COW évite les copies inutiles
    p = Matrix(r)  # Partage initial avec r
    
    k = 0
    while Matrix.norm(r) > tol * Matrix.norm(b):
        k += 1
        a = (p.H * r) / (p.H * A * p)
        x += a * p  # Opération en place avec COW
        r = b - A * x
        beta = -(p.H * A * r) / (p.H * A * p)
        p = r + beta * p
    
    return x, k

# Test avec matrice définie positive
N = 100
A = Matrix.rand(N, N, random_state=42)
A = A + A.H + N * Matrix.eye(N)  # Rend A définie positive
b = Matrix.rand(N, 1, random_state=42)

x, iterations = cg(A, b)
print(f'Iterations: {iterations}, Résidu: {Matrix.norm(b - A * x, np.inf)}')
```

### Analyse spectrale avec matrices complexes

```python
# Matrice hermitienne complexe
A = Matrix.randn(4, 4, dtype=complex)
A = A + A.H  # Rendre hermitienne

# Calcul des valeurs propres (réelles pour matrice hermitienne)
eigenvals, eigenvecs = A.eig()
print("Valeurs propres:", eigenvals)
print("Parties réelles des vecteurs propres:")
print(eigenvecs.real)
print("Parties imaginaires des vecteurs propres:")
print(eigenvecs.imag)

# Vérification de l'orthogonalité
Q = eigenvecs
QHQ = Q.H * Q
print("Vérification orthogonalité (doit être ≈ I):")
print(abs(QHQ - Matrix.eye(4, dtype=complex)))
```

### Génération et analyse de données avec optimisation mémoire

```python
# Génération de grandes matrices avec COW
data = Matrix.randn(10000, 50)  # 100k points en 50D

# Création de vues avec COW pour analyse
train_data = data[:8000, :]     # Vue COW sur données d'entraînement
test_data = data[8000:, :]      # Vue COW sur données de test

# Statistiques (pas de copie mémoire grâce au COW)
train_means = train_data.sum(dim=0) / 8000    # Moyennes par colonne
train_stds = ((train_data - train_means).abs().sum(dim=0)) / 8000

# Normalisation (déclenche COW seulement si nécessaire)
normalized_train = (train_data - train_means) / train_stds
normalized_test = (test_data - train_means) / train_stds

print("Analyse mémoire après normalisation:")
memory_usage_info([data, train_data, test_data, normalized_train, normalized_test])
```

## Compatibilité MATLAB étendue

MatLite reproduit fidèlement le comportement de MATLAB avec des extensions :

| MATLAB | MatLite | Description |
|--------|---------|-------------|
| `A = [1 2; 3 4]` | `A = Matrix([[1, 2], [3, 4]])` | Création de matrice |
| `A(2,3)` | `A[1, 2]` | Indexation (base 0 en Python) |
| `A(:,2)` | `A[:, 1]` | Extraction de colonne |
| `A \ b` | `Matrix.backslash(A, b)` | Résolution système linéaire |
| `max(A)` | `A.max()` | Maximum par colonne |
| `max(A, [], 2)` | `A.max(dim=1)` | Maximum par ligne |
| `norm(A,2)` | `A.norm(2)` | Norme matricielle |
| `norm(A,'fro')` | `A.norm('fro')` | Norme de Frobenius |
| `eig(A)` | `A.eig()` | Valeurs/vecteurs propres |
| `diag(v)` | `Matrix.diag(v)` | Matrice diagonale |
| `diag(A,k)` | `Matrix.diag(A, k)` | Extraction de diagonale |
| `eye(n)` | `Matrix.eye(n)` | Matrice identité |
| `zeros(m,n)` | `Matrix.zeros(m, n)` | Matrice de zéros |
| `ones(m,n)` | `Matrix.ones(m, n)` | Matrice de uns |
| `rand(m,n)` | `Matrix.rand(m, n)` | Matrice aléatoire |
| `randn(m,n)` | `Matrix.randn(m, n)` | Matrice gaussienne |
| `sin(A)` | `A.sin()` | Sinus élément par élément |
| `abs(A)` | `A.abs()` ou `abs(A)` | Valeur absolue |
| `A += B` | `A += B` | **Addition en place (nouveau)** |
| `real(A)` | `A.real` | **Partie réelle (nouveau)** |
| `imag(A)` | `A.imag` | **Partie imaginaire (nouveau)** |
| `atan2(Y,X)` | `Matrix.atan2(Y, X)` | **Fonction atan2 (nouveau)** |

## Performance et optimisation mémoire

### Comparaison avec/sans COW

```python
import time

# Test de performance COW
A = Matrix.rand(2000, 2000)  # ~32MB

# Copie COW (instantanée)
start = time.time()
B = Matrix(A)
print(f"COW copy: {time.time() - start:.6f}s")  # ~0.000001s

# Copie forcée (lente)
start = time.time()
C = A.copy()
print(f"Force copy: {time.time() - start:.3f}s")  # ~0.1s

# Économie mémoire avec COW
matrices = [Matrix(A) for _ in range(10)]  # 10 "copies"
memory_usage_info(matrices)  # Montre le partage mémoire
```

### Recommandations d'utilisation

1. **Utilisez COW par défaut** : Les copies sont automatiquement optimisées
2. **Forcez la copie** si vous savez que vous allez beaucoup modifier : `A.copy()`
3. **Surveillez l'état COW** avec `is_cow_active()` pour le debugging
4. **Préférez les matrices creuses** pour les grandes matrices peu denses
5. **Utilisez les opérations en place** (`+=`, `-=`) pour éviter les allocations

## API de diagnostic avancée

```python
# Analyse détaillée d'une matrice
A = Matrix.rand(100, 100)
B = Matrix(A)

print("Informations COW détaillées:")
print(f"A: {A.is_cow_active()}")
print(f"B: {B.is_cow_active()}")

# Modification et impact sur le COW
B[0, 0] = 999
print(f"Après modification de B: {B.is_cow_active()}")
print(f"Impact sur A: {A.is_cow_active()}")

# Vérification de l'indépendance des données
print(f"A[0,0] = {A[0,0]} (inchangé)")
print(f"B[0,0] = {B[0,0]} (modifié)")
```

## Licence

Ce projet est sous licence MIT.

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :

1. Forker le projet
2. Créer une branche pour votre fonctionnalité (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Commiter vos changements (`git commit -am 'Ajoute une nouvelle fonctionnalité'`)
4. Pousser vers la branche (`git push origin feature/nouvelle-fonctionnalite`)
5. Créer une Pull Request

### Zones de contribution prioritaires

- Extension des fonctions mathématiques
- Optimisations pour matrices très creuses
- Support de nouveaux formats de matrices
- Amélioration des algorithmes de diagnostic COW
- Documentation et exemples supplémentaires

## Contact

Pour toute question ou suggestion :
- **Auteur :** Jean-Christophe Toussaint
- **Affiliation :** Grenoble-INP Phelma

---

*MatLite - Bringing MATLAB syntax to Python's scientific computing ecosystem with advanced memory optimization*