# MatLite

Une bibliothèque Python offrant une interface MATLAB-like pour les opérations matricielles avec support natif des matrices denses et creuses.

## Auteur

**Jean-Christophe Toussaint**  
Grenoble-INP Phelma

## Description

MatLite fournit une classe `Matrix` qui encapsule les matrices NumPy et SciPy.sparse, offrant une syntaxe familière aux utilisateurs de MATLAB tout en bénéficiant de la performance des bibliothèques Python scientifiques.

### Caractéristiques principales

- ✅ Interface MATLAB-like intuitive
- ✅ Support des matrices denses (NumPy) et creuses (SciPy.sparse)
- ✅ Indexation et slicing avancés compatible MATLAB
- ✅ Opérations matricielles optimisées
- ✅ Fonctions mathématiques élémentaires
- ✅ Générateurs de matrices (zeros, ones, eye, rand, randn)
- ✅ Résolution de systèmes linéaires avec `backslash`

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
x = Matrix.backslash(A.H * A, A.H * b)    # Moindres carrés
```

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
```

### Indexation et slicing

```python
A = Matrix.rand(5, 5)

# Accès aux éléments
element = A[2, 3]                         # Élément (2,3)
A[1, 1] = 42                             # Modification

# Slicing
row = A[2, :]                            # Ligne 2 complète  
col = A[:, 1]                            # Colonne 1 complète
submatrix = A[1:4, 0:3]                  # Sous-matrice 3x3

# Indexation avancée
selected = A[[0, 2, 4], [1, 3]]          # Éléments spécifiques
A[:, 0] = [1, 2, 3, 4, 5]               # Affectation de colonne
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

# Opérations avancées
At = A.H                                 # Transposée conjuguée
norm_2 = Matrix.norm(A)                  # Norme 2
norm_inf = Matrix.norm(A, np.inf)        # Norme infinie
trace_val = Matrix.trace(A)              # Trace
det_val = Matrix.det(A)                  # Déterminant
rank_val = A.rank                        # Rang
```

### Fonctions mathématiques

```python
A = Matrix.rand(3, 3)

# Fonctions trigonométriques
sin_A = Matrix.sin(A)
cos_A = Matrix.cos(A)  
tan_A = Matrix.tan(A)
asin_A = Matrix.asin(A)
acos_A = Matrix.acos(A)
atan_A = Matrix.atan(A)

# Autres fonctions
abs_A = Matrix.abs(A)                    # Valeur absolue
exp_A = Matrix.exp(A)                    # Exponentielle

# Fonction atan2
angle = Matrix.atan2(A[:, 0], A[:, 1])   # Angle en coordonnées polaires
```

### Statistiques et réductions

```python
A = Matrix.randn(4, 5)

# Réductions par défaut (MATLAB-like)
max_val = Matrix.max(A)                  # Maximum par colonne → vecteur ligne 1x5
min_val = Matrix.min(A)                  # Minimum par colonne → vecteur ligne 1x5  
sum_val = Matrix.sum(A)                  # Somme par colonne → vecteur ligne 1x5

# Réductions avec dimension spécifiée
max_rows = Matrix.max(A, dim=1)          # Maximum par ligne → vecteur colonne 4x1
sum_all = Matrix.sum(A, dim=0)           # Somme par colonne (explicite)

# Pour un vecteur
v = Matrix.randn(5, 1)
max_scalar = Matrix.max(v)               # Retourne un scalaire
```

### Valeurs et vecteurs propres

```python
A = Matrix.randn(5, 5)
A = A + A.H  # Rendre la matrice hermitienne

# Toutes les valeurs propres (matrices denses)
eigenvals, eigenvecs = Matrix.eig(A)

# Quelques valeurs propres (matrices denses ou creuses)
vals, vecs = Matrix.eig(A, n=3, option='lm')  # 3 plus grandes en module
vals, vecs = Matrix.eig(A, n=2, option='sm')  # 2 plus petites en module
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
```

## Matrices creuses

MatLite gère transparentement les matrices creuses SciPy :

```python
# Création d'une matrice creuse
A = Matrix.eye(1000, sparse=True)                    # Matrice identité creuse
B = Matrix.rand(1000, 1000, sparse=True, density=0.01)  # Matrice aléatoire creuse

# Les opérations préservent la sparsité quand c'est bénéfique
C = A + B                                           # Résultat creuse
D = A * B                                           # Produit creuse
norm_val = Matrix.norm(A, 2)                        # Calcul efficace pour creuse

# Conversion automatique si nécessaire
dense_result = Matrix.sin(A)                        # Peut devenir dense selon l'opération
```

## Exemples d'utilisation

### Résolution d'un système linéaire

```python
# Système Ax = b
A = Matrix([[3, 2], [1, 2]])
b = Matrix([[1], [2]])

x = Matrix.backslash(A, b)
print("Solution:", x)
print("Vérification Ax =", A * x)
```

### Analyse spectrale

```python
# Matrice symétrique aléatoire
A = Matrix.randn(4, 4)
A = A + A.H  # Rendre hermitienne

# Calcul des valeurs propres
eigenvals, eigenvecs = Matrix.eig(A)
print("Valeurs propres:", eigenvals)
print("Trace:", Matrix.trace(A))
print("Vérification (somme des val. propres):", np.sum(eigenvals))
```

### Génération et analyse de données

```python
# Génération de données
data = Matrix.randn(100, 3)  # 100 points en 3D

# Statistiques
means = Matrix.sum(data, dim=0) / 100    # Moyennes par colonne
maxs = Matrix.max(data, dim=0)           # Maximums par colonne
mins = Matrix.min(data, dim=0)           # Minimums par colonne

print("Moyennes:", means)
print("Étendues:", maxs - mins)
```

## Compatibilité MATLAB

MatLite vise à reproduire fidèlement le comportement de MATLAB :

| MATLAB | MatLite | Description |
|--------|---------|-------------|
| `A = [1 2; 3 4]` | `A = Matrix([[1, 2], [3, 4]])` | Création de matrice |
| `A(2,3)` | `A[1, 2]` | Indexation (attention : base 0 en Python) |
| `A(:,2)` | `A[:, 1]` | Extraction de colonne |
| `A \ b` | `Matrix.backslash(A, b)` | Résolution système linéaire |
| `max(A)` | `Matrix.max(A)` | Maximum par colonne |
| `norm(A,2)` | `Matrix.norm(A, 2)` | Norme matricielle |
| `eig(A)` | `Matrix.eig(A)` | Valeurs/vecteurs propres |
| `diag(v)` | `Matrix.diag(v)` | Matrice diagonale |
| `eye(n)` | `Matrix.eye(n)` | Matrice identité |
| `zeros(m,n)` | `Matrix.zeros(m, n)` | Matrice de zéros |
| `ones(m,n)` | `Matrix.ones(m, n)` | Matrice de uns |
| `rand(m,n)` | `Matrix.rand(m, n)` | Matrice aléatoire |
| `sin(A)` | `Matrix.sin(A)` | Sinus élément par élément |

## Licence

Ce projet est sous licence MIT.

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :

1. Forker le projet
2. Créer une branche pour votre fonctionnalité (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Commiter vos changements (`git commit -am 'Ajoute une nouvelle fonctionnalité'`)
4. Pousser vers la branche (`git push origin feature/nouvelle-fonctionnalite`)
5. Créer une Pull Request

## Contact

Pour toute question ou suggestion :
- **Auteur :** Jean-Christophe Toussaint
- **Affiliation :** Grenoble-INP Phelma
---

*MatLite - Bringing MATLAB syntax to Python's scientific computing ecosystem*