import numpy as np
import matplotlib.pyplot as plt
from MatLite import Matrix

def vandermonde_matrix(points, N):
    """
    Crée une matrice de Vandermonde de taille N x N à partir des points donnés.
    
    V[i,j] = points[i]^(j-1) pour i,j = 0, 1, ..., N-1
    
    Args:
        points: vecteur des points (peut être Matrix, np.array, ou liste)
        N: taille de la matrice carrée
    
    Returns:
        Matrix: Matrice de Vandermonde N x N
    """
    if isinstance(points, Matrix):
        pts = points.data.ravel()
    else:
        pts = np.array(points).ravel()
    
    if len(pts) < N:
        raise ValueError(f"Il faut au moins {N} points pour créer une matrice {N}x{N}")
    
    # Prendre les N premiers points
    pts = pts[:N]
    
    # Créer la matrice de Vandermonde
    V_data = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            V_data[i, j] = pts[i] ** j
    
    return Matrix(V_data)

def theoretical_vandermonde_det(points):
    """
    Calcule le déterminant théorique d'une matrice de Vandermonde.
    
    Pour une matrice de Vandermonde avec points [x0, x1, ..., x_{n-1}],
    le déterminant est : ∏_{0 ≤ i < j ≤ n-1} (x_j - x_i)
    
    Args:
        points: vecteur des points
        
    Returns:
        float: déterminant théorique
    """
    if isinstance(points, Matrix):
        pts = points.data.ravel()
    else:
        pts = np.array(points).ravel()
    
    n = len(pts)
    det_theoretical = 1.0
    
    for i in range(n):
        for j in range(i + 1, n):
            det_theoretical *= (pts[j] - pts[i])
    
    return det_theoretical

def test_vandermonde_determinant(N_values, test_type="linspace"):
    """
    Teste le calcul du déterminant de Vandermonde pour différentes tailles.
    
    Args:
        N_values: liste des tailles à tester
        test_type: type de points à utiliser ("linspace", "random", "integers")
    """
    print(f"=== Test matrice de Vandermonde (type: {test_type}) ===")
    print(f"{'N':>3} | {'Det calculé':>15} | {'Det théorique':>15} | {'Erreur relative':>15}")
    print("-" * 65)
    
    results = []
    
    for N in N_values:
        # Génération des points selon le type
        if test_type == "linspace":
            # Points équidistants entre 0 et 1
            points = np.linspace(0, 1, N)
        elif test_type == "random":
            # Points aléatoires entre 0 et 1
            np.random.seed(42)  # Pour la reproductibilité
            points = np.random.rand(N)
        elif test_type == "integers":
            # Points entiers consécutifs
            points = np.arange(1, N + 1, dtype=float)
        else:
            raise ValueError("test_type doit être 'linspace', 'random', ou 'integers'")
        
        # Création de la matrice de Vandermonde
        V = vandermonde_matrix(points, N)
        
        # Calcul du déterminant avec MatLite
        det_calculated = V.det()
        
        # Calcul du déterminant théorique
        det_theoretical = theoretical_vandermonde_det(points)
        
        # Erreur relative
        if abs(det_theoretical) > 1e-15:
            relative_error = abs(det_calculated - det_theoretical) / abs(det_theoretical)
        else:
            relative_error = abs(det_calculated - det_theoretical)
        
        print(f"{N:>3} | {det_calculated:>15.6e} | {det_theoretical:>15.6e} | {relative_error:>15.6e}")
        
        results.append({
            'N': N,
            'det_calculated': det_calculated,
            'det_theoretical': det_theoretical,
            'relative_error': relative_error,
            'points': points.copy()
        })
    
    return results

def visualize_vandermonde_properties(N=5):
    """
    Visualise les propriétés d'une matrice de Vandermonde.
    """
    # Création d'une matrice de Vandermonde avec des points équidistants
    points = np.linspace(0.1, 1.0, N)  # Éviter 0 pour éviter des colonnes nulles
    V = vandermonde_matrix(points, N)
    
    print(f"\n=== Analyse matrice de Vandermonde {N}x{N} ===")
    print(f"Points utilisés: {points}")
    print(f"Matrice V:")
    print(V)
    print(f"Déterminant: {V.det():.6e}")
    print(f"Rang: {V.rank}")
    print(f"Nombre de condition (norme 2): {Matrix.norm(V) * Matrix.norm(Matrix.backslash(V, Matrix.eye(N))):.2e}")
    
    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Graphique 1: Matrice de Vandermonde
    im1 = axes[0].imshow(V.data, cmap='RdBu', aspect='auto')
    axes[0].set_title(f'Matrice de Vandermonde {N}×{N}')
    axes[0].set_xlabel('Colonne (puissance)')
    axes[0].set_ylabel('Ligne (point)')
    plt.colorbar(im1, ax=axes[0])
    
    # Graphique 2: Colonnes de la matrice (polynômes)
    x_plot = np.linspace(0, 1, 100)
    for j in range(min(N, 5)):  # Limite à 5 courbes pour la lisibilité
        y_plot = x_plot ** j
        axes[1].plot(x_plot, y_plot, label=f'x^{j}')
    
    # Marquer les points de la matrice
    for i, point in enumerate(points):
        axes[1].axvline(point, color='red', alpha=0.3, linestyle='--')
        if i == 0:
            axes[1].axvline(point, color='red', alpha=0.3, linestyle='--', label='Points Vandermonde')
    
    axes[1].set_title('Polynômes de base et points')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('x^j')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def condition_number_analysis():
    """
    Analyse du conditionnement des matrices de Vandermonde.
    """
    N_values = range(2, 12)
    conditions = []
    
    print("\n=== Analyse du conditionnement ===")
    print(f"{'N':>3} | {'Condition':>12} | {'Status':>15}")
    print("-" * 35)
    
    for N in N_values:
        points = np.linspace(0.1, 0.9, N)  # Points dans [0.1, 0.9]
        V = vandermonde_matrix(points, N)
        
        try:
            # Calcul du nombre de condition (κ = ||A|| * ||A^(-1)||)
            V_inv = Matrix.backslash(V, Matrix.eye(N))
            condition = Matrix.norm(V) * Matrix.norm(V_inv)
            status = "Bien conditionné" if condition < 1e12 else "Mal conditionné"
            
            print(f"{N:>3} | {condition:>12.2e} | {status:>15}")
            conditions.append(condition)
            
        except Exception as e:
            print(f"{N:>3} | {'Erreur':>12} | {str(e)[:15]:>15}")
            conditions.append(np.inf)
    
    # Graphique du conditionnement
    plt.figure(figsize=(10, 6))
    valid_conditions = [c for c in conditions if c != np.inf]
    valid_N = [N for N, c in zip(N_values, conditions) if c != np.inf]
    
    plt.semilogy(valid_N, valid_conditions, 'o-', linewidth=2, markersize=8)
    plt.axhline(y=1e12, color='red', linestyle='--', alpha=0.7, label='Seuil mal conditionné')
    plt.xlabel('Taille N de la matrice')
    plt.ylabel('Nombre de condition')
    plt.title('Conditionnement des matrices de Vandermonde')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Test avec différentes tailles
    N_values = [2, 3, 4, 5, 6, 7, 8]
    
    # Test avec points équidistants
    print("=" * 70)
    results_linspace = test_vandermonde_determinant(N_values, "linspace")
    
    # Test avec points entiers
    print("\n" + "=" * 70)
    results_integers = test_vandermonde_determinant(N_values, "integers")
    
    # Test avec points aléatoires
    print("\n" + "=" * 70)
    results_random = test_vandermonde_determinant(N_values, "random")
    
    # Exemple détaillé pour N=4
    print("\n" + "=" * 70)
    print("EXEMPLE DÉTAILLÉ: Matrice de Vandermonde 4×4")
    
    # Points distincts pour avoir un déterminant non-nul
    points = np.array([1., 2., 3., 4.])
    V = vandermonde_matrix(points, 4)
    
    print(f"Points: {points}")
    print(f"Matrice de Vandermonde:")
    print(V)
    
    det_calc = V.det()
    det_theo = theoretical_vandermonde_det(points)
    
    print(f"\nDéterminant calculé (MatLite): {det_calc}")
    print(f"Déterminant théorique: {det_theo}")
    print(f"Erreur absolue: {abs(det_calc - det_theo)}")
    print(f"Erreur relative: {abs(det_calc - det_theo) / abs(det_theo):.2e}")
    
    # Vérification : le déterminant théorique pour points [1,2,3,4] est:
    # (2-1)(3-1)(4-1)(3-2)(4-2)(4-3) = 1×2×3×1×2×1 = 12
    print(f"\nVérification manuelle pour [1,2,3,4]:")
    print(f"(2-1)×(3-1)×(4-1)×(3-2)×(4-2)×(4-3) = 1×2×3×1×2×1 = 12")
    
    # Visualisation des propriétés
    visualize_vandermonde_properties(N=6)
    
    # Analyse du conditionnement
    condition_number_analysis()