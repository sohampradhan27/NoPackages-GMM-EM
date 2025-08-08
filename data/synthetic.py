"""
Synthetic data generation for Gaussian Mixture Model testing.

This module provides functions to generate synthetic datasets with known
Gaussian mixture components for testing and demonstration purposes.
"""

import numpy as np
from typing import List, Tuple, Optional, Union


def generate_clusters(
    n_samples: Union[int, List[int]],
    centers: List[np.ndarray],
    covariances: List[np.ndarray],
    weights: Optional[List[float]] = None,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data from a mixture of multivariate Gaussians.
    
    Parameters
    ----------
    n_samples : int or list of int
        Total number of samples to generate, or list of samples per component.
        If int, samples are distributed according to weights.
    centers : list of array-like
        List of mean vectors for each Gaussian component.
        Each center should be shape (n_features,).
    covariances : list of array-like
        List of covariance matrices for each Gaussian component.
        Each covariance should be shape (n_features, n_features).
    weights : list of float, optional
        Mixing weights for each component. Must sum to 1.
        If None, components are weighted equally.
    random_state : int, optional
        Random seed for reproducible results.
        
    Returns
    -------
    X : array-like, shape (n_samples, n_features)
        Generated data points.
    y : array-like, shape (n_samples,)
        True cluster labels for each data point.
        
    Examples
    --------
    >>> centers = [np.array([0, 0]), np.array([3, 3])]
    >>> covariances = [np.eye(2), 0.5 * np.eye(2)]
    >>> X, y = generate_clusters(200, centers, covariances)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    centers = [np.asarray(c) for c in centers]
    covariances = [np.asarray(cov) for cov in covariances]
    
    n_components = len(centers)
    
    # Validate inputs
    if len(covariances) != n_components:
        raise ValueError("Number of centers and covariances must match")
        
    if weights is not None:
        if len(weights) != n_components:
            raise ValueError("Number of weights must match number of components")
        if not np.isclose(sum(weights), 1.0):
            raise ValueError("Weights must sum to 1")
    else:
        weights = [1.0 / n_components] * n_components
    
    # Check dimensions are consistent
    n_features = len(centers[0])
    for i, (center, cov) in enumerate(zip(centers, covariances)):
        if len(center) != n_features:
            raise ValueError(f"All centers must have same dimension. "
                           f"Component {i} has dimension {len(center)}, expected {n_features}")
        if cov.shape != (n_features, n_features):
            raise ValueError(f"Covariance {i} has shape {cov.shape}, "
                           f"expected ({n_features}, {n_features})")
    
    # Determine samples per component
    if isinstance(n_samples, int):
        # Distribute samples according to weights
        samples_per_component = np.random.multinomial(n_samples, weights)
    else:
        if len(n_samples) != n_components:
            raise ValueError("Length of n_samples list must match number of components")
        samples_per_component = n_samples
        n_samples = sum(n_samples)
    
    # Generate data
    X = []
    y = []
    
    for k in range(n_components):
        n_k = samples_per_component[k]
        if n_k > 0:
            # Generate samples from multivariate normal
            X_k = np.random.multivariate_normal(
                centers[k], covariances[k], size=n_k
            )
            y_k = np.full(n_k, k)
            
            X.append(X_k)
            y.append(y_k)
    
    # Concatenate and shuffle
    X = np.vstack(X)
    y = np.concatenate(y)
    
    # Shuffle data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y


def generate_2d_blobs(
    n_samples: int = 300,
    n_components: int = 3,
    cluster_std: float = 1.0,
    center_box: Tuple[float, float] = (-10.0, 10.0),
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 2D blob-like clusters for easy visualization.
    
    Parameters
    ----------
    n_samples : int, default=300
        Total number of samples.
    n_components : int, default=3
        Number of clusters.
    cluster_std : float, default=1.0
        Standard deviation of clusters.
    center_box : tuple of float, default=(-10.0, 10.0)
        Bounding box for cluster centers.
    random_state : int, optional
        Random seed.
        
    Returns
    -------
    X : array-like, shape (n_samples, 2)
        Generated 2D data points.
    y : array-like, shape (n_samples,)
        True cluster labels.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate random centers
    centers = []
    for _ in range(n_components):
        center = np.random.uniform(center_box[0], center_box[1], 2)
        centers.append(center)
    
    # Create isotropic covariances
    covariances = [cluster_std**2 * np.eye(2) for _ in range(n_components)]
    
    return generate_clusters(n_samples, centers, covariances, random_state=random_state)


def generate_elongated_clusters(
    n_samples: int = 300,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 2D data with elongated, overlapping clusters.
    
    This creates a challenging dataset for testing GMM performance
    with non-spherical clusters.
    
    Parameters
    ----------
    n_samples : int, default=300
        Total number of samples.
    random_state : int, optional
        Random seed.
        
    Returns
    -------
    X : array-like, shape (n_samples, 2)
        Generated 2D data points.
    y : array-like, shape (n_samples,)
        True cluster labels.
    """
    centers = [
        np.array([-2, 0]),
        np.array([2, 1]),
        np.array([0, -3])
    ]
    
    # Create elongated covariances
    covariances = [
        np.array([[2.0, 0.8], [0.8, 0.5]]),  # Elongated diagonal
        np.array([[0.8, -0.3], [-0.3, 1.5]]), # Negative correlation  
        np.array([[1.2, 0.0], [0.0, 0.3]])    # Axis-aligned elongation
    ]
    
    weights = [0.4, 0.35, 0.25]
    
    return generate_clusters(
        n_samples, centers, covariances, weights, random_state=random_state
    )


def generate_concentric_circles(
    n_samples: int = 300,
    noise: float = 0.1,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate concentric circular clusters.
    
    This creates a dataset that is challenging for GMM since the clusters
    are not well-separated by linear boundaries.
    
    Parameters
    ----------
    n_samples : int, default=300
        Total number of samples.
    noise : float, default=0.1
        Standard deviation of noise added to the circles.
    random_state : int, optional
        Random seed.
        
    Returns
    -------
    X : array-like, shape (n_samples, 2)
        Generated 2D data points.
    y : array-like, shape (n_samples,)
        True cluster labels (0=inner, 1=outer).
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_inner = n_samples // 2
    n_outer = n_samples - n_inner
    
    # Inner circle
    angles_inner = np.random.uniform(0, 2*np.pi, n_inner)
    radii_inner = np.random.normal(1.0, noise, n_inner)
    X_inner = np.column_stack([
        radii_inner * np.cos(angles_inner),
        radii_inner * np.sin(angles_inner)
    ])
    y_inner = np.zeros(n_inner)
    
    # Outer circle
    angles_outer = np.random.uniform(0, 2*np.pi, n_outer)
    radii_outer = np.random.normal(3.0, noise, n_outer)
    X_outer = np.column_stack([
        radii_outer * np.cos(angles_outer),
        radii_outer * np.sin(angles_outer)
    ])
    y_outer = np.ones(n_outer)
    
    # Combine and shuffle
    X = np.vstack([X_inner, X_outer])
    y = np.concatenate([y_inner, y_outer])
    
    indices = np.random.permutation(len(X))
    return X[indices], y[indices].astype(int)


if __name__ == "__main__":
    """Demonstration of synthetic data generation functions."""
    
    # Example 1: Simple blob clusters
    print("Generating simple blob clusters...")
    X1, y1 = generate_2d_blobs(n_samples=300, n_components=3, random_state=42)
    print(f"Generated {len(X1)} samples with {len(np.unique(y1))} clusters")
    print(f"Cluster sizes: {np.bincount(y1)}")
    
    # Example 2: Elongated clusters
    print("\nGenerating elongated clusters...")
    X2, y2 = generate_elongated_clusters(n_samples=300, random_state=42)
    print(f"Generated {len(X2)} samples with {len(np.unique(y2))} clusters")
    print(f"Cluster sizes: {np.bincount(y2)}")
    
    # Example 3: Custom clusters
    print("\nGenerating custom clusters...")
    centers = [np.array([0, 0]), np.array([5, 5]), np.array([-3, 4])]
    covariances = [
        np.array([[1, 0.5], [0.5, 1]]),
        np.array([[2, 0], [0, 0.5]]),
        np.array([[0.8, -0.2], [-0.2, 1.5]])
    ]
    weights = [0.5, 0.3, 0.2]
    
    X3, y3 = generate_clusters(
        n_samples=400, 
        centers=centers, 
        covariances=covariances, 
        weights=weights,
        random_state=42
    )
    print(f"Generated {len(X3)} samples with {len(np.unique(y3))} clusters")
    print(f"Cluster sizes: {np.bincount(y3)}")
    
    # Example 4: Concentric circles
    print("\nGenerating concentric circles...")
    X4, y4 = generate_concentric_circles(n_samples=300, random_state=42)
    print(f"Generated {len(X4)} samples with {len(np.unique(y4))} clusters")
    print(f"Cluster sizes: {np.bincount(y4)}")
