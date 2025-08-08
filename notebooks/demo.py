"""
Gaussian Mixture Model with Expectation-Maximization Demo

This script demonstrates the implementation of a Gaussian Mixture Model (GMM) 
using the Expectation-Maximization (EM) algorithm from scratch.

Run this script section by section, or convert to Jupyter notebook.
"""

import sys
import os
sys.path.append(os.path.join('..', 'algorithm'))
sys.path.append(os.path.join('..', 'data'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import ListedColormap

from gmm_em import GaussianMixtureEM
from synthetic import generate_2d_blobs, generate_elongated_clusters, generate_concentric_circles

# Set style for better plots
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 60)
print("GAUSSIAN MIXTURE MODEL DEMONSTRATION")
print("=" * 60)

# =============================================================================
# 1. GENERATE SYNTHETIC DATA
# =============================================================================
print("\n1. GENERATING SYNTHETIC DATA")
print("-" * 30)

# Generate elongated clusters for a challenging dataset
X, y_true = generate_elongated_clusters(n_samples=300, random_state=42)

print(f"Generated {len(X)} samples with {len(np.unique(y_true))} true clusters")
print(f"Data shape: {X.shape}")
print(f"True cluster sizes: {np.bincount(y_true)}")

# Plot the true clusters
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green', 'purple', 'orange']
for k in range(len(np.unique(y_true))):
    mask = y_true == k
    plt.scatter(X[mask, 0], X[mask, 1], c=colors[k], alpha=0.7, 
               label=f'True Cluster {k}', s=50)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('True Cluster Assignments')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# =============================================================================
# 2. FIT GAUSSIAN MIXTURE MODEL
# =============================================================================
print("\n2. FITTING GAUSSIAN MIXTURE MODEL")
print("-" * 35)

# Initialize and fit the GMM
n_components = 3
gmm = GaussianMixtureEM(
    n_components=n_components,
    max_iter=100,
    tol=1e-6,
    random_state=42
)

print("Fitting Gaussian Mixture Model...")
gmm.fit(X)

print(f"Converged: {gmm.converged_}")
print(f"Number of iterations: {gmm.n_iter_}")
print(f"Final log-likelihood: {gmm.log_likelihood_history_[-1]:.4f}")
print(f"Mixing weights: {gmm.weights_.round(3)}")

# =============================================================================
# 3. VISUALIZE RESULTS
# =============================================================================
print("\n3. VISUALIZING RESULTS")
print("-" * 25)

# Get predictions
y_pred = gmm.predict(X)
probabilities = gmm.predict_proba(X)

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot predicted clusters
for k in range(n_components):
    mask = y_pred == k
    axes[0].scatter(X[mask, 0], X[mask, 1], c=colors[k], alpha=0.7, 
                   label=f'Predicted Cluster {k}', s=50)

axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].set_title('GMM Predicted Clusters')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot soft assignments (probabilities)
max_probs = np.max(probabilities, axis=1)
scatter = axes[1].scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', 
                         alpha=max_probs, s=50)

axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')
axes[1].set_title('Soft Assignments (Opacity = Confidence)')
axes[1].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[1])

plt.tight_layout()
plt.show()

# Print cluster statistics
print(f"Predicted cluster sizes: {np.bincount(y_pred)}")
print(f"Average prediction confidence: {np.mean(np.max(probabilities, axis=1)):.3f}")

# =============================================================================
# 4. VISUALIZE GAUSSIAN COMPONENTS
# =============================================================================
print("\n4. VISUALIZING GAUSSIAN COMPONENTS")
print("-" * 35)

def plot_gaussian_ellipse(mean, cov, ax, color, alpha=0.3, n_std=1):
    """Plot confidence ellipse for a 2D Gaussian distribution."""
    # Compute eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    
    # Get the angle of the ellipse
    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    
    # Width and height are 2 * sqrt(eigenvalue) * n_std
    width, height = 2 * n_std * np.sqrt(eigenvals)
    
    # Create and add ellipse
    ellipse = Ellipse(mean, width, height, angle=angle, 
                     facecolor=color, alpha=alpha, edgecolor=color, linewidth=2)
    ax.add_patch(ellipse)
    
    # Plot center
    ax.plot(mean[0], mean[1], 'o', color=color, markersize=8, markeredgecolor='black')

# Create the plot
plt.figure(figsize=(12, 8))

# Plot data points colored by prediction
for k in range(n_components):
    mask = y_pred == k
    plt.scatter(X[mask, 0], X[mask, 1], c=colors[k], alpha=0.6, 
               label=f'Cluster {k}', s=30)

# Plot Gaussian ellipses at 1 and 2 standard deviations
for k in range(n_components):
    # 1-sigma ellipse (darker)
    plot_gaussian_ellipse(gmm.means_[k], gmm.covariances_[k], plt.gca(), 
                         colors[k], alpha=0.4, n_std=1)
    # 2-sigma ellipse (lighter)
    plot_gaussian_ellipse(gmm.means_[k], gmm.covariances_[k], plt.gca(), 
                         colors[k], alpha=0.2, n_std=2)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('GMM Components with Confidence Ellipses\n(Dark: 1σ, Light: 2σ)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()

# Print component parameters
print("\nComponent Parameters:")
for k in range(n_components):
    print(f"\nComponent {k}:")
    print(f"  Weight: {gmm.weights_[k]:.3f}")
    print(f"  Mean: [{gmm.means_[k][0]:.3f}, {gmm.means_[k][1]:.3f}]")
    print(f"  Covariance:")
    print(f"    [{gmm.covariances_[k][0,0]:.3f}, {gmm.covariances_[k][0,1]:.3f}]")
    print(f"    [{gmm.covariances_[k][1,0]:.3f}, {gmm.covariances_[k][1,1]:.3f}]")

# =============================================================================
# 5. CONVERGENCE ANALYSIS
# =============================================================================
print("\n5. CONVERGENCE ANALYSIS")
print("-" * 25)

# Plot log-likelihood vs iteration
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(gmm.log_likelihood_history_) + 1), 
         gmm.log_likelihood_history_, 'b-o', linewidth=2, markersize=6)
plt.xlabel('Iteration')
plt.ylabel('Log-Likelihood')
plt.title('EM Algorithm Convergence')
plt.grid(True, alpha=0.3)

# Add convergence info
if len(gmm.log_likelihood_history_) > 1:
    final_ll = gmm.log_likelihood_history_[-1]
    initial_ll = gmm.log_likelihood_history_[0]
    improvement = final_ll - initial_ll
    plt.text(0.02, 0.98, f'Final LL: {final_ll:.2f}\nImprovement: {improvement:.2f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

# Show convergence statistics
if len(gmm.log_likelihood_history_) > 1:
    print(f"Initial log-likelihood: {gmm.log_likelihood_history_[0]:.4f}")
    print(f"Final log-likelihood: {gmm.log_likelihood_history_[-1]:.4f}")
    print(f"Total improvement: {gmm.log_likelihood_history_[-1] - gmm.log_likelihood_history_[0]:.4f}")
    
    if len(gmm.log_likelihood_history_) > 1:
        improvements = np.diff(gmm.log_likelihood_history_)
        print(f"Average per-iteration improvement: {np.mean(improvements):.6f}")
        print(f"Final iteration improvement: {improvements[-1]:.6f}")

# =============================================================================
# 6. MODEL COMPARISON WITH DIFFERENT NUMBERS OF COMPONENTS
# =============================================================================
print("\n6. MODEL COMPARISON")
print("-" * 20)

def compute_aic_bic(gmm, X):
    """Compute AIC and BIC for model selection."""
    n_samples, n_features = X.shape
    n_params = gmm.n_components * (n_features + n_features * (n_features + 1) // 2) + gmm.n_components - 1
    
    log_likelihood = gmm.log_likelihood_history_[-1]
    aic = 2 * n_params - 2 * log_likelihood
    bic = n_params * np.log(n_samples) - 2 * log_likelihood
    
    return aic, bic

# Test different numbers of components
n_components_range = range(1, 7)
results = []

for n_comp in n_components_range:
    print(f"Fitting GMM with {n_comp} components...")
    
    gmm_test = GaussianMixtureEM(
        n_components=n_comp,
        max_iter=100,
        random_state=42
    )
    
    gmm_test.fit(X)
    aic, bic = compute_aic_bic(gmm_test, X)
    
    results.append({
        'n_components': n_comp,
        'log_likelihood': gmm_test.log_likelihood_history_[-1],
        'aic': aic,
        'bic': bic,
        'converged': gmm_test.converged_
    })

# Plot results
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

n_comps = [r['n_components'] for r in results]
log_likelihoods = [r['log_likelihood'] for r in results]
aics = [r['aic'] for r in results]
bics = [r['bic'] for r in results]

# Log-likelihood
axes[0].plot(n_comps, log_likelihoods, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Components')
axes[0].set_ylabel('Log-Likelihood')
axes[0].set_title('Log-Likelihood vs Components')
axes[0].grid(True, alpha=0.3)

# AIC
axes[1].plot(n_comps, aics, 'ro-', linewidth=2, markersize=8)
best_aic_idx = np.argmin(aics)
axes[1].axvline(n_comps[best_aic_idx], color='red', linestyle='--', alpha=0.7)
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('AIC')
axes[1].set_title(f'AIC vs Components (Best: {n_comps[best_aic_idx]})')
axes[1].grid(True, alpha=0.3)

# BIC
axes[2].plot(n_comps, bics, 'go-', linewidth=2, markersize=8)
best_bic_idx = np.argmin(bics)
axes[2].axvline(n_comps[best_bic_idx], color='green', linestyle='--', alpha=0.7)
axes[2].set_xlabel('Number of Components')
axes[2].set_ylabel('BIC')
axes[2].set_title(f'BIC vs Components (Best: {n_comps[best_bic_idx]})')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print results table
print("\nModel Comparison Results:")
print("Components | Log-Likelihood |     AIC     |     BIC     | Converged")
print("-" * 65)
for r in results:
    print(f"    {r['n_components']}      |    {r['log_likelihood']:7.2f}    | {r['aic']:7.2f}   | {r['bic']:7.2f}   |    {r['converged']}")

print(f"\nBest number of components:")
print(f"  AIC: {n_comps[best_aic_idx]} components")
print(f"  BIC: {n_comps[best_bic_idx]} components")
print(f"  True: 3 components")

# =============================================================================
# 7. TESTING ON DIFFERENT DATASET TYPES
# =============================================================================
print("\n7. TESTING ON DIFFERENT DATASETS")
print("-" * 35)

# Test on different dataset types
datasets = [
    ('Simple Blobs', generate_2d_blobs(n_samples=300, n_components=3, random_state=42)),
    ('Elongated Clusters', generate_elongated_clusters(n_samples=300, random_state=42)),
    ('Concentric Circles', generate_concentric_circles(n_samples=300, random_state=42))
]

fig, axes = plt.subplots(3, 3, figsize=(18, 15))

for i, (name, (X_test, y_true_test)) in enumerate(datasets):
    print(f"Testing on {name}...")
    
    # Determine number of components
    n_true_components = len(np.unique(y_true_test))
    
    # Fit GMM
    gmm_test = GaussianMixtureEM(
        n_components=n_true_components,
        max_iter=100,
        random_state=42
    )
    gmm_test.fit(X_test)
    y_pred_test = gmm_test.predict(X_test)
    
    # Plot true clusters
    for k in range(n_true_components):
        mask = y_true_test == k
        axes[i, 0].scatter(X_test[mask, 0], X_test[mask, 1], 
                          c=colors[k], alpha=0.7, s=30)
    axes[i, 0].set_title(f'{name}\nTrue Clusters')
    axes[i, 0].grid(True, alpha=0.3)
    axes[i, 0].set_aspect('equal')
    
    # Plot predicted clusters
    for k in range(n_true_components):
        mask = y_pred_test == k
        axes[i, 1].scatter(X_test[mask, 0], X_test[mask, 1], 
                          c=colors[k], alpha=0.7, s=30)
    axes[i, 1].set_title(f'GMM Predictions\n(LL: {gmm_test.log_likelihood_history_[-1]:.1f})')
    axes[i, 1].grid(True, alpha=0.3)
    axes[i, 1].set_aspect('equal')
    
    # Plot with Gaussian ellipses
    axes[i, 2].scatter(X_test[:, 0], X_test[:, 1], c='lightgray', alpha=0.6, s=20)
    for k in range(n_true_components):
        plot_gaussian_ellipse(gmm_test.means_[k], gmm_test.covariances_[k], 
                             axes[i, 2], colors[k], alpha=0.3, n_std=1)
    axes[i, 2].set_title(f'Gaussian Components\n({gmm_test.n_iter_} iterations)')
    axes[i, 2].grid(True, alpha=0.3)
    axes[i, 2].set_aspect('equal')

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("DEMONSTRATION COMPLETE!")
print("=" * 60)
print("\nKey Takeaways:")
print("• EM algorithm successfully converged on various datasets")
print("• GMM works well with elliptical clusters")
print("• Struggles with non-convex shapes (like concentric circles)")
print("• AIC/BIC help with model selection")
print("• Provides both hard and soft cluster assignments")
print("\nTry modifying the parameters and datasets to explore further!")
