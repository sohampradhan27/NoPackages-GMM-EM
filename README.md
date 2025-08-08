# Gaussian Mixture Model with Expectation-Maximization

A complete from-scratch implementation of Gaussian Mixture Models (GMM) using the Expectation-Maximization (EM) algorithm in Python. This implementation uses only NumPy for mathematical operations and matplotlib for visualization - no scikit-learn or other ML libraries required.

## 🎯 Overview

Gaussian Mixture Models are probabilistic models that assume data comes from a mixture of multivariate Gaussian distributions. The EM algorithm iteratively estimates the parameters of these distributions:

- **E-step**: Compute posterior probabilities (responsibilities) of each data point belonging to each component
- **M-step**: Update model parameters (means, covariances, mixing weights) based on the responsibilities

## 📁 Repository Structure

```
gmm-em/
├── algorithm/
│   └── gmm_em.py          # Core GMM-EM implementation
├── notebooks/
│   └── demo.ipynb         # Interactive demonstration and visualization
├── data/
│   └── synthetic.py       # Synthetic data generation utilities
├── tests/
│   └── test_gmm.py        # Unit tests
├── README.md              # This file
├── requirements.txt       # Python dependencies
└── setup.py              # Package installation script
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gmm-em.git
cd gmm-em
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install as a package:
```bash
pip install -e .
```

### Basic Usage

```python
import numpy as np
from algorithm.gmm_em import GaussianMixtureEM
from data.synthetic import generate_elongated_clusters

# Generate synthetic data
X, y_true = generate_elongated_clusters(n_samples=300, random_state=42)

# Fit Gaussian Mixture Model
gmm = GaussianMixtureEM(n_components=3, random_state=42)
gmm.fit(X)

# Make predictions
cluster_labels = gmm.predict(X)
probabilities = gmm.predict_proba(X)

print(f"Converged: {gmm.converged_}")
print(f"Log-likelihood: {gmm.log_likelihood_history_[-1]:.4f}")
print(f"Mixing weights: {gmm.weights_}")
```

### Command Line Interface

You can also run GMM-EM directly from the command line:

```bash
# Run on CSV data
python algorithm/gmm_em.py data.csv --n_components 3 --max_iter 100

# With custom parameters
python algorithm/gmm_em.py data.csv --n_components 4 --tol 1e-6 --random_state 42
```

## 📊 Key Features

### Core Implementation (`algorithm/gmm_em.py`)

- **Complete GMM Class**: `GaussianMixtureEM` with full API compatibility
- **Custom PDF Calculation**: Multivariate normal PDF implemented using NumPy linear algebra
- **Robust Convergence**: Tracks log-likelihood and stops when improvement < tolerance
- **Numerical Stability**: Regularization for covariance matrices to ensure positive definiteness
- **Type Hints & Validation**: Comprehensive error checking and documentation

### Synthetic Data Generation (`data/synthetic.py`)

```python
from data.synthetic import generate_2d_blobs, generate_elongated_clusters, generate_concentric_circles

# Simple isotropic clusters
X, y = generate_2d_blobs(n_samples=300, n_components=3)

# Challenging elongated clusters
X, y = generate_elongated_clusters(n_samples=300)

# Non-convex patterns (challenging for GMM)
X, y = generate_concentric_circles(n_samples=300)
```

### Interactive Visualization (`notebooks/demo.ipynb`)

The Jupyter notebook provides:
- Step-by-step walkthrough of the EM algorithm
- Convergence analysis with log-likelihood plots
- Confidence ellipse visualization for Gaussian components
- Model selection using AIC/BIC criteria
- Comparison of hard vs soft clustering assignments
- Animation of component evolution during EM iterations

## 🧮 Mathematical Background

### Gaussian Mixture Model

A GMM models data as a weighted sum of K multivariate Gaussian distributions:

```
p(x) = Σ(k=1 to K) π_k * N(x | μ_k, Σ_k)
```

Where:
- `π_k`: mixing weight for component k
- `μ_k`: mean vector for component k  
- `Σ_k`: covariance matrix for component k
- `N(x | μ, Σ)`: multivariate normal distribution

### EM Algorithm

**E-step**: Compute responsibilities
```
γ(z_nk) = (π_k * N(x_n | μ_k, Σ_k)) / Σ(j=1 to K) π_j * N(x_n | μ_j, Σ_j)
```

**M-step**: Update parameters
```
π_k = (1/N) * Σ(n=1 to N) γ(z_nk)
μ_k = Σ(n=1 to N) γ(z_nk) * x_n / Σ(n=1 to N) γ(z_nk)
Σ_k = Σ(n=1 to N) γ(z_nk) * (x_n - μ_k)(x_n - μ_k)^T / Σ(n=1 to N) γ(z_nk)
```

## 📈 Performance Examples

### Example 1: Simple Blob Clusters
```python
from data.synthetic import generate_2d_blobs
import matplotlib.pyplot as plt

X, y_true = generate_2d_blobs(n_samples=300, n_components=3, random_state=42)
gmm = GaussianMixtureEM(n_components=3, random_state=42)
gmm.fit(X)

print(f"Converged in {gmm.n_iter_} iterations")
print(f"Final log-likelihood: {gmm.log_likelihood_history_[-1]:.2f}")
# Output: Converged in 15 iterations
#         Final log-likelihood: -1247.83
```

### Example 2: Model Selection
```python
# Compare different numbers of components
results = []
for k in range(1, 6):
    gmm = GaussianMixtureEM(n_components=k, random_state=42)
    gmm.fit(X)
    
    # Compute information criteria
    n_params = k * (2 + 3) + k - 1  # for 2D data
    aic = 2 * n_params - 2 * gmm.log_likelihood_history_[-1]
    bic = n_params * np.log(len(X)) - 2 * gmm.log_likelihood_history_[-1]
    
    results.append((k, aic, bic))
```

## 🔧 API Reference

### GaussianMixtureEM Class

#### Parameters
- `n_components` (int): Number of Gaussian components
- `tol` (float, default=1e-6): Convergence tolerance
- `max_iter` (int, default=100): Maximum EM iterations
- `reg_covar` (float, default=1e-6): Regularization for covariance matrices
- `random_state` (int, optional): Random seed for reproducibility

#### Methods
- `fit(X)`: Fit the model to data
- `predict(X)`: Predict cluster labels
- `predict_proba(X)`: Predict posterior probabilities

#### Attributes
- `weights_`: Mixing weights for each component
- `means_`: Mean vectors for each component
- `covariances_`: Covariance matrices for each component
- `log_likelihood_history_`: Log-likelihood at each iteration
- `converged_`: Whether the algorithm converged
- `n_iter_`: Number of iterations performed

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

Tests cover:
- Basic functionality and edge cases
- Convergence properties
- Numerical stability
- API compatibility

## 🎨 Visualization Examples

The repository includes several visualization utilities:

1. **Scatter plots** with cluster assignments
2. **Confidence ellipses** showing Gaussian components
3. **Convergence plots** tracking log-likelihood
4. **Soft assignment heatmaps** showing membership probabilities
5. **Uncertainty visualization** highlighting ambiguous regions

## 🔬 Advanced Usage

### Custom Initialization
```python
# Use k-means++ style initialization (default)
gmm = GaussianMixtureEM(n_components=3, random_state=42)

# Multiple random restarts for better solutions
best_gmm = None
best_ll = -np.inf

for seed in range(10):
    gmm = GaussianMixtureEM(n_components=3, random_state=seed)
    gmm.fit(X)
    if gmm.log_likelihood_history_[-1] > best_ll:
        best_ll = gmm.log_likelihood_history_[-1]
        best_gmm = gmm
```

### Monitoring Convergence
```python
gmm = GaussianMixtureEM(n_components=3, max_iter=1000, tol=1e-8)
gmm.fit(X)

# Plot convergence
plt.plot(gmm.log_likelihood_history_)
plt.xlabel('Iteration')
plt.ylabel('Log-Likelihood')
plt.title('EM Convergence')
plt.show()
```

## ⚠️ Limitations

1. **Gaussian Assumption**: Assumes data comes from Gaussian distributions
2. **Local Optima**: EM can converge to local maxima; use multiple random restarts
3. **Initialization Sensitivity**: Results depend on parameter initialization
4. **Computational Complexity**: O(n·k·d²·i) where n=samples, k=components, d=dimensions, i=iterations
5. **Non-convex Clusters**: Struggles with complex shapes like concentric circles

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📚 References

1. Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm.
2. Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Chapter 9: Mixture Models and EM.
3. Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective. Chapter 11: Mixture Models and EM.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For questions or issues, please open a GitHub issue or contact [your-email@example.com].

---

**Happy clustering! 🎯**
