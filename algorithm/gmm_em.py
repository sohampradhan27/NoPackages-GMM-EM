"""
Gaussian Mixture Model implementation using Expectation-Maximization algorithm.

This module provides a complete implementation of GMM-EM using only numpy
for linear algebra operations and the standard library.
"""

import numpy as np
import argparse
import csv
from typing import Tuple, Optional, Union
import warnings


class GaussianMixtureEM:
    """
    Gaussian Mixture Model using Expectation-Maximization algorithm.
    
    This implementation uses only numpy and standard library to fit a mixture
    of multivariate Gaussian distributions to data.
    
    Parameters
    ----------
    n_components : int
        Number of Gaussian components in the mixture.
    tol : float, default=1e-6
        Convergence threshold. EM stops when log-likelihood improvement
        is less than this value.
    max_iter : int, default=100
        Maximum number of EM iterations.
    reg_covar : float, default=1e-6
        Regularization term added to diagonal of covariance matrices
        to ensure positive definiteness.
    random_state : int, optional
        Random seed for reproducible results.
    """
    
    def __init__(
        self, 
        n_components: int, 
        tol: float = 1e-6, 
        max_iter: int = 100, 
        reg_covar: float = 1e-6,
        random_state: Optional[int] = None
    ):
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        if tol <= 0:
            raise ValueError("tol must be positive")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if reg_covar < 0:
            raise ValueError("reg_covar must be non-negative")
            
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.reg_covar = reg_covar
        self.random_state = random_state
        
        # Model parameters (set during fit)
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.log_likelihood_history_ = []
        self.n_iter_ = 0
        self.converged_ = False
        
    def _multivariate_normal_pdf(
        self, 
        X: np.ndarray, 
        mean: np.ndarray, 
        cov: np.ndarray
    ) -> np.ndarray:
        """
        Compute multivariate normal probability density function.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data points.
        mean : array-like, shape (n_features,)
            Mean vector of the Gaussian.
        cov : array-like, shape (n_features, n_features)
            Covariance matrix of the Gaussian.
            
        Returns
        -------
        pdf : array-like, shape (n_samples,)
            Probability density values.
        """
        n_features = X.shape[1]
        
        # Compute determinant and inverse of covariance matrix
        try:
            cov_inv = np.linalg.inv(cov)
            cov_det = np.linalg.det(cov)
        except np.linalg.LinAlgError:
            warnings.warn("Covariance matrix is singular, adding regularization")
            cov += self.reg_covar * np.eye(n_features)
            cov_inv = np.linalg.inv(cov)
            cov_det = np.linalg.det(cov)
            
        if cov_det <= 0:
            warnings.warn("Non-positive determinant, adding regularization")
            cov += self.reg_covar * np.eye(n_features)
            cov_inv = np.linalg.inv(cov)
            cov_det = np.linalg.det(cov)
        
        # Center the data
        X_centered = X - mean
        
        # Compute quadratic form: (x-μ)ᵀ Σ⁻¹ (x-μ)
        quadratic_form = np.sum(X_centered @ cov_inv * X_centered, axis=1)
        
        # Compute normalization constant
        norm_const = 1.0 / np.sqrt((2 * np.pi) ** n_features * cov_det)
        
        # Compute PDF
        pdf = norm_const * np.exp(-0.5 * quadratic_form)
        
        return pdf
    
    def _initialize_parameters(self, X: np.ndarray) -> None:
        """
        Initialize GMM parameters using k-means++ style initialization.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n_samples, n_features = X.shape
        
        # Initialize weights uniformly
        self.weights_ = np.ones(self.n_components) / self.n_components
        
        # Initialize means using k-means++ style selection
        self.means_ = np.zeros((self.n_components, n_features))
        
        # First center is random
        self.means_[0] = X[np.random.randint(n_samples)]
        
        # Subsequent centers chosen with probability proportional to squared distance
        for k in range(1, self.n_components):
            distances = np.array([
                min([np.linalg.norm(x - c) ** 2 for c in self.means_[:k]])
                for x in X
            ])
            probabilities = distances / distances.sum()
            cumulative_prob = probabilities.cumsum()
            r = np.random.rand()
            
            for j, p in enumerate(cumulative_prob):
                if r < p:
                    self.means_[k] = X[j]
                    break
        
        # Initialize covariances as identity matrices scaled by data variance
        data_var = np.var(X, axis=0).mean()
        self.covariances_ = np.array([
            np.eye(n_features) * data_var for _ in range(self.n_components)
        ])
    
    def _e_step(self, X: np.ndarray) -> np.ndarray:
        """
        Expectation step: compute responsibilities.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
            
        Returns
        -------
        responsibilities : array-like, shape (n_samples, n_components)
            Posterior probabilities of component membership.
        """
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))
        
        # Compute weighted likelihoods for each component
        for k in range(self.n_components):
            pdf_k = self._multivariate_normal_pdf(X, self.means_[k], self.covariances_[k])
            responsibilities[:, k] = self.weights_[k] * pdf_k
        
        # Normalize to get responsibilities (avoid division by zero)
        row_sums = responsibilities.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1e-8  # Prevent division by zero
        responsibilities /= row_sums
        
        return responsibilities
    
    def _m_step(self, X: np.ndarray, responsibilities: np.ndarray) -> None:
        """
        Maximization step: update parameters.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        responsibilities : array-like, shape (n_samples, n_components)
            Posterior probabilities from E-step.
        """
        n_samples, n_features = X.shape
        
        # Update mixing weights
        N_k = responsibilities.sum(axis=0)
        self.weights_ = N_k / n_samples
        
        # Update means
        for k in range(self.n_components):
            if N_k[k] > 0:
                self.means_[k] = (responsibilities[:, k, np.newaxis] * X).sum(axis=0) / N_k[k]
            
        # Update covariances
        for k in range(self.n_components):
            if N_k[k] > 0:
                X_centered = X - self.means_[k]
                cov_k = np.zeros((n_features, n_features))
                
                for i in range(n_samples):
                    x_diff = X_centered[i, np.newaxis]
                    cov_k += responsibilities[i, k] * (x_diff.T @ x_diff)
                
                self.covariances_[k] = cov_k / N_k[k]
                
                # Add regularization to ensure positive definiteness
                self.covariances_[k] += self.reg_covar * np.eye(n_features)
    
    def _compute_log_likelihood(self, X: np.ndarray) -> float:
        """
        Compute log-likelihood of the data given current parameters.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
            
        Returns
        -------
        log_likelihood : float
            Log-likelihood of the data.
        """
        n_samples = X.shape[0]
        log_likelihood = 0.0
        
        for i in range(n_samples):
            sample_likelihood = 0.0
            for k in range(self.n_components):
                pdf_k = self._multivariate_normal_pdf(
                    X[i:i+1], self.means_[k], self.covariances_[k]
                )
                sample_likelihood += self.weights_[k] * pdf_k[0]
            
            if sample_likelihood > 0:
                log_likelihood += np.log(sample_likelihood)
            else:
                log_likelihood += -np.inf
                
        return log_likelihood
    
    def fit(self, X: np.ndarray) -> 'GaussianMixtureEM':
        """
        Fit the Gaussian Mixture Model to data using EM algorithm.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
            
        Returns
        -------
        self : GaussianMixtureEM
            Returns the instance itself.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        if X.shape[0] < self.n_components:
            raise ValueError("Number of samples must be >= n_components")
            
        # Initialize parameters
        self._initialize_parameters(X)
        
        # EM iterations
        self.log_likelihood_history_ = []
        prev_log_likelihood = -np.inf
        
        for iteration in range(self.max_iter):
            # E-step
            responsibilities = self._e_step(X)
            
            # M-step
            self._m_step(X, responsibilities)
            
            # Check convergence
            current_log_likelihood = self._compute_log_likelihood(X)
            self.log_likelihood_history_.append(current_log_likelihood)
            
            if abs(current_log_likelihood - prev_log_likelihood) < self.tol:
                self.converged_ = True
                break
                
            prev_log_likelihood = current_log_likelihood
        
        self.n_iter_ = iteration + 1
        
        if not self.converged_:
            warnings.warn(f"EM did not converge after {self.max_iter} iterations")
            
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict posterior probabilities of components for each sample.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        responsibilities : array-like, shape (n_samples, n_components)
            Posterior probabilities.
        """
        if self.means_ is None:
            raise ValueError("Model has not been fitted yet")
            
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
            
        return self._e_step(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict component labels for each sample.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        labels : array-like, shape (n_samples,)
            Component labels.
        """
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)


def load_csv_data(filepath: str) -> np.ndarray:
    """Load data from CSV file."""
    data = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                data.append([float(x) for x in row])
            except ValueError:
                continue  # Skip header or invalid rows
    return np.array(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GMM-EM on CSV data")
    parser.add_argument("csv_file", help="Path to CSV file with data")
    parser.add_argument("--n_components", type=int, default=3, help="Number of components")
    parser.add_argument("--max_iter", type=int, default=100, help="Maximum iterations")
    parser.add_argument("--tol", type=float, default=1e-6, help="Convergence tolerance")
    parser.add_argument("--random_state", type=int, help="Random seed")
    
    args = parser.parse_args()
    
    try:
        # Load data
        print(f"Loading data from {args.csv_file}")
        X = load_csv_data(args.csv_file)
        print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
        
        # Fit GMM
        print(f"Fitting GMM with {args.n_components} components...")
        gmm = GaussianMixtureEM(
            n_components=args.n_components,
            max_iter=args.max_iter,
            tol=args.tol,
            random_state=args.random_state
        )
        
        gmm.fit(X)
        
        # Print results
        print(f"Converged: {gmm.converged_}")
        print(f"Iterations: {gmm.n_iter_}")
        print(f"Final log-likelihood: {gmm.log_likelihood_history_[-1]:.4f}")
        print(f"Mixing weights: {gmm.weights_}")
        
        # Predict labels
        labels = gmm.predict(X)
        print(f"Cluster assignments: {np.bincount(labels)}")
        
    except Exception as e:
        print(f"Error: {e}")
