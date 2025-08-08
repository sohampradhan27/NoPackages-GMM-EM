"""
Unit tests for the Gaussian Mixture Model implementation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'algorithm'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))

import pytest
import numpy as np
from gmm_em import GaussianMixtureEM
from synthetic import generate_2d_blobs, generate_elongated_clusters


class TestGaussianMixtureEM:
    """Test suite for GaussianMixtureEM class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        np.random.seed(42)
        self.X_simple, self.y_simple = generate_2d_blobs(
            n_samples=150, n_components=3, random_state=42
        )
        self.X_complex, self.y_complex = generate_elongated_clusters(
            n_samples=200, random_state=42
        )
    
    def test_initialization(self):
        """Test GMM initialization with valid parameters."""
        gmm = GaussianMixtureEM(n_components=3)
        
        assert gmm.n_components == 3
        assert gmm.tol == 1e-6
        assert gmm.max_iter == 100
        assert gmm.reg_covar == 1e-6
        assert gmm.weights_ is None
        assert gmm.means_ is None
        assert gmm.covariances_ is None
        assert gmm.log_likelihood_history_ == []
        assert gmm.n_iter_ == 0
        assert gmm.converged_ == False
    
    def test_initialization_invalid_params(self):
        """Test GMM initialization with invalid parameters."""
        with pytest.raises(ValueError, match="n_components must be positive"):
            GaussianMixtureEM(n_components=0)
        
        with pytest.raises(ValueError, match="tol must be positive"):
            GaussianMixtureEM(n_components=2, tol=-1e-6)
        
        with pytest.raises(ValueError, match="max_iter must be positive"):
            GaussianMixtureEM(n_components=2, max_iter=0)
        
        with pytest.raises(ValueError, match="reg_covar must be non-negative"):
            GaussianMixtureEM(n_components=2, reg_covar=-1e-6)
    
    def test_multivariate_normal_pdf(self):
        """Test multivariate normal PDF computation."""
        gmm = GaussianMixtureEM(n_components=1)
        
        # Test with simple 2D case
        X = np.array([[0, 0], [1, 1], [2, 2]])
        mean = np.array([1, 1])
        cov = np.eye(2)
        
        pdf = gmm._multivariate_normal_pdf(X, mean, cov)
        
        assert pdf.shape == (3,)
        assert np.all(pdf > 0)  # All probabilities should be positive
        assert pdf[1] > pdf[0]  # Point [1,1] closer to mean [1,1] than [0,0]
        assert pdf[1] > pdf[2]  # Point [1,1] closer to mean [1,1] than [2,2]
    
    def test_fit_basic(self):
        """Test basic fitting functionality."""
        gmm = GaussianMixtureEM(n_components=3, max_iter=50, random_state=42)
        gmm.fit(self.X_simple)
        
        # Check that parameters were set
        assert gmm.weights_ is not None
        assert gmm.means_ is not None
        assert gmm.covariances_ is not None
        
        # Check shapes
        assert gmm.weights_.shape == (3,)
        assert gmm.means_.shape == (3, 2)
        assert gmm.covariances_.shape == (3, 2, 2)
        
        # Check weights sum to 1
        assert np.isclose(np.sum(gmm.weights_), 1.0)
        
        # Check that algorithm ran
        assert gmm.n_iter_ > 0
        assert len(gmm.log_likelihood_history_) == gmm.n_iter_
    
    def test_fit_convergence(self):
        """Test that EM algorithm converges."""
        gmm = GaussianMixtureEM(n_components=3, max_iter=100, tol=1e-6, random_state=42)
        gmm.fit(self.X_simple)
        
        # Check convergence
        if len(gmm.log_likelihood_history_) > 1:
            # Log-likelihood should be non-decreasing
            ll_diff = np.diff(gmm.log_likelihood_history_)
            assert np.all(ll_diff >= -1e-10)  # Allow small numerical errors
    
    def test_predict(self):
        """Test prediction functionality."""
        gmm = GaussianMixtureEM(n_components=3, random_state=42)
        gmm.fit(self.X_simple)
        
        # Test predict
        labels = gmm.predict(self.X_simple)
        assert labels.shape == (len(self.X_simple),)
        assert np.all(labels >= 0)
        assert np.all(labels < 3)
        
        # Test predict_proba
        probs = gmm.predict_proba(self.X_simple)
        assert probs.shape == (len(self.X_simple), 3)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
        
        # Check that probabilities sum to 1
        assert np.allclose(np.sum(probs, axis=1), 1.0)
        
        # Check consistency between predict and predict_proba
        predicted_labels = np.argmax(probs, axis=1)
        assert np.array_equal(labels, predicted_labels)
    
    def test_predict_before_fit(self):
        """Test that prediction fails before fitting."""
        gmm = GaussianMixtureEM(n_components=3)
        
        with pytest.raises(ValueError, match="Model has not been fitted yet"):
            gmm.predict(self.X_simple)
        
        with pytest.raises(ValueError, match="Model has not been fitted yet"):
            gmm.predict_proba(self.X_simple)
    
    def test_invalid_input_shapes(self):
        """Test handling of invalid input shapes."""
        gmm = GaussianMixtureEM(n_components=2)
        
        # Test 1D input
        with pytest.raises(ValueError, match="X must be 2-dimensional"):
            gmm.fit(np.array([1, 2, 3, 4]))
        
        # Test too few samples
        with pytest.raises(ValueError, match="Number of samples must be >= n_components"):
            gmm.fit(np.array([[1, 2]]))  # Only 1 sample for 2 components
    
    def test_single_component(self):
        """Test GMM with single component."""
        gmm = GaussianMixtureEM(n_components=1, random_state=42)
        gmm.fit(self.X_simple)
        
        assert gmm.weights_.shape == (1,)
        assert gmm.means_.shape == (1, 2)
        assert gmm.covariances_.shape == (1, 2, 2)
