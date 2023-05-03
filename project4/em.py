"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep_nonoptimal(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    n = X.shape[0]
    lh = None # initialize likelihood (posterior) value
    for i in range(n):
        cols = np.nonzero(X[i])[0]
        d = cols.shape[0]
        power = - ((X[i, cols] - mixture.mu[:, cols])**2).sum(axis=1) / (2 * mixture.var)
        lh_point = mixture.p * (np.exp(power) / ((2 * np.pi * mixture.var)**(d/2)))
        if lh is None:
            lh = lh_point.reshape(1, -1).copy()
        else:
            lh = np.r_[lh, lh_point.reshape(1, -1).copy()]
    loglh = np.log(lh.sum(axis=1)).sum()
    lh /= lh.sum(axis=1).reshape(-1, 1)
    return (lh, loglh)


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    def log_N_uj(x: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
        cols = np.nonzero(x)[0]
        d = cols.shape[0]
        power = - ((x[cols] - mixture.mu[:, cols])**2).sum(axis=1) / (2 * mixture.var)
        return power - (d / 2) * np.log(2 * np.pi * mixture.var)
    
    def f_uj(x: np.ndarray, mixture: GaussianMixture) -> np.ndarray: # will return shape (k,)
        return np.log(mixture.p + 1e-16) + log_N_uj(x=x, mixture=mixture)
    
    n, d = X.shape
    log_posterior = None
    for u in range(n):
        # value of f(u, j)
        f = f_uj(x=X[u], mixture=mixture)
        if log_posterior is None:
            log_posterior = f.reshape(1, -1).copy()
        else:
            log_posterior = np.r_[log_posterior, f.reshape(1, -1).copy()]
    
    #likelihood = np.log(np.exp(log_posterior).sum(axis=1)).sum()
    likelihood = logsumexp(log_posterior, axis=1).sum()
    
    #log_posterior -= np.log(np.exp(log_posterior).sum(axis=1)).reshape(-1, 1)
    log_posterior -= logsumexp(log_posterior, axis=1).reshape(-1, 1)
    return (np.exp(log_posterior), likelihood)
    #raise NotImplementedError



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    K = post.shape[1]
    indicator = np.where(X > 0, 1, 0)
    n_hat = post.sum(axis=0) # shape (K, )
    p = n_hat / n # shape (K, )

    mu = np.zeros((K, d))
    for k in range(K):
        for coord in range(d):
            denom = post[:, k] * indicator[:, coord]
            if denom.sum() >= 1.:
                mu[k, coord] = (post[:, k] * indicator[:, coord] * X[:, coord]).sum() / denom.sum()
            else:
                mu[k, coord] = mixture.mu[k, coord]

    #mu = (post.T @ X) / n_hat.reshape(-1, 1) # (K, n) @ (n, d) => (K, d), then / (K, 1) => (K, d)
    
    var = np.zeros((K,))
    denom = np.zeros((K,))
    for i in range(n):
        cols = np.nonzero(X[i])[0]
        var += ((X[i, cols] - mu[:, cols])**2).sum(axis=1) * post[i] # (d, ) - (K, d) => (K, d), then sum(axis=1) => (K, ) * (K, )
        denom += post[i] * cols.shape[0]
    var /= denom
    var = np.maximum(var, np.ones_like(var)*min_variance)

    return GaussianMixture(p=p, mu=mu, var=var)
    #raise NotImplementedError


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    epsilon = 1e-6
    old_loglh = None
    new_loglh = None
    while (old_loglh is None) or (np.abs(old_loglh-new_loglh) > epsilon * np.abs(old_loglh)):
        old_loglh = new_loglh
        posterior, new_loglh = estep(X=X, mixture=mixture)
        mixture = mstep(X, post=posterior, mixture=mixture, min_variance=.25)

    return (mixture, post, new_loglh)
    #raise NotImplementedError


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    def log_N_uj(x: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
        cols = np.nonzero(x)[0]
        d = cols.shape[0]
        power = - ((x[cols] - mixture.mu[:, cols])**2).sum(axis=1) / (2 * mixture.var)
        return power - (d / 2) * np.log(2 * np.pi * mixture.var)
    
    def f_uj(x: np.ndarray, mixture: GaussianMixture) -> np.ndarray: # will return shape (k,)
        return np.log(mixture.p + 1e-16) + log_N_uj(x=x, mixture=mixture)
    
    n, d = X.shape
    log_posterior = None
    for u in range(n):
        # value of f(u, j)
        f = f_uj(x=X[u], mixture=mixture)
        if log_posterior is None:
            log_posterior = f.reshape(1, -1).copy()
        else:
            log_posterior = np.r_[log_posterior, f.reshape(1, -1).copy()]
    
    #log_posterior -= np.log(np.exp(log_posterior).sum(axis=1)).reshape(-1, 1)
    log_posterior -= logsumexp(log_posterior, axis=1).reshape(-1, 1)
    posterior = np.exp(log_posterior)
    
    #fill_value = np.zeros_like(X) # wrong option
    # fill_value = mixture.mu[np.argmax(posterior, axis=1)] # wrong option too
    # filling value is not corresponding mu coordinate of most probable Gaussian
    # but average of corresponding coordinates of all mus weighted by posteriors
    fill_value = (posterior @ mixture.mu) / posterior.sum(axis=1).reshape(-1, 1)
    return np.where(X == 0, fill_value, X)
    #raise NotImplementedError
