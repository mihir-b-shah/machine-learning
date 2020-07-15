"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture

def logdensity(Xi, mu, var):
    return -len(Xi)/2*np.log(2*np.pi*var)-((np.linalg.norm(Xi-mu))**2)/(2*var)

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
    EPS = 1e-16
    ret = np.zeros((X.shape[0], mixture.mu.shape[0]))
    loglike = 0
    for i in range(X.shape[0]):
        predicate = X[i,:] != 0
        for j in range(mixture.mu.shape[0]):
            ll = logdensity(X[i, predicate], mixture.mu[j, predicate], mixture.var[j])
            ret[i][j] = np.log(mixture.p[j]+EPS)+ll
        locsum = logsumexp(ret[i,:])
        ret[i,:] -= locsum
        loglike += locsum
    return (np.exp(ret),loglike)
    raise NotImplementedError

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
    mu = np.zeros((post.shape[1], X.shape[1]))
    var = np.zeros((post.shape[1],))
    
    for k in range(post.shape[1]):
        for l in range(X.shape[1]):
            for u in range(X.shape[0]):
                if(X[u][l] != 0):
                    mu[k][l] += post[u][k]*X[u][l]
            sm = 0
            for u in range(post.shape[0]):
                if(X[u][l] != 0):
                    sm += post[u][k]
            if(sm >= 1):
                mu[k][l] /= sm
            else:
                mu[k][l] = mixture.mu[k][l]

    p = np.sum(post, axis=0)/X.shape[0]

    for k in range(post.shape[1]):
        v = 0
        for u in range(X.shape[0]):
            mgn = 0
            for idx in range(X.shape[1]):
                if(X[u][idx] != 0):
                    mgn += (X[u][idx]-mu[k][idx])**2
            v += mgn*post[u][k]
        ct = 0
        for u in range(X.shape[0]):
            ct += len(np.nonzero(X[u])[0])*post[u][k]
        v /= ct
        if(v < min_variance):
            var[k] = min_variance
        else:
            var[k] = v
    return GaussianMixture(mu, var, p)
    raise NotImplementedError

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
    llpr = None
    ll = None
    while(llpr is None or ll-llpr > 1e-6 * abs(ll)):
        llpr = ll
        post, ll = estep(X, mixture)
        mixture = mstep(X, post, mixture)

    return mixture, post, ll
    raise NotImplementedError

def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    X = X.copy()
    EPS = 1e-16
    probs = np.zeros((X.shape[0], mixture.mu.shape[0]))
    for i in range(X.shape[0]):
        predicate = X[i,:] != 0
        for j in range(mixture.mu.shape[0]):
            ll = logdensity(X[i, predicate], mixture.mu[j, predicate], mixture.var[j])
            probs[i][j] = np.log(mixture.p[j]+EPS)+ll
        locsum = logsumexp(probs[i,:])
        probs[i,:] -= locsum
    probs = np.exp(probs)   
    for user in range(X.shape[0]):
        for dim in range(X.shape[1]):
            if(X[user][dim] == 0):
                for j in range(probs.shape[1]):
                    X[user][dim] += mixture.mu[j][dim]*probs[user][j]
    return X      
    raise NotImplementedError
