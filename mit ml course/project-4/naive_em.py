"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture

def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    ret = np.zeros((X.shape[0], mixture.mu.shape[0]))
    loglike = 0
    for i in range(X.shape[0]):
        loc_prob = 0
        sm = 0
        for j in range(mixture.mu.shape[0]):
            loc_prob = mixture.p[j]*np.exp(-0.5*((np.linalg.norm(X[i]-mixture.mu[j]))**2)/mixture.var[j])/((2*np.pi*mixture.var[j])**(X.shape[1]/2))
            ret[i][j] = loc_prob
            sm += loc_prob
        ret[i] /= sm
        loglike += np.log(sm)
    return (ret,loglike)
    raise NotImplementedError


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    mu = np.zeros((post.shape[1], X.shape[1]))
    var = np.zeros((post.shape[1],))
    nj = np.sum(post, axis=0)
    p = nj/X.shape[0]
    for j in range(post.shape[1]):
        for i in range(X.shape[0]):
            mu[j] += post[i][j]*X[i]
        mu[j] /= nj[j]
        for i in range(X.shape[0]):
            var[j] += post[i][j]*np.linalg.norm(X[i]-mu[j])**2;
        var[j] /= nj[j]*X.shape[1]
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
        mixture = mstep(X, post)

    return mixture, post, ll
