import numpy as np
from .utils import tensor_H


def pca(X, n_src=None, return_filters=False, normalize=True):
    """
    Whitens the input signal X using principal component analysis (PCA)

    Parameters
    ----------
    X: ndarray (nframes, nfrequencies, nchannels)
        The input signal
    n_src: int
        The desired number of principal components
    return_filters: bool
        If this flag is set to true, the PCA matrix
        is also returned (default False)
    normalize: bool
        If this flag is set to false, the decorrelated
        channels are not normalized to unit variance
        (default True)
    """
    n_frames, n_freq, n_chan = X.shape

    # default to determined case
    if n_src is None:
        n_src = n_chan

    assert (
        n_src <= n_chan
    ), "The number of sources cannot be more than the number of channels."

    # compute the cov mat (n_freq, n_chan, n_chan)
    X_T = X.transpose([1, 2, 0])
    covmat = (X_T @ tensor_H(X_T)) * (1.0 / n_frames)

    # Compute EVD
    # v.shape == (n_freq, n_chan), w.shape == (n_freq, n_chan, n_chan)
    eig_val, eig_vec = np.linalg.eigh(covmat)

    # Reorder the eigenvalues from so that they are in descending order
    eig_val = eig_val[:, ::-1]
    eig_vec = eig_vec[:, :, ::-1]

    # The whitening matrices
    if normalize:
        Q = (1.0 / np.sqrt(eig_val[:, :, None])) * tensor_H(eig_vec)
    else:
        Q = tensor_H(eig_vec)

    # The decorrelated signal
    Y = (Q[:, :n_src, :] @ X_T).transpose([2, 0, 1]).copy()

    if return_filters:
        return Y, Q
    else:
        return Y
