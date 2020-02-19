# Copyright 2020 Robin Scheibler
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Implementation of the different types of auxiliary function
based update rules for IVA.
"""
import numpy as np

from .utils import tensor_H


def _ip_single(s, X, W, r_inv):
    """
    Performs update of the s-th demixing vector using
    the iterative projection rules
    """

    n_freq, n_chan, n_frames = X.shape

    # Compute Auxiliary Variable
    # shape: (n_freq, n_chan, n_chan)
    V = np.matmul((X * r_inv[None, None, :]), tensor_H(X) / n_frames)

    WV = np.matmul(W, V)
    rhs = np.eye(n_chan)[None, :, s]  # s-th canonical basis vector
    W[:, s, :] = np.conj(np.linalg.solve(WV, rhs))

    # normalize
    denom = np.matmul(
        np.matmul(W[:, None, s, :], V[:, :, :]), np.conj(W[:, s, :, None])
    )
    W[:, s, :] /= np.sqrt(denom[:, :, 0])


def _ip_double(s1, s2, X, W, r_inv):
    """
    Performs a joint update of the s1-th and s2-th demixing vectors
    usint the iterative projection 2 rules
    """

    n_freq, n_chan, n_frames = X.shape

    # right-hand side for computation of null space basis
    # no_update = [i for i in range(n_src) if i != s]
    rhs = np.eye(n_chan)[None, :, [s1, s2]]

    # Compute Auxiliary Variable
    # shape: (n_freq, n_chan, n_chan)
    V = [
        (X * r_inv[None, i, None, :]) @ tensor_H(X) / n_frames
        for i, s in enumerate([s1, s2])
    ]

    # Basis for demixing vector
    H = []
    HVH = []
    for i, s in enumerate([s1, s2]):
        H.append(np.linalg.solve(W @ V[i], rhs))
        HVH.append(tensor_H(H[i]) @ V[i] @ H[i])

    # Now solve the generalized eigenvalue problem
    lmbda_, R = np.linalg.eig(np.linalg.solve(HVH[0], HVH[1]))

    # Order by decreasing order of eigenvalues
    I_inv = lmbda_[:, 0] > lmbda_[:, 1]
    lmbda_[I_inv, :] = lmbda_[I_inv, ::-1]
    R[I_inv, :, :] = R[I_inv, :, ::-1]

    for i, s in enumerate([s1, s2]):
        denom = np.sqrt(np.conj(R[:, None, :, i]) @ HVH[i] @ R[:, :, i, None])
        W[:, s, None, :] = tensor_H(H[i] @ (R[:, :, i, None] / denom))


def _parametric_background_update(n_src, W, Cx):
    """
    Update the backgroud part of a parametrized demixing matrix
    """

    W_target = W[:, :n_src, :]  # target demixing matrix
    J = W[:, n_src:, :n_src]  # background demixing matrix

    tmp = np.matmul(W_target, Cx)
    J[:, :, :] = tensor_H(
        np.linalg.solve(tmp[:, :, :n_src], tmp[:, :, n_src:])
    )


def _joint_demix_background(s, n_src, X, W, r_inv, Cx):
    """
    Joint update of one demixing vector and one block
    """

    n_freq, n_chan, n_frames = X.shape

    # right-hand side for computation of null space basis
    # no_update = [i for i in range(n_src) if i != s]
    update = [s] + list(range(n_src, n_chan))
    rhs = np.eye(n_chan)[None, :, update]

    # Compute Auxiliary Variable
    # shape: (n_freq, n_chan, n_chan)
    V = (
        np.matmul((X * r_inv[None, None, :]), tensor_H(X))
    ) / n_frames

    # Basis for demixing vector
    Hw = np.linalg.solve(W @ V, rhs)
    HVH = np.matmul(tensor_H(Hw), V @ Hw)

    # Basis for the background
    Hb = np.linalg.solve(W @ Cx, rhs)
    HCH = np.matmul(tensor_H(Hb), Cx @ Hb)

    # Now solve the generalized eigenvalue problem
    B_H = np.linalg.cholesky(HCH)
    B_inv = np.linalg.inv(tensor_H(B_H))
    V_tilde = np.linalg.solve(B_H, HVH) @ B_inv
    lmbda_, R = np.linalg.eigh(V_tilde)
    R = np.matmul(B_inv, R)

    # The target demixing vector requires normalization
    R[:, :, -1] /= np.sqrt(lmbda_[:, -1, None])

    # Assign to target and background
    W[:, s, None, :] = tensor_H(Hw @ R[:, :, -1, None])
    W[:, n_src:, :] = tensor_H(Hb @ R[:, :, :-1])


def _block_ip(sources, X, W, V):
    """
    Block iterative projection update
    This is a joint update of the demixing vectors of sources
    that are not independent
    """

    n_freq, n_chan, n_frames = X.shape

    rhs = np.eye(n_chan)[None, :, sources]
    W[:, sources, :] = tensor_H(np.linalg.solve(W @ V, rhs))

    for s in sources:
        # normalize
        denom = np.matmul(
            np.matmul(W[:, None, s, :], V[:, :, :]), np.conj(W[:, s, :, None])
        )
        W[:, s, :] /= np.sqrt(denom[:, :, 0])
