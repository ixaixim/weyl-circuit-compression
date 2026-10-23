# retraction, projection, vector transport.
# the code is adapted to work with batched arrays
from __future__ import annotations

import weyl_compression.jax_config

import jax.numpy as jnp
from jax.numpy.linalg import eigh

#utils
def _adjoint(A):
    """Hermitian adjoint that works with batched arrays (..., d, d)."""
    return jnp.swapaxes(jnp.conj(A), -1, -2)


def _sym(A):
    """Hermitian part."""
    return 0.5 * (A + _adjoint(A))


def _skew(A):
    """Skew-Hermitian part."""
    return 0.5 * (A - _adjoint(A))

# projection 
def project_to_tangent(U, X):
    r"""
    Orthogonal projection of an ambient direction X onto the tangent space
    T_U U(d) with the Euclidean (Frobenius) metric.

    For square unitary U, the standard Stiefel projector simplifies to
        Π_U(X) = X - U * sym(U^† X)  = U * skew(U^† X).
    This function uses the numerically stable `X - U sym(U^† X)` form.

    Shapes:
        U: (..., d, d) unitary
        X: (..., d, d) ambient direction
        returns: (..., d, d)
    """
    UX = _adjoint(U) @ X
    return X - U @ _sym(UX)

# retraction (polar)

def _hermitian_invsqrt(H, eps=1e-12):
    """
    Compute H^{-1/2} for Hermitian PSD H using an eigen-decomposition.
    Supports batched inputs.
    """
    # eigh returns eigenvalues in ascending order
    evals, evecs = eigh(H)
    # Clamp small negatives from roundoff and avoid division by zero
    evals = jnp.clip(evals, eps, None)
    invsqrt_diag = evals ** (-0.5)
    # Recompose: Q * Diag(inv_sqrt) * Q^†
    # Expand diag for broadcasting over last two axes
    invsqrt = (evecs * invsqrt_diag[..., None, :]) @ _adjoint(evecs)
    return invsqrt

# polar retraction is more accurate than QR decomposition, but requires more FLOPS (appx. 13N*3 vs. 4/3N^3, 10 times more expensive)
def retract_polar(U, Xi, eps: float = 1e-12):
    r"""
    Polar retraction R_U(Xi) onto U(d).

    If Xi ∈ T_U U(d), then the polar factor simplifies to
        R_U(Xi) = (U + Xi) * (I + Xi^† Xi)^{-1/2},
    since cross terms cancel for tangent Xi (U^†Xi is skew-Hermitian).

    Shapes:
        U:  (..., d, d) unitary
        Xi: (..., d, d) tangent at U
        returns: U_next (..., d, d) unitary (up to numerical tol)
    """
    I = jnp.eye(U.shape[-1], dtype=U.dtype)
    A = U + Xi
    # Because Xi is tangent, (U + Xi)^†(U + Xi) = I + Xi^† Xi
    H = I + _adjoint(Xi) @ Xi
    H_invsqrt = _hermitian_invsqrt(H, eps=eps)
    U_next = A @ H_invsqrt
    return U_next

# vector transport
def transport_by_projection(U, Xi_step, zeta, eps: float = 1e-12):
    r"""
    Vector transport defined by
        T_{U→U'}(zeta) := Π_{U'}(zeta),
    where U' = R_U(Xi_step) is the retracted point and Π is the tangent projector.

    Inputs:
        U:        (..., d, d) current point
        Xi_step:  (..., d, d) step direction in T_U U(d)
        zeta:     (..., d, d) a tangent vector at U to be transported
    Returns:
        zeta_trans: (..., d, d) tangent at U' = R_U(Xi_step)
        U_next:     (..., d, d) the retracted point (returning it avoids recomputing)
    """
    U_next = retract_polar(U, Xi_step, eps=eps)
    zeta_trans = project_to_tangent(U_next, zeta)
    return zeta_trans, U_next
