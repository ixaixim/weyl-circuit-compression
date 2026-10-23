# generate batch of single qubit and two qubit matrices and test the geometrical properties.
import weyl_compression.jax_config
import pytest

import jax.numpy as jnp
import jax
from jax import random

from weyl_compression.optimization.riemannian_adam.geometry import _adjoint, _sym, _skew, project_to_tangent
def _frob_norm(A):
    # Frobenius norm over the last two axes; returns scalar for unbatched,
    # or per-example norm for batched. We'll reduce with max() in asserts.
    return jnp.linalg.norm(A, ord='fro', axis=(-2, -1))

def random_unitary(key, d, batch_shape=()):
    """Random unitary via QR on complex Gaussian; supports batching."""
    key_r, key_i = random.split(key)
    A = random.normal(key_r, batch_shape + (d, d), dtype=jnp.float64)
    B = random.normal(key_i, batch_shape + (d, d), dtype=jnp.float64)
    Z = (A + 1j * B).astype(jnp.complex128)

    # Batched QR is supported
    Q, R = jnp.linalg.qr(Z)

    # Optional: fix the arbitrary complex phases on the diagonal of R to make Q "unique".
    # This isn't required for the tests, but it keeps Q well-conditioned.
    diag = jnp.diagonal(R, axis1=-2, axis2=-1)
    phase = diag / jnp.where(jnp.abs(diag) > 0, jnp.abs(diag), 1.0)
    Q = Q * jnp.expand_dims(jnp.conj(phase), axis=-2)  # right-multiply by diag(conj(phase))

    return Q

@pytest.mark.parametrize("batch_shape", [(), (3,), (2, 2)])
def test_projector_idempotence(batch_shape):
    """
    Check P_U(P_U(X)) = P_U(X) up to numerical tolerance, for single and batched inputs.
    """
    key = random.PRNGKey(0)
    d = 4
    U = random_unitary(key, d, batch_shape=batch_shape)
    keyX = random.PRNGKey(1)
    # Random ambient direction X
    X = random.normal(keyX, batch_shape + (d, d), dtype=jnp.float64) \
        + 1j * random.normal(random.PRNGKey(2), batch_shape + (d, d), dtype=jnp.float64)
    X = X.astype(jnp.complex128)

    Y = project_to_tangent(U, X)
    YY = project_to_tangent(U, Y)

    diff_norm = _frob_norm(YY - Y)
    # Reduce to scalar for assertion
    max_diff = jnp.max(diff_norm) if diff_norm.ndim > 0 else diff_norm
    assert max_diff < 1e-10, f"Projector not idempotent; max Frobenius diff = {float(max_diff)}"

@pytest.mark.parametrize("batch_shape", [(), (5,), (2, 3)])
def test_projector_range_is_tangent(batch_shape):
    """
    Check that U^† P_U(X) is skew-Hermitian (i.e., in the Lie algebra u(d)),
    which confirms P_U(X) lies in T_U U(d).
    """
    key = random.PRNGKey(3)
    d = 4
    U = random_unitary(key, d, batch_shape=batch_shape)

    keyX = random.PRNGKey(4)
    X = random.normal(keyX, batch_shape + (d, d), dtype=jnp.float64) \
        + 1j * random.normal(random.PRNGKey(5), batch_shape + (d, d), dtype=jnp.float64)
    X = X.astype(jnp.complex128)

    Y = project_to_tangent(U, X)           # (..., d, d)
    A = _adjoint(U) @ Y                    # (..., d, d), should be skew-Hermitian
    herm_part = _sym(A)                    # should be ~0

    herm_norm = _frob_norm(herm_part)
    max_herm = jnp.max(herm_norm) if herm_norm.ndim > 0 else herm_norm
    assert max_herm < 1e-10, f"Projected vector not in tangent space; max sym part = {float(max_herm)}"

def test_fixed_point_on_pure_tangent():
    """
    (Extra safety) If X is already tangent, P_U(X) should return X.
    Construct X = U * skew(H).
    """
    key = random.PRNGKey(6)
    d = 4
    U = random_unitary(key, d)

    # Random H, then make it skew-Hermitian
    keyH = random.PRNGKey(7)
    H = random.normal(keyH, (d, d), dtype=jnp.float64) \
        + 1j * random.normal(random.PRNGKey(8), (d, d), dtype=jnp.float64)
    H = H.astype(jnp.complex128)
    H_skew = _skew(H)  # anti-Hermitian

    X = U @ H_skew
    PX = project_to_tangent(U, X)

    diff = _frob_norm(PX - X)
    assert float(diff) < 1e-12, f"Pure tangent input should be fixed by projector; diff={float(diff)}"
