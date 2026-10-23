import jax.numpy as jnp
import numpy as np
import pytest

import weyl_compression.jax_config  # ensure JAX dtype/config is initialized

from weyl_compression.optimization.utils import overlap_to_loss


def _paulis(dtype):
    X = jnp.array([[0, 1], [1, 0]], dtype=dtype)
    Y = jnp.array([[0, -1j], [1j, 0]], dtype=dtype)
    Z = jnp.array([[1, 0], [0, -1]], dtype=dtype)
    I = jnp.eye(2, dtype=dtype)
    return X, Y, Z, I


def _rot_from_generator(theta: float, P: jnp.ndarray, scale: float, dtype) -> jnp.ndarray:
    # exp(-i * scale * theta * P) for involutory P (P^2 = I)
    c = jnp.cos(scale * theta)
    s = jnp.sin(scale * theta)
    I = jnp.eye(P.shape[0], dtype=dtype)
    return c * I - 1j * s * P


def _dU_dtheta(theta: float, P: jnp.ndarray, scale: float, dtype) -> jnp.ndarray:
    U = _rot_from_generator(theta, P, scale, dtype)
    return (-1j * scale) * (P @ U)


@pytest.mark.parametrize("axis", ["x", "y", "z"])  # test RX/RY/RZ
@pytest.mark.parametrize("is_normalized", [False, True])
def test_hst_grad_matches_central_fd(axis, is_normalized):
    # Problem setup: 1-qubit (d=2), single gate U(theta) vs fixed reference A
    dtype = jnp.complex128
    X, Y, Z, I = _paulis(dtype)

    # choose generator and scale consistent with parametrized_adam.utils (default exp_scale=0.5)
    scale = 0.5
    P = {"x": X, "y": Y, "z": Z}[axis]

    # reference unitary A: compose simple random-ish SU(2) from fixed angles
    alpha, beta, gamma = 0.37, -0.91, 1.13
    A = (_rot_from_generator(alpha, X, scale, dtype)
         @ _rot_from_generator(beta, Y, scale, dtype)
         @ _rot_from_generator(gamma, Z, scale, dtype))

    # variable parameter
    theta0 = 0.42

    # overlap z(theta) = Tr(A^\dagger U(theta))
    def z(theta: float) -> jnp.ndarray:
        U = _rot_from_generator(theta, P, scale, dtype)
        return jnp.trace(jnp.conjugate(A).T @ U)

    # HST loss: 1 - |z|^2 / denom
    n_sites = 1
    d = 2 ** n_sites
    denom = (d if is_normalized else d * d)

    def loss(theta: float) -> jnp.ndarray:
        return overlap_to_loss(z(theta), kind="HST", n_sites=n_sites, normalize=is_normalized)

    # central finite difference for d loss / d theta (second-order)
    eps = 1e-6
    fd = (loss(theta0 + eps) - loss(theta0 - eps)) / (2.0 * eps)

    # analytic chain-rule gradient for HST
    z0 = z(theta0)
    dU = _dU_dtheta(theta0, P, scale, dtype)

    # For z = Tr(A^\dagger U), \partial z / \partial U = A^\dagger (elementwise: conj(A))
    dz_dG = jnp.conjugate(A)  # shape (2,2)
    dz_dtheta = jnp.trace(dz_dG.T @ dU)  # equals Tr(A^\dagger dU/dθ)

    analytic = -(2.0 / denom) * jnp.real(jnp.conjugate(z0) * dz_dtheta)

    # Compare numeric vs analytic
    np.testing.assert_allclose(
        np.asarray(fd),
        np.asarray(analytic),
        rtol=1e-6,
        atol=1e-9,
        err_msg=f"HST grad mismatch for axis={axis}, is_normalized={is_normalized}"
    )

