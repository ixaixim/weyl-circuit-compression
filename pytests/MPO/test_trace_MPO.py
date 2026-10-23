# test_hilbert_schmidt_mpo.py
import pytest
import numpy as np
import jax.numpy as jnp

# Ensure JAX is configured (dtype, etc.) the same way as your project
import weyl_compression.jax_config  # noqa: F401

from weyl_compression.circuit.random_circuits import generate_random_circuit
from weyl_compression.mpo.mpo_builder import circuit_to_mpo
from weyl_compression.tensor_network.core_ops_helpers import hs_inner_product_from_mpo


@pytest.mark.parametrize(
    "n_sites, seed_A, seed_B, n_layers, p_single, p_two",
    [
        (6, 42, 43, 3, 0.3, 0.3),
        # add more quick cases if you like:
        # (3, 1, 2, 2, 0.5, 0.5),
    ],
)
def test_hilbert_schmidt_product_mpo_matches_matrix(n_sites, seed_A, seed_B, n_layers, p_single, p_two):
    # ---- Build circuit A ----
    A_circuit = generate_random_circuit(
        n_sites=n_sites,
        n_layers=n_layers,
        p_single=p_single,
        p_two=p_two,
        seed=seed_A,
        gate_name_single="U1",
        gate_name_two="U2",
        dtype=jnp.complex128,
    )
    A_circuit.sort_layers()
    A_mpo = circuit_to_mpo(A_circuit)
    A_mpo.left_canonicalize()
    A_matrix = np.asarray(A_circuit.to_matrix())

    # ---- Build circuit B ----
    B_circuit = generate_random_circuit(
        n_sites=n_sites,
        n_layers=n_layers,
        p_single=p_single,
        p_two=p_two,
        seed=seed_B,
        gate_name_single="U1",
        gate_name_two="U2",
        dtype=jnp.complex128,
    )
    B_circuit.sort_layers()
    B_mpo = circuit_to_mpo(B_circuit)
    B_mpo.left_canonicalize()
    B_matrix = np.asarray(B_circuit.to_matrix())

    # ---- Explicit Hilbert–Schmidt inner product: Tr(A^\dagger B) ----
    explicit = np.trace(np.conjugate(A_matrix).T @ B_matrix)

    # ---- MPO contraction version ----
    mpo_val = hs_inner_product_from_mpo(A_mpo, B_mpo)
    mpo_val = np.asarray(mpo_val)  # convert JAX array to NumPy scalar

    # ---- Assertions ----
    # 1) Numerical equality
    np.testing.assert_allclose(mpo_val, explicit, rtol=1e-9, atol=1e-9)

    # 2) Sanity checks: scalar-like and finite
    assert np.ndim(mpo_val) == 0
    assert np.isfinite(mpo_val)

