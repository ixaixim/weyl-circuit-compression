import pytest
import jax.numpy as jnp
import numpy as np

from weyl_compression.circuit.random_circuits import generate_random_circuit
from weyl_compression.mpo.mpo_builder import circuit_to_mpo


# @pytest.mark.parametrize("n_sites, n_layers", [(6, 25), (4, 10)])
# def test_mpo_is_normalized(n_sites, n_layers):
#     # Compute norm: <U|U> - This tests that the MPO representation preserves 
#     # the unitary property: for a properly normalized unitary operator U, 
#     # the inner product <U|U> = Tr(U†U) should equal 1.0

#     # Generate random target circuit
#     U_circuit = generate_random_circuit(
#         n_sites=n_sites,
#         n_layers=n_layers,
#         p_single=0.3,
#         p_two=0.4,
#         seed=43,
#         gate_name_single='U1',
#         gate_name_two='U2',
#         dtype=jnp.complex128
#     )
#     U_circuit.sort_layers()

#     # Convert target into MPO
#     U = circuit_to_mpo(U_circuit, svd_cutoff=0.0)
#     U.left_canonicalize()
#     U.normalize()

#     # Compute norm: <U|U>
#     U_dag = U.dagger()
#     L_env = jnp.eye(1).reshape(1, 1)
#     for i in range(n_sites):
#         L_env = jnp.einsum('ae, abcd, ecbf -> df', L_env, U[i], U_dag[i])

#     mpo_product = L_env[0, 0]  # Should be ~1.0
#     assert np.isclose(mpo_product, 1.0, atol=1e-10), \
#         f"MPO norm mismatch: got {mpo_product}"


# note: need to modify left_canonicalize (and right) in case the mpo is already left/right canonical
#       the norm is not properly returned/calculated. You can add a calculate_norm if it is already left/right canonical.
#       this code works only for ODD number of layers. (bc generate_random_circuit create a canonical MPO in the opposite direction)
@pytest.mark.parametrize("n_sites, n_layers", [
    (4, 5),  # Hilbert space dim = 2^4 = 16, Expected Norm = sqrt(16) = 4.0
    (6, 9),  # Hilbert space dim = 2^6 = 64, Expected Norm = sqrt(64) = 8.0
])
def test_unitary_mpo_frobenius_norm(n_sites, n_layers):
    """
    Tests that a left-canonicalized MPO from a unitary has the correct
    Frobenius norm, which is sqrt(2**n_sites).
    """
    # --- 1. Setup: Generate the MPO from a random unitary circuit ---
    U_circuit = generate_random_circuit(
        n_sites=n_sites,
        n_layers=n_layers,
        # Other params are not needed for the mock but included for context
        p_single=0.3,
        p_two=0.4,
        seed=43,
        gate_name_single='U1',
        gate_name_two='U2',
        dtype=jnp.complex128,
    )
    U_circuit.sort_layers()
    U_circuit.print_gates()
    U_mpo = circuit_to_mpo(U_circuit, svd_cutoff=0.0)

    actual_norm = U_mpo.left_canonicalize()

    # --- 3. Assertion: Check if the norm is correct ---
    expected_norm = np.sqrt(2**n_sites)

    assert np.isclose(actual_norm, expected_norm, atol=1e-8), \
        (f"MPO Frobenius norm mismatch for {n_sites} sites. "
         f"Expected: {expected_norm}, Got: {actual_norm}")
