import numpy as np
import jax.numpy as jnp
import pytest

from weyl_compression.circuit.random_circuits import generate_random_circuit
from weyl_compression.mpo.mpo_builder import circuit_to_mpo, get_id_mpo
from weyl_compression.optimization.gradient import sweeping_euclidean_gradient_bottom_up

# --------------------------------------------------------------------
# Helpers: numeric Wirtinger derivatives
# --------------------------------------------------------------------
def grad_G(f, G0, h: float = 1e-6):
    """
    4-point centered FD for Wirtinger ∂f/∂G at G0.
    """
    E_num = jnp.zeros_like(G0)
    for idx in np.ndindex(*G0.shape):
        delta = jnp.zeros_like(G0).at[idx].set(1.0)
        fp  = f(G0 +      h * delta)
        fm  = f(G0 -      h * delta)
        fip = f(G0 + 1j * h * delta)
        fim = f(G0 - 1j * h * delta)
        deriv = ((fp - fm) - 1j * (fip - fim)) / (4.0 * h)
        E_num = E_num.at[idx].set(deriv)
    return E_num

# no need to test ∂/∂G* here; bottom-up expects holomorphic

# --------------------------------------------------------------------
# Pytest test for sweeping_euclidean_gradient_bottom_up
# --------------------------------------------------------------------
@pytest.mark.parametrize("n_sites, n_layers_init, n_layers_target", [
    (5, 15, 4),
    (5, 2, 4),
])
def test_bottom_up_sweep_matches_numeric(n_sites, n_layers_init, n_layers_target):
    seed_init = 42
    seed_target = 44

    # Generate target and initial circuits
    target = generate_random_circuit(
        n_sites=n_sites,
        n_layers=n_layers_target,
        p_single=0.3,
        p_two=0.3,
        seed=seed_target,
        gate_name_single='U1',
        gate_name_two='U2',
        dtype=jnp.complex128
    )
    target.sort_layers()
    target.print_gates()
    mpo_target = circuit_to_mpo(target)
    U_target = target.to_matrix()
    mpo_target.left_canonicalize()

    init = generate_random_circuit(
        n_sites=n_sites,
        n_layers=n_layers_init,
        p_single=0.3,
        p_two=0.3,
        seed=seed_init,
        gate_name_single='U1',
        gate_name_two='U2',
        dtype=jnp.complex128
    )
    init.sort_layers()
    init.print_gates()

    # Call bottom-up sweep
    loss, grads_ordered, info = sweeping_euclidean_gradient_bottom_up(
        circuit=init,
        mpo_ref=mpo_target,
        max_bondim_env=128,
        svd_cutoff=0.0,
    )

    # Sanity checks
    assert info['num_gates'] == len(grads_ordered)
    assert isinstance(loss, jnp.ndarray)

    # Now compare each grad with numeric Wirtinger
    k = 0
    for layer_idx, layer in enumerate(init.layers):
        for gate_idx, gate in enumerate(layer.gates):
            G0 = gate.matrix
            # Define f: trace(target^† U(init with displaced gate))
            def f(displaced_gate):
                tmp = init.copy()
                tmp.layers[layer_idx].gates[gate_idx].matrix = displaced_gate
                U_mat = tmp.to_matrix()
                return jnp.trace(jnp.conjugate(U_target).T @ U_mat)

            E_num = grad_G(f, G0, h=1e-6)
            E_sweep = grads_ordered[k]

            # Reshape if two-qubit gate
            if gate.is_two_qubit():
                E_num = E_num.reshape(E_sweep.shape)

            # Compare
            assert jnp.allclose(E_sweep, E_num, rtol=1e-4, atol=1e-8), (
                f"Mismatch at gate {k} (layer {gate.layer_index}): "
                f"max|Δ| = {jnp.max(jnp.abs(E_sweep - E_num)):.2e}"
            )
            k += 1

    # Ensure we tested all gates
    assert k == info['num_gates']



from weyl_compression.optimization.gradient import sweeping_euclidean_gradient_top_down

# --------------------------------------------------------------------
# Pytest for sweeping_euclidean_gradient_top_down
# --------------------------------------------------------------------

@pytest.mark.parametrize("n_sites, n_layers_init, n_layers_target", [
    (5, 10, 4),
    (5, 2, 4),
])
def test_top_down_matches_numeric(
    n_sites, n_layers_init, n_layers_target
):
    seed_init   = 42
    seed_target = 44

    # --- target circuit & MPO ---
    target = generate_random_circuit(
        n_sites=n_sites,
        n_layers=n_layers_target,
        p_single=0.3,
        p_two=0.3,
        seed=seed_target,
        gate_name_single="U1",
        gate_name_two="U2",
        dtype=jnp.complex128,
    )
    target.sort_layers()
    mpo_target = circuit_to_mpo(target)
    mpo_target.left_canonicalize()      # ensures good conditioning

    # --- init circuit (to be differentiated) ---
    init = generate_random_circuit(
        n_sites=n_sites,
        n_layers=n_layers_init,
        p_single=0.3,
        p_two=0.3,
        seed=seed_init,
        gate_name_single="U1",
        gate_name_two="U2",
        dtype=jnp.complex128,
    )
    init.sort_layers()

    # --- call top-down sweep ---
    loss, grads_ordered, info = sweeping_euclidean_gradient_top_down(
        circuit=init,
        mpo_ref=mpo_target,
        max_bondim_env=128,
        svd_cutoff=1e-12,
    )

    # Sanity checks
    assert info["num_gates"] == len(grads_ordered)
    assert isinstance(loss, jnp.ndarray)

    # --- compare each sweep gradient to numeric FD ---
    k = 0
    for layer_idx, layer in enumerate(init.layers):
        for gate_idx, gate in enumerate(layer.gates):
            G0 = gate.matrix

            # f = Tr(V† · U(init_with_displaced_gate))
            def f(displaced_gate):
                tmp = init.copy()
                tmp.layers[layer_idx].gates[gate_idx].matrix = displaced_gate
                return jnp.trace(
                    mpo_target.dagger().to_matrix() @ tmp.to_matrix()
                )

            E_num   = grad_G(f, G0, h=1e-6)
            E_sweep = grads_ordered[k]

            # reshape numeric result for two-qubit gates
            if gate.is_two_qubit():
                E_num = E_num.reshape(E_sweep.shape)

            assert jnp.allclose(
                E_sweep,
                E_num,
                rtol=1e-4,
                atol=1e-8,
            ), (
                f"Mismatch at gate {k} (layer {gate.layer_index}); "
                f"max|Δ| = {jnp.max(jnp.abs(E_sweep - E_num)):.2e}"
            )
            k += 1

    # ensure all gates were checked
    assert k == info["num_gates"]



def setup_circ_and_mpo(n_sites, n_layers_init, n_layers_target, seed_init, seed_target):
    target = generate_random_circuit(
        n_sites=n_sites,
        n_layers=n_layers_target,
        p_single=0.3,
        p_two=0.3,
        seed=seed_target,
        gate_name_single="U1",
        gate_name_two="U2",
        dtype=jnp.complex128,
    )
    target.sort_layers()
    mpo_target = circuit_to_mpo(target)
    mpo_target.left_canonicalize()      # ensures good conditioning

    # --- init circuit (to be differentiated) ---
    init = generate_random_circuit(
        n_sites=n_sites,
        n_layers=n_layers_init,
        p_single=0.3,
        p_two=0.3,
        seed=seed_init,
        gate_name_single="U1",
        gate_name_two="U2",
        dtype=jnp.complex128,
    )
    init.sort_layers()
    return init, mpo_target

@pytest.mark.parametrize(
    "gradient_function",
    [sweeping_euclidean_gradient_bottom_up, sweeping_euclidean_gradient_top_down]
)
@pytest.mark.parametrize(
    "n_sites, n_layers_init, n_layers_target, seed_init, seed_target",
    [(5, 10, 30, 42, 44)],
)
def test_trace_calculation(
    gradient_function,
    n_sites,
    n_layers_init,
    n_layers_target,
    seed_init,
    seed_target,
):
    """
    Tests that the trace (loss) is computed correctly by comparing the result
    from sweeping functions against a ground truth from dense matrix contraction.
    """
    circuit, mpo_ref = setup_circ_and_mpo(
        n_sites=n_sites,
        n_layers_init=n_layers_init,
        n_layers_target=n_layers_target,
        seed_init=seed_init,
        seed_target=seed_target,
    )

    # ground-truth trace
    trace_exact = jnp.trace(mpo_ref.dagger().to_matrix() @ circuit.to_matrix())

    # calculation using the sweeping algorithm
    trace_num, _, _ = gradient_function(        
        circuit=circuit,
        mpo_ref=mpo_ref,
        max_bondim_env=128, 
        svd_cutoff=0.0
    )

    print(f"\nTesting trace for {gradient_function.__name__}:")
    print(f"  Trace from sweep: {trace_num}")
    print(f"  Exact trace:      {trace_exact}")
    
    np.testing.assert_allclose(
        trace_num,
        trace_exact,
        rtol=1e-7,
        atol=1e-10,
        err_msg=f"Trace calculation failed for {gradient_function.__name__}"
    )
