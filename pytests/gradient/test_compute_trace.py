import weyl_compression.jax_config

import numpy as np
import jax.numpy as jnp
import pytest

from weyl_compression.circuit.random_circuits import generate_random_circuit
from weyl_compression.mpo.mpo_builder import circuit_to_mpo
from weyl_compression.optimization.gradient import (
    compute_top_envs,
    compute_bottom_envs,
    compute_layer_boundary_environments,
    compute_gate_environment_tensor,
    compute_trace,
)

#NOTE: phase mismatch: the tr(Env^adj gate) and tr(U_ref^adj U__circuit) might differ by a phase. I think this might be due to SVD phase freedom 
# (i.e. you can multiply the same U col and V row by conjugate phases, preserving the SVD)
#NOTE: for longer circuits, the test might still fail due to bondim cuts. Increase the bondim.
#  ,
@pytest.mark.parametrize("n_sites, n_layers_init, n_layers_target", [
      (9, 10, 30), (6, 6, 10), (5, 6, 4), 
])
def test_compute_trace_matches_env_contraction(n_sites, n_layers_init, n_layers_target):
    """
    For each gate in a circuit, verify that the global trace
    Tr(V_target^† U_circuit) equals the local contraction tr(Env^*^T · G)
    where Env is the environment tensor for that gate and G its gate tensor.
    """
    seed_init = 123
    seed_target = 321

    # Build target circuit and MPO
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
    target_matrix = target.to_matrix()
    mpo_target = circuit_to_mpo(target, max_bondim=None)
    mpo_target.left_canonicalize()

    # Init circuit to be compared against target
    circuit = generate_random_circuit(
        n_sites=n_sites,
        n_layers=n_layers_init,
        p_single=0.3,
        p_two=0.3,
        seed=seed_init,
        gate_name_single="U1",
        gate_name_two="U2",
        dtype=jnp.complex128,
    )
    circuit.sort_layers()

    # Precompute top and bottom MPO environments per layer
    init_dir_top = "left_to_right" if mpo_target.dagger().is_right_canonical else "right_to_left"
    all_E_top = compute_top_envs(
        circuit=circuit,
        mpo_ref_top=mpo_target.dagger(),
        max_bondim_env=256,
        svd_cutoff=0.0,
        init_sweep_dir=init_dir_top,
    )
    all_E_bottom = compute_bottom_envs(
        circuit=circuit,
        max_bondim_env=256,
        svd_cutoff=0.0,
        init_sweep_dir="left_to_right",
    )

    # Global exact trace via dense contraction
    trace_exact = jnp.trace(target_matrix.conjugate().T @ circuit.to_matrix())

    # For every gate, compute its environment and compare traces.
    global_phase = None
    tol_phase_mag = 1e-12
    for l, layer in enumerate(circuit.layers):
        E_top_l = all_E_top[l]
        E_bottom_l = all_E_bottom[l]

        # Precompute boundary environments on both sides within the layer
        E_left_boundaries = compute_layer_boundary_environments(
            E_top=E_top_l, E_bottom=E_bottom_l, layer=layer, side="left"
        )
        E_right_boundaries = compute_layer_boundary_environments(
            E_top=E_top_l, E_bottom=E_bottom_l, layer=layer, side="right"
        )

        for gate in layer.iterate_gates(reverse=False):
            leftmost_qb = min(gate.qubits)
            rightmost_qb = max(gate.qubits)

            E_left = E_left_boundaries[leftmost_qb]
            E_right = E_right_boundaries[rightmost_qb]
            assert E_left is not None and E_right is not None

            Env = compute_gate_environment_tensor(
                gate_qubits=gate.qubits,
                E_top_layer=E_top_l,
                E_bottom_layer=E_bottom_l,
                E_left_boundary=E_left,
                E_right_boundary=E_right,
            )

            # Ensure Env has the same physical index structure as the gate tensor
            # compute_trace handles (2,2) and (2,2,2,2)
            trace_local = compute_trace(Env, gate.tensor)

            if jnp.abs(trace_exact) <= tol_phase_mag or jnp.abs(trace_local) <= tol_phase_mag:
                np.testing.assert_allclose(
                    trace_local,
                    trace_exact,
                    rtol=1e-7,
                    atol=1e-10,
                    err_msg=(
                        f"Trace mismatch at layer {l}, gate on qubits {gate.qubits}.\n"
                        f"local: {trace_local}, exact: {trace_exact}"
                    ),
                )
                continue

            ratio = trace_local / trace_exact
            ratio_mag = np.abs(ratio)

            np.testing.assert_allclose(
                ratio_mag,
                1.0,
                rtol=1e-7,
                atol=1e-10,
                err_msg=(
                    f"Trace ratio not unit magnitude at layer {l}, gate {gate.qubits}.\n"
                    f"ratio: {ratio}, |ratio|: {ratio_mag}"
                ),
            )

            ratio_unit = ratio / ratio_mag
            if global_phase is None:
                global_phase = ratio_unit
            else:
                np.testing.assert_allclose(
                    ratio_unit,
                    global_phase,
                    rtol=1e-7,
                    atol=1e-10,
                    err_msg=(
                        f"Inconsistent global phase at layer {l}, gate {gate.qubits}.\n"
                        f"ratio: {ratio_unit}, expected: {global_phase}"
                    ),
                )

            np.testing.assert_allclose(
                trace_local,
                global_phase * trace_exact,
                rtol=1e-7,
                atol=1e-10,
                err_msg=(
                    f"Trace mismatch (up to phase) at layer {l}, gate {gate.qubits}.\n"
                    f"local: {trace_local}, exact: {trace_exact}, phase: {global_phase}"
                ),
            )
