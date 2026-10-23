import weyl_compression.jax_config
import numpy as np
from weyl_compression.circuit.circuit_dataclasses import Circuit, Gate, GateLayer
from qiskit.synthesis import OneQubitEulerDecomposer

def rzz_angle(U):
    """
    Extract the angle theta from a 4x4 numpy array representing RZZ(theta).
    Assumes U is diagonal (pure ZZ interaction).
    """
    # Extract phases of diagonal entries
    phi00 = np.angle(U[0, 0]) # lies in [-π, π]
    phi11 = np.angle(U[1, 1]) # lies in [-π, π]

    # RZZ has the pattern: diag(exp(-iθ/2), exp(+iθ/2), exp(+iθ/2), exp(-iθ/2))
    # Difference phi11 - phi00 = (iθ/2) - (-iθ/2) = iθ (in phase) -> theta
    theta = phi11 - phi00

    # Wrap to [-π, π]
    theta = np.arctan2(np.sin(theta), np.cos(theta))
    return theta

def _get_euler_angles(matrix, decomposer):
    """
    Decompose 2x2 matrix into (th, psi, ph) for Rz(th) @ Rx(psi) @ Rz(ph).
    Qiskit 'ZXZ' returns (theta, phi, lam) for Rz(phi) Rx(theta) Rz(lam).
    """
    if matrix is None:
        return np.zeros(3)
    
    # Qiskit returns (theta, phi, lam) for Rz(phi) Rx(theta) Rz(lam)
    theta, phi, lam = decomposer.angles(matrix)
    
    # Our convention stores (th, psi, ph) for Rz(th) Rx(psi) Rz(ph),
    # so: th=phi, psi=theta, ph=lam
    return np.array([phi, theta, lam])

def _create_parametrized_gate(matrix, qubits, layer_idx):
    """Creates a Gate with 'RZZ' parametrization."""
    theta = rzz_angle(matrix)

    # Param structure: (theta,)
    # Cast to arrays to ensure they are mutable/compatible with JAX pytrees later
    params = (
        np.array([theta]),
    )
    
    return Gate(
        matrix=matrix,
        qubits=qubits,
        layer_index=layer_idx,
        name="RZZ",
        params=params
    )

def _create_parametrized_1q_gate(matrix, qubits, layer_idx, decomposer):
    """Creates a Gate with '1q' parametrization."""
    # Decompose the single qubit gate
    params_euler = _get_euler_angles(matrix, decomposer)

    return Gate(
        matrix=matrix,
        qubits=qubits,
        layer_index=layer_idx,
        name="1q",
        params=(params_euler,)
    )

def _create_identity_1q_gate(qubit, layer_idx, dtype):
    """
    Identity-initialized '1q' gate (Rz(0) Rx(0) Rz(0) = I).
    Used as the optimizable basis-change layer between the two RZZ rotations
    of a doubled 2Q block. In the hybrid optimizer the 1Q gate is updated
    Riemannian-ly from its matrix, so only `matrix` matters here.
    """
    return Gate(
        matrix=np.eye(2, dtype=dtype),
        qubits=(qubit,),
        layer_index=layer_idx,
        name="1q",
        params=(np.array([0.0, 0.0, 0.0]),)
    )

def _create_identity_rzz_gate(qubits, layer_idx, dtype):
    """Identity-initialized 'RZZ' gate (RZZ(0) = I), the second rotation."""
    return Gate(
        matrix=np.eye(4, dtype=dtype),
        qubits=qubits,
        layer_index=layer_idx,
        name="RZZ",
        params=(np.array([0.0]),)
    )

def parameterize_rzz_circuit(orig: Circuit, second_rotation: bool = False) -> Circuit:
    """
    Decomposes the circuit into 'RZZ' and '1q' gates.
    Does NOT merge layers.
    2Q Layers -> RZZ.
    1Q Layers -> 1q.

    If `second_rotation` is True, each 2Q (RZZ) layer is followed by an
    identity-initialized 1Q layer and a second identity-initialized RZZ layer
    on the same bonds. The block `RZZ(theta1) . (1Q (x) 1Q) . RZZ(theta2)` can
    reach any 2Q gate with two nonzero Weyl coordinates (a *face* of the Weyl
    chamber), which a single RZZ cannot. The inserted layers start as the
    identity, so the circuit is unchanged at initialization (warm start).
    """
    sorted_layers = sorted(orig.layers, key=lambda layer: layer.layer_index)
    new_layers = []
    decomposer = OneQubitEulerDecomposer(basis='ZXZ')
    
    for current_layer in sorted_layers:
        new_gates = []
        
        # Check if the layer contains 2-qubit gates or 1-qubit gates
        # We assume uniform layers (all gates are same type roughly)
        if not current_layer.gates:
            # Empty layer, just copy structure
            new_layer = GateLayer(
                layer_index=len(new_layers),
                is_odd=current_layer.is_odd,
                gates=[],
                n_sites=current_layer.n_sites
            )
            new_layers.append(new_layer)
            continue
            
        first_gate = current_layer.gates[0]
        
        if len(first_gate.qubits) == 2:
            # Interaction Layer (2Q)
            for gate in current_layer.iterate_gates():
                new_gate = _create_parametrized_gate(gate.matrix, gate.qubits, len(new_layers))
                new_gates.append(new_gate)
                
        elif len(first_gate.qubits) == 1:
            # Field Layer (1Q)
            for gate in current_layer.iterate_gates():
                new_gate = _create_parametrized_1q_gate(
                    gate.matrix, gate.qubits, len(new_layers), decomposer
                )
                new_gates.append(new_gate)
        
        else:
            raise ValueError(f"Unsupported gate size: {len(first_gate.qubits)} qubits")

        new_layer = GateLayer(
            layer_index=len(new_layers),
            is_odd=current_layer.is_odd,
            gates=new_gates,
            n_sites=current_layer.n_sites
        )
        new_layers.append(new_layer)

        # Add a second native RZZ rotation per 2Q gate, separated by an
        # optimizable (identity-initialized) 1Q layer. Both inserted layers
        # start as the identity, so the circuit is unchanged at init.
        if second_rotation and len(first_gate.qubits) == 2:
            dtype = first_gate.matrix.dtype
            n_sites = current_layer.n_sites

            # (a) Basis-change 1Q layer (identity init) on every qubit this
            #     2Q layer touches.
            touched = sorted({q for g in new_gates for q in g.qubits})
            mid_gates = [
                _create_identity_1q_gate(q, len(new_layers), dtype)
                for q in touched
            ]
            new_layers.append(GateLayer(
                layer_index=len(new_layers),
                is_odd=False,
                gates=mid_gates,
                n_sites=n_sites,
            ))

            # (b) Second RZZ layer (identity init) on the same bonds.
            second_gates = [
                _create_identity_rzz_gate(g.qubits, len(new_layers), dtype)
                for g in new_gates
            ]
            new_layers.append(GateLayer(
                layer_index=len(new_layers),
                is_odd=current_layer.is_odd,
                gates=second_gates,
                n_sites=n_sites,
            ))

    return Circuit(n_sites=orig.n_sites, dtype=orig.dtype, layers=new_layers)
