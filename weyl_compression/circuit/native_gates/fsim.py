import weyl_compression.jax_config
import numpy as np
from weyl_compression.circuit.circuit_dataclasses import Circuit, Gate, GateLayer
from qiskit.synthesis import OneQubitEulerDecomposer

def fsim_angles(U):
    """
    Extract the angles theta_xy and theta_z from a 4x4 numpy array representing 
    the XXZ evolution U = exp(-i * (theta_xy * (XX + YY) + theta_z * ZZ)).
    """
    # U[0,0] = exp(-i * theta_z)
    theta_z = -np.angle(U[0, 0])
    
    # The Hamiltonian is
    #
    #     H = theta_xy * (XX + YY) + theta_z * ZZ
    #
    # ZZ is diagonal: +1 if the two bits are equal, -1 if they differ,
    #     ZZ = diag(1, -1, -1, 1)
    #
    # XX + YY only touches the "odd parity" states {|01>, |10>}.
    # Using Y|0> = i|1> and Y|1> = -i|0>:
    #     XX|01> = |10>
    #     YY|01> = (i)(-i)|10> = |10>      ->  (XX + YY)|01> = 2|10>
    #     XX|00> = |11>
    #     YY|00> = (i)(i)|11> = -|11>      ->  (XX + YY)|00> = 0
    # (the same happens for |11>)
    #
    # so XX + YY = 2 * (|01><10| + |10><01|)
    # It swaps |01> <-> |10> with weight 2 and kills |00>, |11>.
    #
    # Putting it together, H is block diagonal:
    #
    # * On {|00>, |11>}: H = theta_z * I
    #   This is why U[0,0] = exp(-i * theta_z).
    #
    # * On {|01>, |10>}: the ZZ term gives -theta_z on the diagonal, and the
    #   flip term gives 2 * theta_xy off the diagonal:
    #
    #       H_sub = [[-theta_z,    2*theta_xy],
    #                [2*theta_xy, -theta_z   ]]
    #             = -theta_z * I + 2 * theta_xy * X    
    phase = np.exp(1j * theta_z)
    c = (U[1, 1] / phase).real
    s = (U[1, 2] / (-1j * phase)).real
    
    two_theta_xy = np.arctan2(s, c)
    theta_xy = two_theta_xy / 2.0
    
    return theta_xy, theta_z

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
    """Creates a Gate with 'fSim' parametrization."""
    theta_xy, theta_z = fsim_angles(matrix)
    
    # Param structure: theta_xy, theta_z
    params = (
        np.array([theta_xy, theta_z]),
    )
    
    return Gate(
        matrix=matrix,
        qubits=qubits,
        layer_index=layer_idx,
        name="fSim",
        params=params
    )

def _create_parametrized_1q_gate(matrix, qubits, layer_idx, decomposer):
    """Creates a Gate with '1q' parametrization."""
    params_euler = _get_euler_angles(matrix, decomposer)
    
    return Gate(
        matrix=matrix,
        qubits=qubits,
        layer_index=layer_idx,
        name="1q",
        params=(params_euler,)
    )

def parameterize_fsim_circuit(orig: Circuit) -> Circuit:
    """
    Decomposes the circuit into 'fSim' and '1q' gates.
    Does NOT merge layers.
    2Q Layers -> fSim.
    1Q Layers -> 1q.
    """
    sorted_layers = sorted(orig.layers, key=lambda layer: layer.layer_index)
    new_layers = []
    decomposer = OneQubitEulerDecomposer(basis='ZXZ')
    
    for current_layer in sorted_layers:
        new_gates = []
        
        if not current_layer.gates:
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

    return Circuit(n_sites=orig.n_sites, dtype=orig.dtype, layers=new_layers)
