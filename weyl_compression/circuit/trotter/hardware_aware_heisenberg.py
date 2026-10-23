import weyl_compression.jax_config
import jax.numpy as jnp
import jax.scipy.linalg as jsp
from typing import List, Optional
from weyl_compression.circuit.circuit_dataclasses import GateLayer, Gate, Circuit
from weyl_compression.circuit.trotter.gates import _local_evolution_xyz, pauli_x, pauli_y, pauli_z

def _single_qubit_heisenberg_field_matrix(hx: float, hy: float, hz: float, coeff: float, dtype: jnp.dtype) -> jnp.ndarray:
    """exp(-i * coeff * (hx*X + hy*Y + hz*Z))"""
    X = pauli_x(dtype)
    Y = pauli_y(dtype)
    Z = pauli_z(dtype)
    H = hx * X + hy * Y + hz * Z
    return jsp.expm(-1j * coeff * H)

def _two_qubit_heisenberg_interaction_matrix(J: float, Delta: float, coeff: float, dtype: jnp.dtype) -> jnp.ndarray:
    """exp(-i * coeff * J * (XX + YY + Delta*ZZ))"""
    return _local_evolution_xyz(
        Jx=J, Jy=J, Jz=J*Delta,
        hx_left=0.0, hy_left=0.0, hz_left=0.0,
        hx_right=0.0, hy_right=0.0, hz_right=0.0,
        coeff=coeff, dtype=dtype
    )

def _add_interaction_layer(n_sites: int, J: float, Delta: float, coeff: float, parity: str, layer_idx: int, dtype: jnp.dtype) -> GateLayer:
    U = _two_qubit_heisenberg_interaction_matrix(J, Delta, coeff, dtype)
    is_odd = (parity == "odd")
    bonds = range(1, n_sites - 1, 2) if is_odd else range(0, n_sites - 1, 2)
    layer = GateLayer(layer_index=layer_idx, is_odd=is_odd, n_sites=n_sites)
    for left in bonds:
        gate = Gate(matrix=U, qubits=(left, left + 1), layer_index=layer_idx, name="Uxxz")
        layer.add_gate(gate)
    return layer

def _add_field_layer(n_sites: int, hx: float, hy: float, hz: float, coeff: float, layer_idx: int, dtype: jnp.dtype) -> GateLayer:
    U = _single_qubit_heisenberg_field_matrix(hx, hy, hz, coeff, dtype)
    layer = GateLayer(layer_index=layer_idx, is_odd=False, n_sites=n_sites)
    for q in range(n_sites):
        gate = Gate(matrix=U, qubits=(q,), layer_index=layer_idx, name="Rfield")
        layer.add_gate(gate)
    return layer

def heisenberg_hardware_aware_trotter_layers(
    n_sites: int,
    J: float,
    Delta: float,
    hx: float,
    dt: float,
    reps: int,
    *,
    hy: float = 0.0,
    hz: float = 0.0,
    order: int = 2,
    dtype: Optional[jnp.dtype] = None
) -> List[GateLayer]:
    """
    Hardware-aware Trotterization for the Heisenberg XXZ model with general field:
    H = J * sum(X_i X_{i+1} + Y_i Y_{i+1} + Delta * Z_i Z_{i+1}) + sum(hx*X_i + hy*Y_i + hz*Z_i)
    
    Splitting Strategy:
    - Interaction-Outside (IO): |J| > sqrt(hx^2 + hy^2 + hz^2)
    - Field-Outside (FO): |J| <= sqrt(hx^2 + hy^2 + hz^2)
    
    Orders supported: 1, 2, 4.
    """
    if n_sites % 2:
        raise ValueError(f"n_sites must be even (got {n_sites})")
    if order not in (1, 2, 4):
        raise ValueError(f"order must be 1, 2, or 4 (got {order})")
    
    if dtype is None:
        dtype = jnp.complex128
    
    h_norm = jnp.sqrt(hx**2 + hy**2 + hz**2)
    interaction_outside = (jnp.abs(J) > h_norm)
    
    layers = []
    layer_idx = 0
    
    def add_int(coeff, parity):
        nonlocal layer_idx
        layers.append(_add_interaction_layer(n_sites, J, Delta, coeff, parity, layer_idx, dtype))
        layer_idx += 1
        
    def add_field(coeff):
        nonlocal layer_idx
        layers.append(_add_field_layer(n_sites, hx, hy, hz, coeff, layer_idx, dtype))
        layer_idx += 1

    # --- 1st Order ---
    if order == 1:
        for _ in range(reps):
            if interaction_outside:
                add_int(dt, "even"); add_int(dt, "odd")
                add_field(dt)
            else:
                add_field(dt)
                add_int(dt, "even"); add_int(dt, "odd")
                
    # --- 2nd Order ---
    elif order == 2:
        if interaction_outside:
            # IO: E(dt/2) O(dt/2) F(dt) O(dt/2) E(dt/2)
            # Merging boundaries: E(dt/2) + E(dt/2) -> E(dt)
            add_int(dt/2, "even")
            for r in range(reps):
                add_int(dt/2, "odd")
                add_field(dt)
                add_int(dt/2, "odd")
                if r < reps - 1:
                    add_int(dt, "even")
                else:
                    add_int(dt/2, "even")
        else:
            # FO: F(dt/2) E(dt/2) O(dt) E(dt/2) F(dt/2)
            # Merging boundaries: F(dt/2) + F(dt/2) -> F(dt)
            add_field(dt/2)
            for r in range(reps):
                add_int(dt/2, "even")
                add_int(dt, "odd")
                add_int(dt/2, "even")
                if r < reps - 1:
                    add_field(dt)
                else:
                    add_field(dt/2)
                    
    # --- 4th Order (Suzuki Fractal) ---
    elif order == 4:
        cbrt4 = 4.0**(1.0/3.0)
        p = 1.0 / (4.0 - cbrt4)
        k1 = p * dt
        k2 = (1.0 - 4.0 * p) * dt
        
        steps = []
        for _ in range(reps):
            steps.extend([k1, k1, k2, k1, k1])
        
        num_steps = len(steps)
        if interaction_outside:
            # S4 based on symmetric S2: E(s/2) O(s/2) F(s) O(s/2) E(s/2)
            add_int(steps[0]/2, "even")
            for i, step in enumerate(steps):
                add_int(step/2, "odd")
                add_field(step)
                add_int(step/2, "odd")
                
                if i < num_steps - 1:
                    add_int((step + steps[i+1])/2, "even")
                else:
                    add_int(step/2, "even")
        else:
            # S4 based on symmetric S2: F(s/2) E(s/2) O(s) E(s/2) F(s/2)
            add_field(steps[0]/2)
            for i, step in enumerate(steps):
                add_int(step/2, "even")
                add_int(step, "odd")
                add_int(step/2, "even")
                
                if i < num_steps - 1:
                    add_field((step + steps[i+1])/2)
                else:
                    add_field(step/2)
                
    return layers

def heisenberg_hardware_aware_trotter_circuit(
    n_sites: int,
    J: float,
    Delta: float,
    hx: float,
    dt: float,
    reps: int,
    *,
    hy: float = 0.0,
    hz: float = 0.0,
    order: int = 2,
    dtype: Optional[jnp.dtype] = None
) -> Circuit:
    layers = heisenberg_hardware_aware_trotter_layers(
        n_sites=n_sites, J=J, Delta=Delta, hx=hx, hy=hy, hz=hz, dt=dt, reps=reps, order=order, dtype=dtype
    )
    return Circuit(
        n_sites=n_sites,
        layers=layers,
        dtype=dtype or jnp.complex128
    )

__all__ = [
    "heisenberg_hardware_aware_trotter_layers",
    "heisenberg_hardware_aware_trotter_circuit",
]
