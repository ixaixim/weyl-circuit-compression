import weyl_compression.jax_config
import jax.numpy as jnp
import jax.scipy.linalg as jsp
from typing import List, Optional
from weyl_compression.circuit.circuit_dataclasses import GateLayer, Gate, Circuit

def _single_qubit_ising_field_matrix(hx: float, hz: float, coeff: float, dtype: jnp.dtype) -> jnp.ndarray:
    """exp(-i * coeff * (hx*X + hz*Z))"""
    X = jnp.array([[0, 1], [1, 0]], dtype=dtype)
    Z = jnp.array([[1, 0], [0, -1]], dtype=dtype)
    H = hx * X + hz * Z
    return jsp.expm(-1j * coeff * H)

def _two_qubit_ising_interaction_matrix(J: float, coeff: float, dtype: jnp.dtype) -> jnp.ndarray:
    """exp(-i * coeff * J * ZZ)"""
    Z = jnp.array([[1, 0], [0, -1]], dtype=dtype)
    ZZ = jnp.kron(Z, Z)
    return jsp.expm(-1j * coeff * J * ZZ)

def _add_interaction_layer(n_sites: int, J: float, coeff: float, parity: str, layer_idx: int, dtype: jnp.dtype) -> GateLayer:
    U = _two_qubit_ising_interaction_matrix(J, coeff, dtype)
    is_odd = (parity == "odd")
    bonds = range(1, n_sites - 1, 2) if is_odd else range(0, n_sites - 1, 2)
    layer = GateLayer(layer_index=layer_idx, is_odd=is_odd, n_sites=n_sites)
    for left in bonds:
        gate = Gate(matrix=U, qubits=(left, left + 1), layer_index=layer_idx, name="Rzz")
        layer.add_gate(gate)
    return layer

def _add_field_layer(n_sites: int, hx: float, hz: float, coeff: float, layer_idx: int, dtype: jnp.dtype) -> GateLayer:
    U = _single_qubit_ising_field_matrix(hx, hz, coeff, dtype)
    layer = GateLayer(layer_index=layer_idx, is_odd=False, n_sites=n_sites)
    for q in range(n_sites):
        gate = Gate(matrix=U, qubits=(q,), layer_index=layer_idx, name="Rfield")
        layer.add_gate(gate)
    return layer

def ising_hardware_aware_trotter_layers(
    n_sites: int,
    J: float,
    hx: float,
    hz: float,
    dt: float,
    reps: int,
    order: int = 2,
    dtype: Optional[jnp.dtype] = None
) -> List[GateLayer]:
    """
    Hardware-aware Trotterization for the Ising model:
    H = J * sum(Z_i Z_{i+1}) + hx * sum(X_i) + hz * sum(Z_i)
    
    Splitting Strategy:
    - Interaction-Outside (IO): |J| > sqrt(hx^2 + hz^2)
    - Field-Outside (FO): |J| <= sqrt(hx^2 + hz^2)
    
    Orders supported: 1, 2, 4.
    """
    if n_sites % 2:
        raise ValueError(f"n_sites must be even (got {n_sites})")
    if order not in (1, 2, 4):
        raise ValueError(f"order must be 1, 2, or 4 (got {order})")
    
    if dtype is None:
        dtype = jnp.complex128
    
    h_norm = jnp.sqrt(hx**2 + hz**2)
    interaction_outside = (jnp.abs(J) > h_norm)
    
    layers = []
    layer_idx = 0
    
    def add_int(coeff, parity):
        nonlocal layer_idx
        layers.append(_add_interaction_layer(n_sites, J, coeff, parity, layer_idx, dtype))
        layer_idx += 1
        
    def add_field(coeff):
        nonlocal layer_idx
        layers.append(_add_field_layer(n_sites, hx, hz, coeff, layer_idx, dtype))
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
        # Optimized with boundary merging
        if interaction_outside:
            # E(dt/2) O(dt/2) F(dt) [E(dt) O(dt) F(dt)]^{reps-1} O(dt/2) E(dt/2)
            add_int(dt/2, "even"); add_int(dt/2, "odd")
            for r in range(reps):
                add_field(dt)
                if r < reps - 1:
                    add_int(dt, "even"); add_int(dt, "odd")
                else:
                    add_int(dt/2, "odd"); add_int(dt/2, "even")
        else:
            # F(dt/2) [E(dt) O(dt) F(dt)]^{reps-1} E(dt) O(dt) F(dt/2)
            add_field(dt/2)
            for r in range(reps):
                add_int(dt, "even"); add_int(dt, "odd")
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
            # IO S2(step) = E(step/2) O(step/2) F(step) O(step/2) E(step/2)
            # Boundary absorption: ... O(si/2) E(si/2) E(sj/2) O(sj/2) ... = ... O((si+sj)/2) E((si+sj)/2) ...
            add_int(steps[0]/2, "even")
            add_int(steps[0]/2, "odd")
            for i, step in enumerate(steps):
                add_field(step)
                if i < num_steps - 1:
                    add_int((step + steps[i+1])/2, "odd")
                    add_int((step + steps[i+1])/2, "even")
                else:
                    add_int(step/2, "odd")
                    add_int(step/2, "even")
        else:
            # FO S2(step) = F(step/2) E(step) O(step) F(step/2)
            # Boundary absorption: ... F(si/2) F(sj/2) ... = ... F((si+sj)/2) ...
            add_field(steps[0]/2)
            for i, step in enumerate(steps):
                add_int(step, "even")
                add_int(step, "odd")
                if i < num_steps - 1:
                    add_field((step + steps[i+1])/2)
                else:
                    add_field(step/2)
                
    return layers

def ising_hardware_aware_trotter_circuit(
    n_sites: int,
    J: float,
    hx: float,
    hz: float,
    dt: float,
    reps: int,
    order: int = 2,
    dtype: Optional[jnp.dtype] = None
) -> Circuit:
    layers = ising_hardware_aware_trotter_layers(
        n_sites=n_sites, J=J, hx=hx, hz=hz, dt=dt, reps=reps, order=order, dtype=dtype
    )
    return Circuit(
        n_sites=n_sites,
        layers=layers,
        dtype=dtype or jnp.complex128
    )

__all__ = [
    "ising_hardware_aware_trotter_layers",
    "ising_hardware_aware_trotter_circuit",
]
