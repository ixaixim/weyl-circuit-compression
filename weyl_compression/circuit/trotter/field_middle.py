import weyl_compression.jax_config  # ensures JAX defaults are consistent

from typing import List, Sequence, Optional

import jax.numpy as jnp
import jax.scipy.linalg as jsp

from weyl_compression.circuit.circuit_dataclasses import GateLayer, Gate, Circuit
from weyl_compression.circuit.trotter.gates import _local_evolution_xyz

# -----------------------------------------------------------------------------
# Local Helpers
# -----------------------------------------------------------------------------

def _pauli_x(dtype: jnp.dtype) -> jnp.ndarray:
    return jnp.array([[0, 1], [1, 0]], dtype=dtype)

def _pauli_y(dtype: jnp.dtype) -> jnp.ndarray:
    return jnp.array([[0, -1j], [1j, 0]], dtype=dtype)

def _pauli_z(dtype: jnp.dtype) -> jnp.ndarray:
    return jnp.array([[1, 0], [0, -1]], dtype=dtype)

def _get_pauli(name: str, dtype: jnp.dtype) -> jnp.ndarray:
    if name == 'X': return _pauli_x(dtype)
    if name == 'Y': return _pauli_y(dtype)
    if name == 'Z': return _pauli_z(dtype)
    raise ValueError(f"Unknown Pauli {name}")

# -----------------------------------------------------------------------------
# Layer Builders
# -----------------------------------------------------------------------------

def _single_layer_interaction(
    *,
    n_sites: int,
    Jx: float,
    Jy: float,
    Jz: float,
    coeff: float,
    parity: str,
    layer_idx: int,
    dtype: jnp.dtype,
) -> GateLayer:
    """
    Creates a layer of two-qubit interaction gates (Jx XX + Jy YY + Jz ZZ terms).
    Field terms are explicitly EXCLUDED (hx=hy=hz=0, include_field=False).
    """
    if n_sites % 2:
        raise ValueError(f"n_sites must be even, got {n_sites}")
    
    # Determine bonds based on parity
    if parity == "even":
        bonds = range(0, n_sites - 1, 2)
        is_odd_layer = False
    elif parity == "odd":
        bonds = range(1, n_sites - 1, 2)
        is_odd_layer = True
    else:
        raise ValueError(f"parity must be 'even' or 'odd', got {parity}")

    # Build the evolution matrix: exp(-i * coeff * H_int)
    # We pass hx/y/z_left/right=0.0 to ensure only Jx, Jy, Jz interaction terms are present.
    U_local = _local_evolution_xyz(
        Jx=Jx,
        Jy=Jy,
        Jz=Jz,
        hx_left=0.0,
        hy_left=0.0,
        hz_left=0.0,
        hx_right=0.0,
        hy_right=0.0,
        hz_right=0.0,
        coeff=coeff,
        dtype=dtype,
    )

    layer = GateLayer(layer_index=layer_idx, is_odd=is_odd_layer, n_sites=n_sites)
    
    for left in bonds:
        gate = Gate(
            matrix=U_local,
            qubits=(left, left + 1),
            layer_index=layer_idx,
            name=f"Exp(H_{parity})",
        )
        layer.add_gate(gate)
        
    return layer


def _single_layer_field(
    *,
    n_sites: int,
    field_name: str, # 'X', 'Y', or 'Z'
    h_strength: float,
    coeff: float,
    layer_idx: int,
    dtype: jnp.dtype,
) -> GateLayer:
    """
    Creates a layer of single-qubit rotation gates: exp(-i * coeff * h_strength * P).
    """
    # Matrix P
    P = _get_pauli(field_name, dtype)
    
    # Exponent: -i * coeff * h * P
    # Note: scalar multiplication
    exponent = -1j * coeff * h_strength * P
    U_single = jsp.expm(exponent)
    
    # We use is_odd=False arbitrarily for single qubit layers as they act on all sites
    layer = GateLayer(layer_index=layer_idx, is_odd=False, n_sites=n_sites)
    
    for q in range(n_sites):
        gate = Gate(
            matrix=U_single,
            qubits=(q,),
            layer_index=layer_idx,
            name=f"R{field_name}",
        )
        layer.add_gate(gate)
        
    return layer


def _single_layer_combined_field(
    *,
    n_sites: int,
    hx: float,
    hy: float,
    hz: float,
    coeff: float,
    sequence: Sequence[str], # e.g. ["X", "Y", "Z"]
    layer_idx: int,
    dtype: jnp.dtype,
) -> GateLayer:
    """
    Creates a single layer that applies the product of field rotations.
    If sequence is ["X", "Y", "Z"], the unitary is U_Z @ U_Y @ U_X,
    corresponding to circuit order: Layer(X) -> Layer(Y) -> Layer(Z).
    """
    U_combined = jnp.eye(2, dtype=dtype)
    
    # Iterate through sequence and multiply matrices.
    # If circuit is Gate(X) then Gate(Y), the effect on state is Y @ X @ state.
    # So we multiply new rotation on the left.
    for name in sequence:
        P = _get_pauli(name, dtype)
        strength = 0.0
        if name == 'X': strength = hx
        elif name == 'Y': strength = hy
        elif name == 'Z': strength = hz
        
        exponent = -1j * coeff * strength * P
        U_next = jsp.expm(exponent)
        
        # Accumulate: U_new = U_next @ U_current
        U_combined = U_next @ U_combined

    layer = GateLayer(layer_index=layer_idx, is_odd=False, n_sites=n_sites)
    
    name_str = "".join(sequence)
    for q in range(n_sites):
        gate = Gate(
            matrix=U_combined,
            qubits=(q,),
            layer_index=layer_idx,
            name=f"R{name_str}_coll",
        )
        layer.add_gate(gate)
        
    return layer


# -----------------------------------------------------------------------------
# Main Decomposition
# -----------------------------------------------------------------------------

def field_middle_trotter_layers(
    n_sites: int,
    Jx: float,
    Jy: float,
    Jz: float,
    *,
    hx: float = 0.0,
    hy: float = 0.0,
    hz: float = 0.0,
    order: int = 1,
    method: str = "yoshida",
    dt: float,
    reps: int,
    collapse: bool = False,
    dtype: jnp.dtype | None = None,
) -> List[GateLayer]:
    r"""
    Implements 1st, 2nd, and 4th order Trotter decompositions of the XYZ Hamiltonian
    with explicit splitting of Interaction and Field terms.
    
    The Hamiltonian is split as:
    H = H_even + H_X + H_Y + H_Z + H_odd
    
    Orders implemented: N
    - 1: E(dt) -> X(dt) -> Y(dt) -> Z(dt) -> O(dt)
    - 2: E(dt/2) -> X(dt/2) -> Y(dt/2) -> Z(dt/2) -> O(dt) -> Z(dt/2) -> Y(dt/2) -> X(dt/2) -> E(dt/2)
    - 4: Yoshida or Suzuki construction using the symmetric 2nd order stepper.
    
    For orders 2 and 4, boundary merging of the E layers is performed.
    If `collapse=True`, the consecutive single-qubit field layers (e.g. X, Y, Z)
    are merged into a single layer of gates.
    NOTE: HIGHLY INEFFICIENT FOR ISING MODEL (you are better off with different split)
    NOTE: the _single_layer_combined_field creates U_combined = U_Z @ U_Y @ U_X, but this introduces trotter ERROR between the field themselves. You can avoid this
    by exponentiating the sum of the matrices rather than multiplying the exponentials. 
    NOTE: REGARDING THE ORDER: we do Even Field Odd order. The sequence is a valid second order splitting that will guarantee the correct asymptotic behavior. 
            however, the physical error depends on the leading error coefficient, which is governed by nested commutators from the BCH expansion.
        For the sequence $e^{A/2} e^B e^{A/2}$, the leading error term is proportional to:$$-\frac{1}{24} [A, [A, B]] + \frac{1}{12} [B, [B, A]]$$Whether your sequence or the alternative produces a smaller absolute error depends entirely on the parameters of your system:If your fields are much stronger than your interactions ($h \gg J$), placing the fields on the outside often minimizes the prefactor.If your interactions are much stronger than your fields ($J \gg h$), placing the interactions on the outside (like you did) can be slightly more accurate.


    """
    if n_sites % 2:
        raise ValueError(f"n_sites must be even (got {n_sites})")
    if order not in (1, 2, 4):
        raise ValueError(f"order must be 1, 2, or 4 (got {order})")
    if method not in ("yoshida", "suzuki"):
        raise ValueError(f"method must be 'yoshida' or 'suzuki' (got {method})")
        
    # Determine efficient dtype if not provided
    if dtype is None:
        dtype = jnp.complex128

    layers: List[GateLayer] = []
    layer_idx = 0
    
    def add_interaction(parity, coeff):
        nonlocal layer_idx
        l = _single_layer_interaction(
            n_sites=n_sites, Jx=Jx, Jy=Jy, Jz=Jz, coeff=coeff, parity=parity, layer_idx=layer_idx, dtype=dtype
        )
        layers.append(l)
        layer_idx += 1
        
    def add_field(name, strength, coeff):
        nonlocal layer_idx
        l = _single_layer_field(
            n_sites=n_sites, field_name=name, h_strength=strength, coeff=coeff, layer_idx=layer_idx, dtype=dtype
        )
        layers.append(l)
        layer_idx += 1
        
    def add_collapsed_fields(sequence, coeff):
        nonlocal layer_idx
        l = _single_layer_combined_field(
            n_sites=n_sites, hx=hx, hy=hy, hz=hz, coeff=coeff, sequence=sequence, layer_idx=layer_idx, dtype=dtype
        )
        layers.append(l)
        layer_idx += 1

    # --- 1st Order ---
    if order == 1:
        for _ in range(reps):
            add_interaction("even", dt)
            if collapse:
                add_collapsed_fields(["X", "Y", "Z"], dt)
            else:
                add_field("X", hx, dt)
                add_field("Y", hy, dt)
                add_field("Z", hz, dt)
            add_interaction("odd", dt)
        return layers

    # --- 2nd Order ---
    if order == 2:
        dt_half = dt / 2.0
        
        # First E(dt/2)
        add_interaction("even", dt_half)
        
        for r in range(reps):
            # Inner Core
            if collapse:
                add_collapsed_fields(["X", "Y", "Z"], dt_half)
            else:
                add_field("X", hx, dt_half)
                add_field("Y", hy, dt_half)
                add_field("Z", hz, dt_half)
            
            add_interaction("odd", dt)
            
            if collapse:
                add_collapsed_fields(["Z", "Y", "X"], dt_half)
            else:
                add_field("Z", hz, dt_half)
                add_field("Y", hy, dt_half)
                add_field("X", hx, dt_half)
            
            if r < reps - 1:
                # Merge End E(dt/2) + Start E(dt/2) -> E(dt)
                add_interaction("even", dt)
            else:
                # Final E(dt/2)
                add_interaction("even", dt_half)

        if collapse:
            layers = absorb_boundary_1q_gates(layers)
            
        return layers

    # --- 4th Order ---
    if order == 4:
        steps = []
        if method == "yoshida":
            # Yoshida coefficients
            cbrt2 = 2.0 ** (1.0 / 3.0)
            x1 = 1.0 / (2.0 - cbrt2)
            x0 = 1.0 - 2.0 * x1  # negative
            for _ in range(reps):
                steps.extend([x1 * dt, x0 * dt, x1 * dt])
        elif method == "suzuki":
            # Suzuki constant p = 1 / (4 - 4^{1/3})
            cbrt4 = 4.0 ** (1.0 / 3.0)
            p = 1.0 / (4.0 - cbrt4)
            k1 = p * dt
            k2 = (1.0 - 4.0 * p) * dt
            for _ in range(reps):
                steps.extend([k1, k1, k2, k1, k1])
            
        num_substeps = len(steps)
        
        # Start with first E(t_0 / 2)
        t0 = steps[0]
        add_interaction("even", t0 / 2.0)
        
        for i, step_dt in enumerate(steps):
            t_half = step_dt / 2.0
            
            # Inner Core of S2(step_dt)
            if collapse:
                add_collapsed_fields(["X", "Y", "Z"], t_half)
            else:
                add_field("X", hx, t_half)
                add_field("Y", hy, t_half)
                add_field("Z", hz, t_half)
            
            add_interaction("odd", step_dt)
            
            if collapse:
                add_collapsed_fields(["Z", "Y", "X"], t_half)
            else:
                add_field("Z", hz, t_half)
                add_field("Y", hy, t_half)
                add_field("X", hx, t_half)
            
            if i < num_substeps - 1:
                # Merge current end E(t/2) with next start E(next_t/2)
                next_t_half = steps[i+1] / 2.0
                add_interaction("even", t_half + next_t_half)
            else:
                # Final trailing E
                add_interaction("even", t_half)

        if collapse:
            layers = absorb_boundary_1q_gates(layers)
            
        return layers

def absorb_boundary_1q_gates(layers: List[GateLayer]):
    # NOTE: this only works for order 2 and 4!
    if not layers:
        return layers

    n_sites = layers[0].n_sites
    # Boundary qubits are the first and last qubits
    boundary_qubits = [0, n_sites - 1]

    # Iterate through layers based on the pattern where redundancy occurs
    for idx in range(3, len(layers), 4):
        target_idx = idx - 2
        
        source_layer = layers[idx]
        target_layer = layers[target_idx]
        
        for q in boundary_qubits:
            # Find the gate acting on qubit q in the source layer (idx)
            source_gate = None
            for g in source_layer.gates:
                if g.qubits == (q,):
                    source_gate = g
                    break
            
            # Find the gate acting on qubit q in the target layer (idx-2)
            target_gate = None
            for g in target_layer.gates:
                if g.qubits == (q,):
                    target_gate = g
                    break
            
            if source_gate is not None and target_gate is not None:
                # Absorb source into target.
                # Since layer idx is applied AFTER layer idx-2,
                # New Matrix = Source Matrix @ Target Matrix
                new_matrix = source_gate.matrix @ target_gate.matrix
                target_gate.matrix = new_matrix
                
                # Remove the source gate from the source layer
                source_layer.gates.remove(source_gate)
                
    return layers
    
def field_middle_trotter_circuit(
    n_sites: int,
    Jx: float,
    Jy: float,
    Jz: float,
    *,
    hx: float = 0.0,
    hy: float = 0.0,
    hz: float = 0.0,
    order: int = 1,
    method: str = "yoshida",
    dt: float,
    reps: int,
    collapse: bool = True,
    dtype: jnp.dtype | None = None,
) -> Circuit:
    """
    Creates a Circuit object using `field_middle_trotter_layers`.
    """
    layers = field_middle_trotter_layers(
        n_sites=n_sites, Jx=Jx, Jy=Jy, Jz=Jz,
        hx=hx, hy=hy, hz=hz,
        order=order, method=method, dt=dt, reps=reps, collapse=collapse, dtype=dtype
    )

    return Circuit(
        n_sites=n_sites,
        layers=layers,
        hamiltonian_type="xyz_field_middle",
        trotter_params={"Jx": Jx, "Jy": Jy, "Jz": Jz, "hx": hx, "hy": hy, "hz": hz, "order": order, "method": method, "dt": dt, "reps": reps, "collapse": collapse}
    )

__all__ = [
    "field_middle_trotter_layers",
    "field_middle_trotter_circuit",
]
