import weyl_compression.jax_config  # ensures JAX defaults are consistent

from typing import List

import jax.numpy as jnp

from weyl_compression.circuit.circuit_dataclasses import GateLayer, Circuit
from weyl_compression.circuit.trotter.gates import even_odd_layer

def even_odd_trotter_layers(
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
    dtype: jnp.dtype | None = None,
) -> List[GateLayer]:
    """Return a list of :class:`GateLayer` objects implementing a Suzuki–Trotter
    approximation of *exp(-i·t·H)* for an **even-length** XYZ chain.

    Parameters
    ----------
    n_sites
        Total number of qubits (must be **even**).
    Jx / Jy / Jz
        Couplings in *H = Jx X₁X₂ + Jy Y₁Y₂ + Jz Z₁Z₂ + ...*.
    hx / hy / hz
        Fields in *H = ... + hx X + hy Y + hz Z*.
    order
        Trotter–Suzuki order (1, 2 or 4).
    method
        Method for order=4: "yoshida" (minimal 6·reps + 1 layers)
        or "suzuki" (fractal, 10·reps + 1 layers).
    dt
        Elementary time step **per bond layer**.
    reps
        How many times the elementary Trotter sequence is repeated.
    dtype
        Optional override for the JAX dtype of every gate matrix.

    Returns
    -------
    List[GateLayer]
        The brick-wall circuit layers in execution order.
    """
    # --- Validation ---------------------------------------------------------
    if order not in (1, 2, 4):
        raise ValueError(f"order must be 1, 2 or 4 (got {order})")
    if method not in ("yoshida", "suzuki"):
        raise ValueError(f"method must be 'yoshida' or 'suzuki' (got {method})")
    if n_sites % 2:
        raise ValueError(f"n_sites must be even (got {n_sites})")

    def _add_layer(parity: str, coeff: float, idx: int) -> GateLayer:
        return even_odd_layer(
            n_sites=n_sites,
            Jx=Jx, Jy=Jy, Jz=Jz,
            hx=hx, hy=hy, hz=hz,
            coeff=coeff,
            parity=parity,
            layer_idx=idx,
            dtype=dtype,
        )

    layers: List[GateLayer] = []
    layer_idx = 0

    # -----------------------------------------------------------------------
    # 1st-order Lie–Trotter  →  E(dt) O(dt)  repeated
    # -----------------------------------------------------------------------
    if order == 1:
        for _ in range(reps):
            for parity in ("even", "odd"):          # same coeff for both
                layers.append(_add_layer(parity, dt, layer_idx))
                layer_idx += 1
        return layers

    if order == 2: # 2*reps + 1 layers
        # --- leading E(dt/2) ----------------------------------------------
        layers.append(_add_layer("even", dt / 2, layer_idx)); layer_idx += 1

        # --- middle blocks  [O(dt) E(dt)]^{reps-1}  ------------------------
        for _ in range(reps - 1):
            layers.append(_add_layer("odd", dt, layer_idx));  layer_idx += 1
            layers.append(_add_layer("even", dt, layer_idx)); layer_idx += 1

        # --- trailing O(dt) E(dt/2) ---------------------------------------
        layers.append(_add_layer("odd",  dt,      layer_idx)); layer_idx += 1
        layers.append(_add_layer("even", dt / 2,  layer_idx)); layer_idx += 1

        return layers
    
    # ======================================================================
    # 4th-order  (Yoshida, minimal 6·reps + 1 layers)
    # ======================================================================
    if method == "yoshida":
        # Yoshida constant  s = 1 / (2 − 2^{1/3})
        cbrt2 = 2.0 ** (1.0 / 3.0)
        s = 1.0 / (2.0 - cbrt2)

        # time-step coefficients
        a = s * dt                  #  a  = s·dt        (positive)
        b = (1.0 - 2.0 * s) * dt    #  b  = (1−2s)·dt   (negative)
        e_half = a / 2.0            # leading / trailing  E(s·dt/2)
        e_mid  = (1.0 - s) * dt / 2 # central even layers E((1−s)·dt/2)
        e_full = a                  # inter-rep merged   E(s·dt)


        # ---- leading  E(s·dt/2) ---------------------------------------------
        layers.append(_add_layer("even", e_half, layer_idx)); layer_idx += 1

        # ---- repeat block  ---------------------------------------------------
        for rep in range(1, reps + 1):
            #  O(a)
            layers.append(_add_layer("odd",  a,      layer_idx)); layer_idx += 1
            #  E((1−s)·dt/2)
            layers.append(_add_layer("even", e_mid,  layer_idx)); layer_idx += 1
            #  O(b)
            layers.append(_add_layer("odd",  b,      layer_idx)); layer_idx += 1
            #  E((1−s)·dt/2)
            layers.append(_add_layer("even", e_mid,  layer_idx)); layer_idx += 1
            #  O(a)
            layers.append(_add_layer("odd",  a,      layer_idx)); layer_idx += 1

            #  Inter-rep merger  E(s·dt)  (skip after final repetition)
            if rep < reps:
                layers.append(_add_layer("even", e_full, layer_idx)); layer_idx += 1

        # ---- trailing  E(s·dt/2) --------------------------------------------
        layers.append(_add_layer("even", e_half, layer_idx)); layer_idx += 1

        return layers

    # ======================================================================
    # 4th-order  (Suzuki, 10·reps + 1 layers)
    # ======================================================================
    if method == "suzuki":
        # Suzuki constant p = 1 / (4 - 4^{1/3})
        cbrt4 = 4.0 ** (1.0 / 3.0)
        p = 1.0 / (4.0 - cbrt4)

        # time-step coefficients
        k1 = p * dt                 # positive
        k2 = (1.0 - 4.0 * p) * dt   # negative
        
        e_half = k1 / 2.0           # leading / trailing  E(p·dt/2)
        e_full = k1                 # inter-rep merged    E(p·dt)
        e_k1   = k1                 # E layer between O(k1) and O(k1)
        e_mid  = (1.0 - 3.0 * p) * dt / 2.0  # (k1 + k2)/2

        # ---- leading  E(k1/2) ---------------------------------------------
        layers.append(_add_layer("even", e_half, layer_idx)); layer_idx += 1

        # ---- repeat block  ---------------------------------------------------
        for rep in range(1, reps + 1):
            # S2(k1) S2(k1) S2(k2) S2(k1) S2(k1)
            # expands to: O(k1), E(k1), O(k1), E(e_mid), O(k2), E(e_mid), O(k1), E(k1), O(k1)
            layers.append(_add_layer("odd",  k1,     layer_idx)); layer_idx += 1
            layers.append(_add_layer("even", e_k1,   layer_idx)); layer_idx += 1
            layers.append(_add_layer("odd",  k1,     layer_idx)); layer_idx += 1
            layers.append(_add_layer("even", e_mid,  layer_idx)); layer_idx += 1
            layers.append(_add_layer("odd",  k2,     layer_idx)); layer_idx += 1
            layers.append(_add_layer("even", e_mid,  layer_idx)); layer_idx += 1
            layers.append(_add_layer("odd",  k1,     layer_idx)); layer_idx += 1
            layers.append(_add_layer("even", e_k1,   layer_idx)); layer_idx += 1
            layers.append(_add_layer("odd",  k1,     layer_idx)); layer_idx += 1

            # Inter-rep merger E(p*dt)
            if rep < reps:
                layers.append(_add_layer("even", e_full, layer_idx)); layer_idx += 1

        # ---- trailing  E(k1/2) --------------------------------------------
        layers.append(_add_layer("even", e_half, layer_idx)); layer_idx += 1

        return layers


def even_odd_trotter_circuit(
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
    dtype: jnp.dtype | None = None,
):
    layers = even_odd_trotter_layers(
        n_sites=n_sites, Jx=Jx, Jy=Jy, Jz=Jz,
        hx=hx, hy=hy, hz=hz,
        order=order, method=method, dt=dt, reps=reps,
        dtype=dtype
    )
    circ = Circuit(
        n_sites=n_sites,
        layers=layers,
    )
    return circ
__all__ = [
    "even_odd_trotter_layers",
    "even_odd_trotter_circuit",
]
