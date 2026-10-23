import weyl_compression.jax_config

from weyl_compression.circuit.circuit_dataclasses import Gate, GateLayer
from weyl_compression.mpo.mpo_dataclass import MPO
from typing import Dict, Optional


def gate_map(layer: GateLayer, n_sites: int):
    """
    Generate mappings from qubit site indices to gates for a given GateLayer.

    This function scans through all gates in the provided layer and constructs two
    dictionaries:
    1. gate_map_left: Maps each starting site index to the Gate that begins there.
    2. gate_map_right: Maps each ending site index to the Gate that ends there.

    If multiple gates share the same starting or ending site, a warning is printed,
    and the last encountered gate at that site will be used in the mapping.

    Parameters:
        layer (GateLayer): The layer of gates containing Gate objects with qubit indices.
        n_sites (int): The total number of qubit sites to consider (indices 0 to n_sites-1).

    Returns:
        Tuple[Dict[int, Optional[Gate]], Dict[int, Optional[Gate]]]:
            - gate_map_left: A dictionary mapping starting site indices to their corresponding Gate or None.
            - gate_map_right: A dictionary mapping ending site indices to their corresponding Gate or None.
    """
    # Map starting site to gate
    gate_map_left: Dict[int, Optional[Gate]] = {i: None for i in range(n_sites)}
    # Map ending site to gate
    gate_map_right: Dict[int, Optional[Gate]] = {i: None for i in range(n_sites)}

    for gate in layer.gates:
        leftmost_site = min(gate.qubits)
        rightmost_site = max(gate.qubits)
        if leftmost_site < n_sites:
            if gate_map_left[leftmost_site] is not None:
                 print(f"Warning: Multiple gates starting at site {leftmost_site}. Using last found: {gate.name}.")
            gate_map_left[leftmost_site] = gate
        if rightmost_site < n_sites:
             if gate_map_right[rightmost_site] is not None:
                 print(f"Warning: Multiple gates ending at site {rightmost_site}. Using last found: {gate.name}.")
             gate_map_right[rightmost_site] = gate
    return gate_map_left, gate_map_right


def compute_init_horizontal_dir(mpo_ref_adj: MPO, num_layers: int) -> str:
    """
    Computes the initial horizontal sweep direction for DMRG based on
    the reference MPO's canonicalization and the number of layers.

    Args:
        mpo_ref_adj (MPO): The reference MPO, expected to be canonicalized.
        num_layers (int): The number of layers in the MPO optimization.

    Returns:
        str: The initial sweep direction, either "left_to_right" or "right_to_left".

    Raises:
        ValueError: If mpo_ref_adj is not in a definite canonical form or
                    if num_layers is not a positive integer.
    """
    if not isinstance(num_layers, int) or num_layers <= 0:
        raise ValueError("num_layers must be a positive integer.")

    # Determine the reference MPO's canonical direction
    if mpo_ref_adj.is_left_canonical and not mpo_ref_adj.is_right_canonical:
        ref_direction = "left_to_right"
    elif mpo_ref_adj.is_right_canonical and not mpo_ref_adj.is_left_canonical:
        ref_direction = "right_to_left"
    else:
        # Covers cases where both are True (scalar MPO, unlikely here),
        # both are False (not canonicalized), or attributes are in an unexpected state.
        raise ValueError(
            "mpo_ref_adj must be in a definite left- or right-canonical form. "
            f"Got is_left_canonical={mpo_ref_adj.is_left_canonical}, "
            f"is_right_canonical={mpo_ref_adj.is_right_canonical}."
        )
    
        # Determine the initial sweep direction based on parity of layers
    if num_layers % 2 == 1:  # Odd number of layers
        # Sweep direction needs to be opposite to the reference MPO's canonical direction
        if ref_direction == "left_to_right":
            init_horizontal_dir = "right_to_left"
        else:  # ref_direction == "right_to_left"
            init_horizontal_dir = "left_to_right"
    else:  # Even number of layers
        # Sweep direction is the same as the reference MPO's canonical direction
        init_horizontal_dir = ref_direction

    return init_horizontal_dir