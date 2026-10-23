import weyl_compression.jax_config

import jax.numpy as jnp
import numpy as np
from typing import Optional, Dict, List, Tuple
from weyl_compression.utils.utils import gate_map
from weyl_compression.circuit.circuit_dataclasses import Gate, GateLayer, Circuit
from weyl_compression.mpo.mpo_dataclass import MPO
from weyl_compression.mpo.mpo_builder import get_id_mpo
from weyl_compression.tensor_network.core_ops import contract_mpo_with_layer
from weyl_compression.utils.utils import compute_init_horizontal_dir 

# ---------------------------------------------------------------------
# 1.  SMALL, RE-USABLE HELPER
# ---------------------------------------------------------------------
def _update_left_env(
    L: jnp.ndarray,
    A1: jnp.ndarray,
    B1: jnp.ndarray,
    gate: Optional["Gate"] = None,
    A2: Optional[jnp.ndarray] = None,
    B2: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, int]:
    """
    Contract the current *left* environment with one site (or two if a
    nearest-neighbour gate starts here).

    Returns
    -------
    new_L : jnp.ndarray   Updated environment
    step  : int           How many physical sites were consumed (1 or 2)
    """
    if gate is None:                       # no gate
        new_L = jnp.einsum(
            "ai, abcd, icbj -> dj", L, A1, B1, optimize="optimal"
        )
        return new_L, 1

    if gate.is_single_qubit():             # 1-qubit gate
        g = gate.tensor
        new_L = jnp.einsum(
            "ai, abcd, ch, ihbj -> dj", L, A1, g, B1, optimize="optimal"
        )
        return new_L, 1

    if gate.is_two_qubit():                # 2-qubit gate
        if A2 is None or B2 is None:
            raise ValueError("Two-qubit gate requires A2/B2 tensors.")
        g = gate.tensor
        new_L = jnp.einsum(
            "ai,abcd,defg,cfhk,ihbj,jkel->gl",
            L, A1, A2, g, B1, B2,
            optimize="optimal",
        )
        return new_L, 2

    raise ValueError("Unsupported gate type.")

def _update_right_env(
    R: jnp.ndarray,
    A1: jnp.ndarray,
    B1: jnp.ndarray,
    gate: Optional["Gate"] = None,
    A0: Optional[jnp.ndarray] = None,
    B0: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, int]:
    """
    Contract the current *right* environment with one site (or two if a
    nearest-neighbour gate ends here).

    Returns
    -------
    new_R : jnp.ndarray   Updated environment
    step  : int           Negative number of sites to move left (-1 or -2)
    """
    if gate is None:                       # no gate
        new_R = jnp.einsum(
            "dj, abcd, icbj -> ai", R, A1, B1, optimize="optimal"
        )
        return new_R, -1

    if gate.is_single_qubit():             # 1-qubit gate
        g = gate.tensor
        new_R = jnp.einsum(
            "dj, abcd, ch, ihbj -> ai", R, A1, g, B1, optimize="optimal"
        )
        return new_R, -1

    if gate.is_two_qubit():                # 2-qubit gate
        if A0 is None or B0 is None:
            raise ValueError("Two-qubit gate requires A0/B0 tensors.")
        g = gate.tensor
        new_R = jnp.einsum(
            "gl,abcd,defg,cfhk,ihbj,jkel->ai",
            R, A0, A1, g, B0, B1,
            optimize="optimal",
        )
        return new_R, -2

    raise ValueError("Unsupported gate type.")

def compute_layer_boundary_environments(
    E_top: MPO,
    E_bottom: MPO,
    layer: GateLayer,
    side: str,
) -> List[jnp.ndarray]:
    """
    Computes all left or right boundary environment tensors for a given layer.

    These represent the contraction of everything to the left/right of each
    potential gate starting position within the layer, sandwiched between
    the top (E_top) and bottom (E_bottom) environment MPOs.

    Args:
        E_top: MPO representing the environment above the layer.
        E_bottom: MPO representing the environment below the layer.
        layer: The GateLayer object containing the gates.
        side: Which boundary environments to compute ('left' or 'right').

    Returns:
        A list of boundary environment tensors (or None). 
        The list will have length n_sites.
        If side='left', boundary_envs[i] is the environment to the LEFT of
        the gate STARTING at site i (or None if no gate starts at i).
        If side='right', boundary_envs[i] is the environment to the RIGHT of
        the gate ENDING at site i (or None if no gate ends at i).

    Raises:
        ValueError: If side is not 'left' or 'right', or if MPO sizes mismatch.
    """
    n_sites = E_top.n_sites
    if n_sites != E_bottom.n_sites:
        raise ValueError(f"E_top ({n_sites} sites) and E_bottom ({E_bottom.n_sites} sites) must have the same length.")
    if side not in ['left', 'right']:
        raise ValueError("side must be 'left' or 'right'")

    dtype = E_top.tensors[0].dtype 
    # --- Prepare Gate Maps ---
    gate_map_left, gate_map_right = gate_map(layer, n_sites)

    # --- Compute Environments ---
    boundary_envs = [None] * (n_sites) # Env i is boundary to the left/right of site i

    if side == 'left':
        # --- Left Sweep ---
        # Initial L: contract virtual bonds of E_top[0] and E_bottom[0]
        # Assumes dummy bonds are dim 1. Needs shape (1, 1) or compatible.
        L = jnp.eye(1, dtype=dtype) # Assuming E_top/E_bottom start with bond dim 1
        if E_top[0].shape[0] != 1 or E_bottom[0].shape[0] != 1:
             print(f"Warning: Left boundary MPOs shapes ({E_top[0].shape}, {E_bottom[0].shape}) "
                   f"do not start with bond dim 1. Initial environment might be incorrect.")
             # Adjust L initialization if boundary bond dims are different
        
        # cache left env only if a gate starts at the site
        if gate_map_left.get(0) is not None:
                    boundary_envs[0] = L # Environment left of site 0

        i = 0 # sweeping index
        # stop left sweep after we computed the left environment of the rightmost gate in the layer
        stop_index = max(s for s, g in gate_map_left.items() if g is not None)

        while i < stop_index:
            
            A1 = E_top[i]       # Top MPO at site i
            B1 = E_bottom[i]    # Bottom MPO at site i
            gate = gate_map_left.get(i) # Gate starting at site i
            
            # grab next-site MPOs only when required
            A2 = B2 = None
            if gate is not None and gate.is_two_qubit():
                A2, B2 = E_top[i + 1], E_bottom[i + 1]
            L, step = _update_left_env(L, A1, B1, gate, A2, B2)
            i += step

            if i < n_sites and gate_map_left.get(i) is not None:
                boundary_envs[i] = L

    elif side == 'right':
        # --- Right Sweep ---
        R = jnp.eye(1, dtype=dtype) # Assuming E_top/E_bottom end with bond dim 1
        if E_top[-1].shape[-1] != 1 or E_bottom[-1].shape[-1] != 1:
             print(f"Warning: Right boundary MPOs shapes ({E_top[-1].shape}, {E_bottom[-1].shape}) "
                   f"do not end with bond dim 1. Initial environment R might be incorrect.")
             # Re-initialize R if needed

        # Store initial environment if a gate ends at site n_sites-1, otherwise continue. 
        if gate_map_right.get(n_sites-1) is not None:
             boundary_envs[n_sites-1] = R

        # Determine the leftmost site a gate ends at
        start_index = min(s for s, g in gate_map_right.items() if g is not None)

        i = n_sites - 1
        while i > start_index : # Loop down to the first gate end or start of chain
            A1 = E_top[i]       # Top MPO at site i
            B1 = E_bottom[i]    # Bottom MPO at site i
            gate = gate_map_right.get(i) # Gate ending at site i

            A0 = B0 = None
            if gate is not None and gate.is_two_qubit() and i - 1 >= 0:
                A0, B0 = E_top[i - 1], E_bottom[i - 1]

            R, step = _update_right_env(R, A1, B1, gate, A0, B0)
            next_i = i + step        # step is negative


            # Store environment R if a gate ends at the *next* position (to the left)
            if next_i >= 0 and gate_map_right.get(next_i) is not None:
                boundary_envs[next_i] = R

            i = next_i # Move to the next position


    # --- Final Check ---
    if side == "left":
        n_gates = sum(g is not None for g in gate_map_left.values())
    else:                               # side == "right"
        n_gates = sum(g is not None for g in gate_map_right.values())

    n_envs = sum(env is not None for env in boundary_envs)

    if n_envs != n_gates:
        raise RuntimeError(
            f"compute_layer_boundary_environments: mismatch detected "
            f"({n_envs} boundary envs, {n_gates} gate(s) in layer)."
        )

    return boundary_envs



def compute_gate_environment_tensor(
    gate_qubits: Tuple[int, ...],
    E_top_layer: MPO,
    E_bottom_layer: MPO,
    E_left_boundary: jnp.ndarray,
    E_right_boundary: jnp.ndarray
) -> jnp.ndarray:
    """
    Computes the environment tensor for a specific gate location.

    Contracts the relevant site tensors from E_top and E_bottom with the
    provided left and right boundary tensors.

    Args:
        gate_qubits: The tuple of qubits the gate acts on (e.g., (i,) or (i, i+1)).
        E_top_layer: The top environment MPO for the current layer.
        E_bottom_layer: The bottom environment MPO for the current layer.
        E_left_boundary: The tensor representing the environment to the left.
        E_right_boundary: The tensor representing the environment to the right.

    Returns:
        The environment tensor, with indices corresponding to the gate's
        output and input physical legs. The index order
        should match the gate tensor convention (e.g., out1, out2, in1, in2).

    Notes on complex calculus conventions
    -------------------------------------
    Throughout this module we distinguish between:
    - Wirtinger derivative ∂f/∂G (treating G and G* as independent), and
    - The Riesz/Euclidean gradient Grad f under the complex inner product
      ⟨A,B⟩ = Tr(A^† B).

    For a holomorphic scalar f(G) = Tr(V^† U(G)), the horizontal sweep returns
    an "environment" tensor Env that equals the Wirtinger derivative ∂f/∂G.
    The scalar can be reconstructed via the complex inner product as
        f(G) = Tr((Grad f)^† G) = Tr((conj(∂f/∂G))^† G),
    which in our code appears as contractions of Env.conj() with the gate tensor.
    """
    n_sites = len(E_top_layer)
    if n_sites != len(E_bottom_layer):
        raise ValueError(f"E_top ({n_sites} sites) and E_bottom ({len(E_bottom_layer)} sites) must have the same length.")

    num_qubits = len(gate_qubits)
    # 1. Identify the site indices involved
    if num_qubits == 1:
        i = gate_qubits[0]
        # Check bounds
        if not (0 <= i < n_sites):
            raise ValueError(f"Single qubit index {i} out of bounds for MPO with {n_sites} sites.")

        # 2. Extract the relevant MPO tensors
        E_top_tensors = (E_top_layer[i],)
        E_bottom_tensors = (E_bottom_layer[i],)
        environment_tensor = jnp.einsum('ab, acde, bfcg, eg -> df', E_left_boundary, E_top_tensors[0], E_bottom_tensors[0], E_right_boundary)

    elif num_qubits == 2:
        # Assume adjacent qubits for standard environment calculation
        i = min(gate_qubits)
        ip1 = max(gate_qubits)
        if ip1 != i + 1:
            # This calculation typically assumes adjacent sites for the direct contraction.
            # Non-adjacent gates require a different, more complex contraction scheme
            # which is not implemented here.
            raise ValueError(f"Environment calculation currently assumes adjacent qubits for 2Q gates. Got {gate_qubits}.")
        # Check bounds
        if not (0 <= i < n_sites and 0 <= ip1 < n_sites):
             raise ValueError(f"Two qubit indices {(i, ip1)} out of bounds for MPO with {n_sites} sites.")

        # 2. Extract the relevant MPO tensors
        E_top_tensors = (E_top_layer[i], E_top_layer[ip1])
        E_bottom_tensors = (E_bottom_layer[i], E_bottom_layer[ip1])

        environment_tensor = jnp.einsum('ab, acde, efgh, bick, kjfl, hl -> dgij', E_left_boundary, E_top_tensors[0], E_top_tensors[1], E_bottom_tensors[0], E_bottom_tensors[1], E_right_boundary)
    else:
        raise ValueError(f"Unsupported number of gate qubits: {num_qubits}. Expected 1 or 2.")
    
    return environment_tensor


def compute_trace(Environment: jnp.ndarray, gate_tensor: jnp.ndarray):
    # The environment returned by the sweep equals the Wirtinger derivative
    #   Env = ∂/∂G Tr(U_ref^† U_circ(G)).
    # To reconstruct the scalar via the complex inner product ⟨A,B⟩=Tr(A^†B),
    # we use the Riesz gradient Grad f = conj(Env), hence:
    #   Tr(U_ref^† U_circ) = Tr((Grad f)^† G) = Tr((conj(Env))^† G)
    # which is implemented below via einsum of Environment.conj() with G.
    if gate_tensor.shape == (2,2,2,2):
        trace = jnp.einsum('ijdg, ijdg ->', Environment.conj(), gate_tensor)
    elif gate_tensor.shape == (2,2):
        trace = jnp.einsum('ij, ij ->', Environment.conj(), gate_tensor)

    else:
        raise ValueError(f"Unsupported gate_tensor shape: {gate_tensor.shape}. Expected (2,2) or (2,2,2,2).")
    return trace

def compute_upper_lower_environments(
        mpo_ref: MPO, 
        circuit: Circuit, 
        direction: str, 
        init_direction: str, 
        max_bondim_env: int,
        svd_cutoff: float = 1e-12
    ) -> Dict[int, MPO]:
    """
    Compute and cache the upper or lower MPO environments for a given circuit.

    This routine contracts a reference MPO (`mpo_ref`) through the layers of
    `circuit` in the specified `direction`, truncating at each step to at most
    `max_bondim_env` bond dimension. The result is a dictionary mapping each
    layer index to the environment MPO situated above (for `direction='top'`) or
    below (for `direction='bottom'`) that layer.

    Parameters
    ----------
    mpo_ref : MPO
        The initial MPO at the boundary of the circuit (top or bottom).
    circuit : Circuit
        The circuit whose layered structure defines the contraction path.
    direction : {'top', 'bottom'}
        Which environment to build:
        - `'top'`: start from the top boundary and move downward through layers.
        - `'bottom'`: start from the bottom boundary and move upward.
    init_direction : {'left_to_right', 'right_to_left'}
        The direction to contract MPOs within each layer on the first step;
        alternates on each subsequent layer.
    max_bondim_env : int
        Maximum bond dimension to retain when truncating the environment MPO
        after each layer contraction.

    Returns
    -------
    Dict[int, MPO]
        A mapping from layer index `ℓ` to the MPO representing the environment
        just above (for `'top'`) or just below (for `'bottom'`) layer ℓ. For
        `direction='top'`, keys run from 0 up to `circuit.num_layers-1`.

    Raises
    ------
    ValueError
        If `direction` is not one of `'top'` or `'bottom'`.
    """

    if direction not in ('top', 'bottom'):
        raise ValueError(f"direction must be 'top' or 'bottom', got '{direction}'")

    if direction == 'top':
        E_top_current = mpo_ref
        all_E_top: Dict[int, MPO] = {circuit.num_layers-1: E_top_current} # environment above the last layer

        sweep_direction = init_direction #'left_to_right' # Or 'right_to_left', choose convention. depends on canonical state of ref mpo
        for l in range(circuit.num_layers - 2, -1, -1): # L-2 down to 0
            layer_above = circuit.layers[l+1] # The layer just processed

            E_top_current = contract_mpo_with_layer(
                E_top_current,
                layer_above,
                layer_is_below=True,          # contracting a layer *below* E_top
                direction=sweep_direction,
                max_bondim=max_bondim_env,
                svd_cutoff=svd_cutoff
            )

            # flip for the next layer
            sweep_direction = "right_to_left" if sweep_direction == "left_to_right" else "left_to_right"

            # Cache environment that now sits *above* layer l
            all_E_top[l] = E_top_current  

        return all_E_top   

    if direction == 'bottom':
        E_bot_current = mpo_ref # identity MPO below the first layer
        all_E_bot: Dict[int, MPO] = {0: E_bot_current}

        sweep_direction = init_direction #'left_to_right' # Or 'right_to_left', choose convention. depends on canonical state of ref mpo
        for l in range(1, circuit.num_layers): # 1 to L-1
            layer_below = circuit.layers[l-1] # The layer just processed


            E_bot_current = contract_mpo_with_layer(
                E_bot_current,
                layer_below,
                layer_is_below=False,          # contracting a layer *above* E_bot
                direction=sweep_direction,
                max_bondim=max_bondim_env,
                svd_cutoff=svd_cutoff
            )

            # flip for the next layer
            sweep_direction = "right_to_left" if sweep_direction == "left_to_right" else "left_to_right"

            # Cache environment that now sits *above* layer l
            all_E_bot[l] = E_bot_current  


        return all_E_bot

def compute_top_envs(
    circuit: Circuit,
    mpo_ref_top: MPO,
    *,
    max_bondim_env: int,
    svd_cutoff: float,
    init_sweep_dir: str,
) -> Dict[int, MPO]:
    """
    Wrapper
    Precompute only the *top* environments E_top[l] (sitting above layer l), no bottoms.
    """
    all_E_top = compute_upper_lower_environments(
        mpo_ref=mpo_ref_top, circuit=circuit, direction='top',
        init_direction=init_sweep_dir,
        max_bondim_env=max_bondim_env, svd_cutoff=svd_cutoff
    )
    return all_E_top

def compute_bottom_envs(
    circuit: Circuit,
    *,
    max_bondim_env: int,
    svd_cutoff: float,
    init_sweep_dir: str,
) -> Dict[int, MPO]:
    """
    Pre-compute only the *bottom* environments E_bot[l] (sitting below layer l),
    analogous to `compute_top_envs`.
    """
    all_E_bot = compute_upper_lower_environments(
        mpo_ref=get_id_mpo(nsites=circuit.n_sites, dtype=circuit.dtype),
        circuit=circuit,
        direction="bottom",
        init_direction=init_sweep_dir,
        max_bondim_env=max_bondim_env,
        svd_cutoff=svd_cutoff,
    )
    return all_E_bot


def sweeping_euclidean_gradient_bottom_up(
    circuit: Circuit,
    mpo_ref: MPO,                     # usually V^† or the “top” MPO you already use
    *,
    max_bondim_env: int,
    svd_cutoff: float = 1e-12,
) -> Tuple[jnp.ndarray, List[jnp.ndarray], Dict[str, int]]:
    """
    Computes the Euclidean gradient of the loss function Tr(V_ref^adj U_circuit)
    with respect to all gates in the circuit using a structured sweeping approach.

    This function performs a single bottom-to-top pass over the circuit layers.
    For each layer, it sweeps horizontally (left-to-right or right-to-left) to
    compute gradients for all gates within that layer *without* updating the gates.
    The environment "below" (E_bottom) is updated after processing each layer.

    The loss value returned is a proxy based on intermediate contractions and
    might not represent the full final loss. The primary output is the list of gradients.
    Strategy:
      1. Precompute E_top[l] environments for all layers using a top-down pass.
         The sweep direction for this is determined by the canonical form of `mpo_ref_adj`.
      2. Initialize E_bottom to the identity MPO.
      3. Loop through layers from bottom (l=0) to top (l=num_layers-1):
         a. Determine the horizontal sweep direction for the current layer `l`.
            This alternates starting from an initial direction computed based on
            `mpo_ref_adj`'s canonical form and the number of layers.
         b. Perform a horizontal sweep (left-to-right or right-to-left) on the layer:
             - Compute the necessary single-site boundary environment for the sweep direction.
             - Iterate through gates in the sweep order, accumulating gradients.
             - Update the running E_bottom MPO by contracting it with the current layer's gates.
      4. Assemble the computed gradients into a canonical order (layer-major, gate-left-to-right).

    Args:
        circuit (Circuit): The quantum circuit for which to compute gradients.
        mpo_ref (MPO): The reference MPO (e.g., the target V_target).
        max_bondim_env (int): The maximum bond dimension for environment MPO truncation.
        svd_cutoff (float, optional): The cutoff for SVD truncation when compressing
                                      environment MPOs. Defaults to 1e-12.

    Returns:
        Tuple[jnp.ndarray, List[jnp.ndarray], Dict[str, int]]:
            - loss_scalar (jnp.ndarray): A scalar loss value (Note: likely a proxy, check implementation).
            - grads_ordered (List[jnp.ndarray]): A list of gradient arrays for gates,
              ordered canonically (layer 0 gates left-to-right, then layer 1 gates, etc.).
              Each gradient array corresponds to a gate's parameters.
            - info (Dict[str, int]): A dictionary containing metadata, e.g.,
            """
    # TODO: make sure that you are NOT storing unnecessary envs.
    # i.e. only store Env_bottom_current and Env_bottom_current-1
            # if current_dir = left to right, only store Env_left_current and Env_left_current-1
    
    # I. Compute all top environments + loss
    ## init
    dtype = mpo_ref[0].dtype
    n_sites = mpo_ref.n_sites
    mpo_ref_adj = mpo_ref.dagger()
    mpo_identity = get_id_mpo(circuit.n_sites, dtype=dtype)
    E_bottom_cur = mpo_identity

    
    all_E_top = compute_top_envs(
        circuit, mpo_ref_adj,
        max_bondim_env=max_bondim_env, svd_cutoff=svd_cutoff,
        init_sweep_dir=('left_to_right' if mpo_ref_adj.is_right_canonical else 'right_to_left')
    )

    all_grads_map: Dict[Gate, jnp.ndarray] = {}

    init_horizontal_dir = compute_init_horizontal_dir(mpo_ref_adj, circuit.num_layers)
    
    # II. Layer loop: bottom-up, alternate right/left pass.
    def dir_for_layer(l: int) -> str:
        if init_horizontal_dir not in ('left_to_right', 'right_to_left'):
            raise ValueError("init_horizontal_dir must be 'left_to_right' or 'right_to_left'")
        if l % 2 == 0:
            return init_horizontal_dir
        return 'right_to_left' if init_horizontal_dir == 'left_to_right' else 'left_to_right'
    
    for l in range(circuit.num_layers):
        layer: GateLayer = circuit.layers[l]
        E_top_l    = all_E_top[l]
        sweep_dir  = dir_for_layer(l)

            # TODO: this can easily be another function, since it is repeated in the analogous top-down function
            #       will call this: sweeping_euclidean_gradient_left_to_right is a wrapper to sweep_left_to_right
        if sweep_dir == 'left_to_right':
            layer_grads_map = _layer_pass_left_to_right(layer, E_top_l, E_bottom_cur, n_sites, dtype)
        elif sweep_dir == 'right_to_left':
            layer_grads_map = _layer_pass_right_to_left(layer, E_top_l, E_bottom_cur, n_sites, dtype)
        
        # aggregate the layer's gradients into the master map.
        all_grads_map.update(layer_grads_map)

    # III. Contract MPO with layer: update E_bottom

        # After finishing layer l: update bottom env ONCE and store it
        E_bottom_cur = contract_mpo_with_layer(
            E_bottom_cur, layer, layer_is_below=False,
            direction=sweep_dir, max_bondim=max_bondim_env, svd_cutoff=svd_cutoff
        )

    # IV. Compute loss 
    L_boundary = jnp.eye(1, dtype=dtype)  # Start with a 1x1 identity
    for i in range(n_sites):
        top_tensor = mpo_ref_adj[i]
        bottom_tensor = E_bottom_cur[i]
        # The einsum is the same as _update_left_env for a site with no gate
        L_boundary = jnp.einsum("ai, abcd, icbj -> dj", L_boundary, top_tensor, bottom_tensor, optimize="optimal")
    trace = L_boundary.squeeze()


    # Assemble grads in a STABLE canonical order (layer-major, left→right)
    grads_ordered: List[jnp.ndarray] = []
    for layer in circuit.layers:
        for gate in layer.iterate_gates(reverse=False):
            if gate in all_grads_map:
                grads_ordered.append(all_grads_map[gate])
            else: 
                raise ValueError(f"No gradient found for gate: {gate}. "
                                f"Check if it was excluded or if gradient computation failed.")

    info: Dict[str, int] = {
        "num_gates": len(grads_ordered), # placeholder info
    }
    return trace, grads_ordered, info

def sweeping_euclidean_gradient_top_down(
    circuit: Circuit,
    mpo_ref: MPO,                       # usually V† placed on the *top* boundary
    *,
    max_bondim_env: int,
    svd_cutoff: float = 1e-12,
) -> Tuple[jnp.ndarray, List[jnp.ndarray], Dict[str, int]]:
    """
    Top-to-bottom analogue of `sweeping_euclidean_gradient_bottom_up`.

    Strategy
    --------
    1.  Pre-compute **bottom** environments E_bot[l] (below each layer) using an
        identity MPO at the physical bottom of the circuit.
    2.  Initialise a running *top* environment `E_top_cur` to `mpo_ref†`
        (the boundary MPO sitting *above* the highest layer).
    3.  Sweep layers from **top (l = L-1) down to 0**, alternating horizontal
        directions starting in the direction opposite to the canonical direction of mpo_ref†
    4.  After computing a layer’s gate-wise environments, absorb the layer into
        `E_top_cur` so that it becomes the new environment for the layer below.
    5.  Assemble gradients in canonical order (layer-major, left→right) and
        return them together with a (placeholder) loss proxy and metadata.
    """
    # I.  Book-keeping and pre-computation
    dtype      = mpo_ref[0].dtype
    n_sites    = mpo_ref.n_sites
    mpo_ref_adj = mpo_ref.dagger()

    # Bottom environments start from the identity MPO
    all_E_bottom = compute_bottom_envs(
        circuit,
        max_bondim_env=max_bondim_env,
        svd_cutoff=svd_cutoff,
        init_sweep_dir="left_to_right",
    )
    # TODO: use init_horizontal_dir from utils

    # Running top environment (above current layer)
    E_top_cur = mpo_ref_adj

    # Horizontal sweep convention
    init_horizontal_dir = 'left_to_right' if mpo_ref_adj.is_right_canonical else 'right_to_left'

    def dir_for_layer(l: int) -> str:
        """Return L→R / R→L according to layer index parity."""
        if l % 2 == 0:
            return init_horizontal_dir
        return "right_to_left" if init_horizontal_dir == "left_to_right" else "left_to_right"

    all_grads_map: Dict[Gate, jnp.ndarray] = {}

    # II. Layer loop: TOP → BOTTOM
    for l in range(circuit.num_layers - 1, -1, -1):
        layer            = circuit.layers[l]
        E_bottom_l       = all_E_bottom[l]        # environment *below* layer l
        sweep_dir        = dir_for_layer(l)

        if sweep_dir == "left_to_right":
            layer_grads  = _layer_pass_left_to_right(
                layer, E_top_cur, E_bottom_l, n_sites, dtype
            )
        else:
            layer_grads  = _layer_pass_right_to_left(
                layer, E_top_cur, E_bottom_l, n_sites, dtype
            )

        all_grads_map.update(layer_grads)

        # III. Update the running *top* environment by absorbing the layer
        E_top_cur = contract_mpo_with_layer(
            E_top_cur,
            layer,
            layer_is_below=True,        # layer sits *below* E_top_cur
            direction=sweep_dir,
            max_bondim=max_bondim_env,
            svd_cutoff=svd_cutoff,
        )
    
    identity_mpo = get_id_mpo(n_sites, dtype=dtype)
    L_boundary = jnp.eye(1, dtype=dtype)  # Start with a 1x1 identity
    for i in range(n_sites):
        top_tensor = E_top_cur[i]
        bottom_tensor = identity_mpo[i]
        # The einsum is the same as _update_left_env for a site with no gate
        L_boundary = jnp.einsum("ai, abcd, icbj -> dj", L_boundary, top_tensor, bottom_tensor, optimize="optimal")
    trace = L_boundary.squeeze()

    # NOTE: consider also other ways of storing the euclidean gradients, this is ambiguous.
    # IV. Collect gradients in canonical (layer-major) order
    grads_ordered: List[jnp.ndarray] = []
    for layer in circuit.layers:
        for gate in layer.iterate_gates(reverse=False):
            try:
                grads_ordered.append(all_grads_map[gate])
            except KeyError as err:
                raise ValueError(
                    f"Gradient missing for gate {gate} – "
                    "check environment construction."
                ) from err

    info = {"num_gates": len(grads_ordered)}
    return trace, grads_ordered, info

def _layer_pass_left_to_right(    
    current_layer: GateLayer,        # <-- mutated in-place
    E_top_l: MPO,
    E_bottom_current: MPO,
    n_sites: int,
    dtype: jnp.dtype,
) -> Dict[Gate, jnp.ndarray]: # what should I return? a dict of some sort (key: gate_id, value: Env):
    """
    Sweeps left-to-right across a layer, computing the gradient for each gate.
    
    Returns:
        A dictionary mapping each Gate object in the layer to its
        gradient tensor (the environment tensor, Env).
    """
    layer_gradients: Dict[Gate, jnp.ndarray] = {}


    E_right_boundaries = compute_layer_boundary_environments(
        E_top_l, E_bottom_current, current_layer, side="right"
    )
    E_left_current = jnp.eye(1, dtype=dtype)

    current_site = 0 # cursor tracking left env position

    for gate in current_layer.iterate_gates(reverse=False):

        # Filling the "gap" between the current site and the next gate by propagating the environment with identity operators.
        gap_start_site = current_site
        gap_end_site = gate.qubits[0]
        for site_i in range(gap_start_site, gap_end_site):
            E_left_current, _ = _update_left_env(
                E_left_current,
                E_top_l[site_i], E_bottom_current[site_i],
            )
        current_site = gap_end_site # Move cursor to the start of the gate

        # gate is present. Gather its right environment and build Env
        rightmost_qb = max(gate.qubits)
        E_right_current = E_right_boundaries[rightmost_qb]

        Env = compute_gate_environment_tensor(
            gate.qubits,
            E_top_l, E_bottom_current,
            E_left_current, E_right_current
        )

        # reshape in matrix form to serve the optimizer well
        env_ndim     = Env.ndim
        out_shape    = Env.shape[env_ndim // 2 :]
        in_shape     = Env.shape[: env_ndim // 2]
        Env   = Env.reshape(np.prod(out_shape), np.prod(in_shape))

        layer_gradients[gate] = Env

        # update left_env
        if gate.is_two_qubit():
            E_left_current, step = _update_left_env(
                E_left_current,
                E_top_l[current_site], E_bottom_current[current_site],
                gate,
                E_top_l[rightmost_qb], E_bottom_current[rightmost_qb]
            )
        else: # single-qubit gate
            E_left_current, step = _update_left_env(
                E_left_current,
                E_top_l[current_site], E_bottom_current[current_site],
                gate
            )
        
        # advance the current cursor past the processed gate:
        current_site += step

    return layer_gradients

def _layer_pass_right_to_left(    
    current_layer: GateLayer,        # <-- mutated in-place
    E_top_l: MPO,
    E_bottom_current: MPO,
    n_sites: int,
    dtype: jnp.dtype,
) -> Dict[Gate, jnp.ndarray]:
    """
    Sweeps right-to-left across a layer, computing the gradient for each gate.

    Returns:
        A dictionary mapping each Gate object in the layer to its
        gradient tensor (the environment tensor, Env).
    """
    layer_gradients: Dict[Gate, jnp.ndarray] = {}


    # For an R->L sweep we need, for each gate, the *left* boundary env
    # at its starting site, and we propagate a running *right* env.
    E_left_boundaries = compute_layer_boundary_environments(
        E_top_l, E_bottom_current, current_layer, side="left"
    )
    E_right_current = jnp.eye(1, dtype=dtype)

    current_site = n_sites - 1  # cursor tracking right env position

    for gate in current_layer.iterate_gates(reverse=True):
        leftmost_qb = min(gate.qubits)
        rightmost_qb = max(gate.qubits)

        # Fill the "gap" between the current site and the next gate (to the left)
        # by propagating the environment with identity operators.
        gap_start_site = current_site
        gap_end_site = rightmost_qb
        for site_i in range(gap_start_site, gap_end_site, -1):
            # propagate one site to the left (no gate)
            E_right_current, _ = _update_right_env(
                E_right_current,
                E_top_l[site_i], E_bottom_current[site_i],
            )
        current_site = gap_end_site  # Move cursor to the *end* of the gate

        # gate is present. Gather its left environment and build Env
        E_left_current = E_left_boundaries[leftmost_qb]

        Env = compute_gate_environment_tensor(
            gate.qubits,
            E_top_l, E_bottom_current,
            E_left_current, E_right_current
        )
        env_ndim     = Env.ndim
        out_shape    = Env.shape[env_ndim // 2 :]
        in_shape     = Env.shape[: env_ndim // 2]
        Env   = Env.reshape(np.prod(out_shape), np.prod(in_shape))

        layer_gradients[gate] = Env

        # Update the running *right* environment by absorbing the gate region
        if gate.is_two_qubit():
            # For _update_right_env, A1/B1 correspond to site i=current_site (rightmost),
            # and A0/B0 correspond to the left neighbor (leftmost_qb).
            E_right_current, step = _update_right_env(
                E_right_current,
                E_top_l[current_site], E_bottom_current[current_site],
                gate,
                E_top_l[leftmost_qb], E_bottom_current[leftmost_qb]
            )
        else:  # single-qubit gate
            E_right_current, step = _update_right_env(
                E_right_current,
                E_top_l[current_site], E_bottom_current[current_site],
                gate
            )

        # step is negative (-1 or -2). Move cursor left past the processed gate.
        current_site += step

    return layer_gradients

def overlap_and_environments(
    circuit: Circuit,
    mpo_ref: MPO,                     
    *,
    max_bondim_env: int,
    svd_cutoff: float = 1e-12,
    vertical_sweep: str,
):
    # depending on vertical direction:
    if vertical_sweep == 'bottom-up':
        # does a bottom-up sweep
        loss, grad_e, info = sweeping_euclidean_gradient_bottom_up(circuit, mpo_ref=mpo_ref, 
                                                             max_bondim_env=max_bondim_env, 
                                                             svd_cutoff=svd_cutoff, 
                                                            )
    else:
        # does a top-down sweep
        loss, grad_e, info = sweeping_euclidean_gradient_top_down(circuit, mpo_ref=mpo_ref,
                                                                  max_bondim_env=max_bondim_env, 
                                                                  svd_cutoff=svd_cutoff, 
                                                                  ) 
    return loss, grad_e, info
