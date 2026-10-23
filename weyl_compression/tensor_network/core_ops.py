import weyl_compression.jax_config

import jax.numpy as jnp
from weyl_compression.circuit.circuit_dataclasses import GateLayer
from weyl_compression.mpo.mpo_dataclass import MPO
from weyl_compression.utils.utils import gate_map
from typing import Tuple, Optional, List

def canonicalize_local_tensor(
    tensor: jnp.ndarray,
    mode: str
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Canonicalizes a single MPO tensor using QR or RQ decomposition.

    Assumes the input tensor has shape (left_bond, phys_out, phys_in, right_bond).

    - If mode is 'left', performs QR decomposition (T = QR). The resulting Q
      tensor is left-canonical. Returns Q (reshaped) and R. R should be
      absorbed into the tensor to the right.
    - If mode is 'right', performs RQ decomposition (T = RQ). The resulting Q
      tensor is right-canonical. Returns Q (reshaped) and R. R should be
      absorbed into the tensor to the left.

    Args:
        tensor: The 4D MPO tensor to canonicalize.
        mode: The canonicalization mode ('left' or 'right').

    Returns:
        A tuple containing:
        - canonical_tensor (jnp.ndarray): The orthogonal factor (Q), reshaped
            to have 4 indices, with one virtual bond dimension potentially changed.
        - factor_matrix (jnp.ndarray): The factor (R) to be absorbed into the
            neighboring tensor. Shape depends on the mode:
            - 'left' mode: R has shape (new_right_bond, old_right_bond)
            - 'right' mode: R has shape (old_left_bond, new_left_bond)

    Raises:
        ValueError: If the input tensor is not 4D or if the mode is invalid.

                                                                                                                     
          Example of 'right' canonicalization:                                                                   
                                                                                                                 
                                                                                                                 
                                                                                                                 
                                                     | phys_out                               | phys_out         
                                                     |                                        |                  
 left_bond           new_left_bond                   |                                        |                  
           +--------+              new_left_bond +--------+   right_bond        left_bond +--------+   right_bond
           |        |                            |        |                               |        |             
   --------|   R    |--------            --------|   Q    |--------   <------     --------|        |--------     
           |        |                            |        |                               |        |             
           +--------+                            +--------+                               +--------+             
                                                     |                                        |                  
                                                     | phys_in                                | phys_in          
                                                     |                                        |                  
    """
    if len(tensor.shape) != 4:
        raise ValueError(f"Input tensor must be 4D (l, po, pi, r), got shape {tensor.shape}")

    original_shape = tensor.shape
    l_dim, po_dim, pi_dim, r_dim = original_shape

    if mode == 'left':
        # --- Left Canonicalization (QR) ---
        # Reshape T into a matrix M: Combine (l, po, pi) dims for rows, keep r for columns.
        # Target shape for M: (l*po*pi, r)
        matrix_to_decompose = tensor.reshape((l_dim * po_dim * pi_dim, r_dim))

        # Perform QR decomposition: M = Q @ R
        # Q will have shape (l*po*pi, new_r), R will have shape (new_r, r)
        Q, R_factor = jnp.linalg.qr(matrix_to_decompose, mode='reduced')

        # Reshape Q back into a 4D tensor: (l, po, pi, new_r)
        new_r_dim = Q.shape[-1]
        canonical_tensor = Q.reshape((l_dim, po_dim, pi_dim, new_r_dim))

        # R_factor is the matrix to be absorbed by the right neighbor
        factor_matrix = R_factor

    elif mode == 'right':
        # --- Right Canonicalization (RQ) ---
        # Reshape T into a matrix M: Keep l for rows, combine (po, pi, r) for columns.
        # Target shape for M: (l, po*pi*r)
        matrix_to_decompose = tensor.reshape((l_dim, po_dim * pi_dim * r_dim))

        # Perform RQ decomposition via QR of transpose
        # M = R @ Q
        # M.T = Q.T @ R.T = Q_prime @ R_prime
        # So Q = Q_prime.T, R = R_prime.T
        matrix_T = matrix_to_decompose.T
        Q_prime, R_prime = jnp.linalg.qr(matrix_T, mode='reduced')
        
        Q = Q_prime.T
        R_factor = R_prime.T

        # Reshape Q back into a 4D tensor: (new_l, po, pi, r)
        new_l_dim = Q.shape[0]
        canonical_tensor = Q.reshape((new_l_dim, po_dim, pi_dim, r_dim))

        # R_factor is the matrix to be absorbed by the left neighbor
        factor_matrix = R_factor

    else:
        raise ValueError(f"Invalid mode '{mode}'. Choose 'left' or 'right'.")

    return canonical_tensor, factor_matrix

def merge_one_mpo_and_gate(
        mpo: jnp.ndarray,
        gate: jnp.ndarray,
        gate_is_below: bool = None
) -> jnp.ndarray:
    
    if gate_is_below:
        return jnp.einsum('iabk, bc -> iack', mpo, gate, optimize='optimal')
    elif not gate_is_below:
        return jnp.einsum('ab, ibck -> iack', gate, mpo, optimize='optimal')
    else:
        raise ValueError("You need to specify the relative position of the gate and the MPO.")
    
def merge_two_mpos_and_gate(
    mpo1: jnp.ndarray,
    mpo2: jnp.ndarray,
    gate: jnp.ndarray,
    gate_is_below: bool = True
) -> jnp.ndarray:
    """
    Merges two adjacent MPO tensors with a two-qubit gate tensor.

    Contracts the network formed by mpo1 -- mpo2 with the gate V acting
    across their physical indices. This is a fundamental step in contracting
    a circuit layer with an MPO representation (like U_ref or an environment).

    Args:
        mpo1: Left MPO tensor. Expected shape (l1, p1_out, p1_in, r1).
        mpo2: Right MPO tensor. Expected shape (r1, p2_out, p2_in, r2).
              The bond dimension r1 must match mpo1's right bond.
        gate: Two-qubit gate tensor. Expected shape (p1', p2', p1'', p2'').
              The specific physical dimension interpretation (which index is in/out)
              depends on the 'gate_is_below' flag and the context (see below).
              Typically assumes (p_out1, p_out2, p_in1, p_in2) or compatible. Out = upper index, In = lower index.
        gate_is_below: Controls the contraction pattern.
            - If True: Assumes the gate tensor `gate` logically applies *before*
              (or below in diagrams) the MPO tensors `mpo1` and `mpo2`.
              The gate's output legs connect to the `p_in` legs of the MPOs.
              Pattern typically used when building the E_top environment.
              Einsum path: 'iabc, cdef, begh -> iadghf'
              (where 'begh' represents the gate tensor).
              Result indices: (l1, p1_out, p2_out, gate_p1_in, gate_p2_in, r2)
            - If False: Assumes the gate tensor `gate` logically applies *after*
              (or above in diagrams) the MPO tensors `mpo1` and `mpo2`.
              The gate's input legs connect to the `p_out` legs of the MPOs.
              This pattern is typically used when building the E_bottom environment.
              Einsum path: 'abcd, icef, fdgh -> iabegh'
              (where 'abcd' represents the gate tensor).
              Result indices: (l1, gate_p1_out, gate_p2_out, p1_in, p2_in, r2)

    Returns:
        The resulting merged tensor with 6 indices. The specific index order
        and meaning depends on the 'gate_is_below' flag. This tensor typically
        needs to be split back into two MPO tensors (e.g., using SVD).

    Raises:
        ValueError: If input tensor dimensions or shapes are incompatible.

                                                            
      if gate_is_left == True (i.e. two-qubit gate is below)
                                                            
                                                                                      
          | p1_out          | p2_out                                                  
          |                 |                                                         
 l1     +----+  r1    r1  +----+     r2                       p1_out      p2_out      
   -----|mpo1|-----  -----|mpo2|-----                            |         |          
        +----+            +----+                                 |         |          
          |                 |                               +-------------------+     
          | p1_in           | p2_in                    l1   |                   |  r2 
                                       ---------->     -----|                   |-----
          | gp_out1         | gp_out2                       |                   |     
          |                 |                               +-------------------+     
        +----------------------+                                 |         |          
        |        gate          |                                 |         |          
        +----------------------+                              gp_in1      gp_in2      
          |                 |                                                         
          | gp_in1          | gp_in2                                                  
                                                                                      
    """
    # --- Input Validation ---
    if len(mpo1.shape) != 4 or len(mpo2.shape) != 4:
        raise ValueError(f"MPO tensors must be 4D. Got shapes {mpo1.shape} and {mpo2.shape}.")
    if mpo1.shape[-1] != mpo2.shape[0]:
        raise ValueError(f"Bond dimension mismatch between mpo1 right bond ({mpo1.shape[-1]}) "
                         f"and mpo2 left bond ({mpo2.shape[0]}).")
    if len(gate.shape) != 4:
        raise ValueError(f"Gate tensor must be 4D. Got shape {gate.shape}.")

    # Infer expected physical dimension (assuming square, e.g., 2 for qubits)
    phys_dim = mpo1.shape[1] # Assume p1_out = p1_in = p2_out = p2_in
    if (mpo1.shape[1] != mpo1.shape[2] or mpo2.shape[1] != mpo2.shape[2] or
            mpo1.shape[1] != mpo2.shape[1] or mpo1.shape[1] != gate.shape[0] or
            mpo1.shape[1] != gate.shape[1] or mpo1.shape[1] != gate.shape[2] or
            mpo1.shape[1] != gate.shape[3]):
        print(f"Warning: Physical dimensions seem inconsistent. "
              f"mpo1: {mpo1.shape}, mpo2: {mpo2.shape}, gate: {gate.shape}. Assuming dimension {phys_dim}.")
        # You might add stricter checks here depending on expected gate/MPO structure

    # --- Contraction ---
    if gate_is_below: # i.e. if layer is below
        # Path: 'iabc, cdef, begh -> iadghf'
        # mpo1: i a b c (l1, p1_out, p1_in, r1)
        # mpo2: c d e f (r1, p2_out, p2_in, r2)
        # gate: b e g h (gate_p1_out=p1_in, gate_p2_out=p2_in, gate_p1_in, gate_p2_in)
        # Result: i a d g h f (l1, p1_out, p2_out, g_p1_in, g_p2_in, r2)
        # Check physical dimensions compatibility
        if gate.shape[0] != mpo1.shape[2] or gate.shape[1] != mpo2.shape[2]:
             raise ValueError(f"Physical dimension mismatch for gate_is_below=True. "
                              f"Gate output dims ({gate.shape[0]},{gate.shape[1]}) vs "
                              f"MPO p_in dims ({mpo1.shape[2]},{mpo2.shape[2]}).")

        merged_tensor = jnp.einsum('iabc, cdef, begh -> iadghf', mpo1, mpo2, gate, optimize='optimal')

    else: # gate_is_below is False
        # Path: 'abcd, icef, fdgh -> iabegh'
        # gate: a b c d (gate_p1_out, gate_p2_out, gate_p1_in=p1_out, gate_p2_in=p2_out)
        # mpo1: i c e f (l1, p1_out, p1_in, r1)
        # mpo2: f d g h (r1, p2_out, p2_in, r2)
        # Result: i a b e g h (l1, g_p1_out, g_p2_out, p1_in, p2_in, r2)
         # Check physical dimensions compatibility
        if gate.shape[2] != mpo1.shape[1] or gate.shape[3] != mpo2.shape[1]:
             raise ValueError(f"Physical dimension mismatch for gate_is_below=False. "
                              f"Gate input dims ({gate.shape[2]},{gate.shape[3]}) vs "
                              f"MPO p_out dims ({mpo1.shape[1]},{mpo2.shape[1]}).")

        merged_tensor = jnp.einsum('abcd, icef, fdgh -> iabegh', gate, mpo1, mpo2, optimize='optimal')


    return merged_tensor

def compress_SVD(
    u: jnp.ndarray,
    s: jnp.ndarray,
    vh: jnp.ndarray,
    max_bondim: Optional[int] = None,
    cutoff: float = 1e-12
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    """
    Truncates SVD results based on maximum bond dimension and/or cutoff.
    This version performs physical truncation (changes array shapes),
    which is memory-efficient but not JIT-compatible with dynamic shapes.

    Args:
        u: Left singular vectors.
        s: Singular values (1D array, sorted descending).
        vh: Right singular vectors (vh = v.conj().T).
        max_bondim: The maximum number of singular values to keep.
        cutoff: Singular values below this threshold (relative to the largest)
                are discarded.

    Returns:
        A tuple containing:
        - u_trunc: Truncated left singular vectors.
        - s_trunc: Truncated singular values.
        - vh_trunc: Truncated right singular vectors.
        - k_trunc: The number of singular values kept.
    """
    if s.size == 0:
        return u, s, vh, 0

    # Determine dynamic rank based on cutoff
    if cutoff > 0.0:
        denom = jnp.where(s[0] > 0, s[0], 1.0)
        k_cutoff = jnp.sum(s / denom > cutoff)
    else:
        k_cutoff = s.size

    # Cap by max_bondim if provided
    if max_bondim is not None:
        k_trunc = jnp.minimum(k_cutoff, max_bondim)
    else:
        k_trunc = k_cutoff

    # Ensure at least one singular value is kept and cast to concrete int for slicing
    k_trunc = int(jnp.maximum(k_trunc, 1))

    # Perform physical truncation (slicing)
    u_trunc = u[:, :k_trunc]
    s_trunc = s[:k_trunc]
    vh_trunc = vh[:k_trunc, :]

    return u_trunc, s_trunc, vh_trunc, k_trunc


def split_tensor_into_half_canonical_mpo_pair(
    merged_tensor: jnp.ndarray,
    canonical_mode: str = 'left',
    max_bondim: Optional[int] = None,
    svd_cutoff: float = 1e-12
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Splits a 6-index merged tensor (from merge_two_mpos_and_gate)
    into two 4-index MPO tensors using SVD, creating a half-canonical pair.

    The merged tensor is assumed to have indices corresponding to:
    (left_virt, phys1_out, phys2_out, phys1_in, phys2_in, right_virt)
    where (phys1_out, phys1_in) are the physical indices associated with the
    first resulting MPO tensor, and (phys2_out, phys2_in) with the second.

    Args:
        merged_tensor: The 6-index tensor output from merging two MPOs and a gate.
        canonical_mode: Specifies which output tensor should be canonical.
            - 'left': The left output tensor (mpo1) will be left-canonical (orthogonal).
            - 'right': The right output tensor (mpo2) will be right-canonical (orthogonal).
        max_bondim: Maximum virtual bond dimension allowed for the new bond
                    connecting the two output tensors. Defaults to None (no limit).
        svd_cutoff: Cutoff threshold for truncating singular values during SVD.

    Returns:
        A tuple containing:
        - mpo1: The left MPO tensor (shape: left_virt, phys1_out, phys1_in, new_virt).
        - mpo2: The right MPO tensor (shape: new_virt, phys2_out, phys2_in, right_virt).

    Raises:
        ValueError: If the input tensor is not 6D or if canonical_mode is invalid.
                                                                                                                              
    Note: depending on canonical mode: left or right mpo are respectively left or right canonicalized
                       p1_out      p2_out                                             
                          |         |                                                                                         
                          |         |                                    | p1_out                  | p2_out                           
                     +-------------------+                               |                         |                                  
                l    |                   |  r                   l      +----+new_dim     new_dim +----+     r                         
                -----|                   |-----   ---------->     -----|mpo1|-----          -----|mpo2|-----                          
                     |                   |                             +----+                    +----+                               
                     +-------------------+                               |                         |                                  
                          |         |                                    | p1_in                   | p2_in                            
                          |         |                                                                                         
                       gp_in1      gp_in2                                                                                 
    """
    if len(merged_tensor.shape) != 6:
        raise ValueError(f"Input merged_tensor must be 6D, got shape {merged_tensor.shape}")

    original_shape = merged_tensor.shape
    l_dim = original_shape[0]
    p1o_dim, p2o_dim = original_shape[1], original_shape[2]
    p1i_dim, p2i_dim = original_shape[3], original_shape[4]
    r_dim = original_shape[5]

    # --- Transpose and Reshape for SVD ---
    # Group indices for the left tensor (rows): (l, p1o, p1i) -> indices (0, 1, 3)
    # Group indices for the right tensor (columns): (p2o, p2i, r) -> indices (2, 4, 5)
    # Permute axes from (0, 1, 2, 3, 4, 5) to (0, 1, 3, 2, 4, 5)
    permuted_tensor = jnp.transpose(merged_tensor, axes=(0, 1, 3, 2, 4, 5))

    # Reshape into matrix: rows = l*p1o*p1i, cols = p2o*p2i*r
    rows_dim = l_dim * p1o_dim * p1i_dim
    cols_dim = p2o_dim * p2i_dim * r_dim
    matrix_to_decompose = permuted_tensor.reshape((rows_dim, cols_dim))

    # --- Perform SVD ---
    U, S, Vh = jnp.linalg.svd(matrix_to_decompose, full_matrices=False)

    # --- Truncate (Compress) ---
    U_trunc, S_trunc, Vh_trunc, k_trunc = compress_SVD(U, S, Vh, max_bondim, svd_cutoff)

    # --- Form Half-Canonical Pair and Reshape Back ---
    # Use static shape of truncated U to determine new bond dimension for reshape
    new_bond_dim = U_trunc.shape[1]
    S_mat = jnp.diag(S_trunc) # Convert 1D S_trunc to diagonal matrix

    if canonical_mode == 'left':
        # mpo1 = U (left canonical)
        # mpo2 = S @ Vh (absorbs singular values)
        mpo1_matrix = U_trunc
        mpo2_matrix = S_mat @ Vh_trunc

        # Reshape mpo1_matrix (rows_dim, k) back to tensor (l, p1o, p1i, k)
        mpo1 = mpo1_matrix.reshape((l_dim, p1o_dim, p1i_dim, new_bond_dim))

        # Reshape mpo2_matrix (k, cols_dim) back to tensor (k, p2o, p2i, r)
        mpo2 = mpo2_matrix.reshape((new_bond_dim, p2o_dim, p2i_dim, r_dim))

    elif canonical_mode == 'right':
        # mpo1 = U @ S (absorbs singular values)
        # mpo2 = Vh (right canonical)
        mpo1_matrix = U_trunc @ S_mat
        mpo2_matrix = Vh_trunc

        # Reshape mpo1_matrix (rows_dim, k) back to tensor (l, p1o, p1i, k)
        mpo1 = mpo1_matrix.reshape((l_dim, p1o_dim, p1i_dim, new_bond_dim))

        # Reshape mpo2_matrix (k, cols_dim) back to tensor (k, p2o, p2i, r)
        mpo2 = mpo2_matrix.reshape((new_bond_dim, p2o_dim, p2i_dim, r_dim))

    else:
        raise ValueError(f"Invalid canonical_mode '{canonical_mode}'. Choose 'left' or 'right'.")

    return mpo1, mpo2

def contract_mpo_with_layer_right_to_left(
    mpo_init: MPO,
    layer: GateLayer,
    layer_is_below: bool,
    max_bondim: Optional[int] = None,
    svd_cutoff: float = 1e-12
) -> MPO:
    """
    Contracts an MPO with a circuit layer from right to left. (Corrected R absorption)

    Handles both 1-qubit and 2-qubit gates within the layer. Uses RQ
    decomposition for single sites and SVD + RQ for two-qubit gate merges
    to maintain right-canonical form during the sweep.

    Args:
        mpo_init: The initial MPO object.
        layer: The GateLayer object containing gates to contract.
        layer_is_below: True if the gate layer is logically below the MPO
                       (contracts with MPO's p_in), False if above
                       (contracts with MPO's p_out).
        max_bondim: Maximum bond dimension for SVD truncation.
        svd_cutoff: Cutoff for SVD truncation.

    Returns:
        A new MPO object representing the contracted result. This MPO
        will be approximately right-canonical.
    """
    n_sites = mpo_init.n_sites
    if n_sites == 0:
        return MPO(tensors=[])
    if n_sites < 2:
        print("Warning: Left-to-right sweep requires at least 2 sites for 2Q gate logic. Behavior might be limited for n_sites=1.")
        # Handle n_sites=1 separately if needed, depends on gate types allowed


    initial_tensors = mpo_init.tensors
    dtype = initial_tensors[0].dtype

    # --- Prepare Gate Lookup ---
    # Initialize with None since at most one gate (or no gate) acts on each qubit.
    _ , gate_map_right = gate_map(layer=layer, n_sites=n_sites)
    # --- Right-to-Left Sweep ---
    mpo_res_tensors: List[jnp.ndarray] = [None] * n_sites
    # R_factor: shape (old_right_bond_dim, new_right_bond_dim) from perspective of mpo_i
    right_bond_dim = initial_tensors[-1].shape[-1]
    R_to_carry_left = jnp.eye(right_bond_dim, dtype=dtype)

    i = n_sites - 1
    while i >= 0:
        # Retrieve the gate (or None if no gate acts on site i)
        gate_acting_here = gate_map_right.get(i)
        gate_1q = None
        gate_2q = None

        # Only proceed if a gate is found at this site
        if gate_acting_here is not None:
            if gate_acting_here.is_two_qubit():
                # For a valid two-qubit gate, the gate should act on (i-1, i)
                if i > 0 and tuple(sorted(gate_acting_here.qubits)) == (i - 1, i):
                    gate_2q = gate_acting_here
                else:
                    print("Warning: Two-qubit gate found with unexpected qubit ordering.")
            elif gate_acting_here.is_single_qubit() and gate_acting_here.qubits == (i,):
                gate_1q = gate_acting_here

        # --- Process Site(s) ---
        if gate_2q is not None:
            # --- Two-Qubit Gate Case ---
            if i == 0: raise RuntimeError("2Q gate logic error at site i=0.")

            mpo_i = initial_tensors[i]
            mpo_im1 = initial_tensors[i - 1]

            # 1. Absorb R factor into mpo_i (**CORRECTED EINSUM**)
            #  mpo_i shape (l,p,p,r). R_factor shape (r, new_r). Contract 'r'.
            # Output shape (l,p,p,new_r)
            mpo_i_prime = jnp.einsum('iabj, jk -> iabk', mpo_i, R_to_carry_left, optimize='optimal')

            # 2. Merge MPOs and the Gate
            merged_T = merge_two_mpos_and_gate(mpo_im1, mpo_i_prime, gate_2q.tensor, gate_is_below=layer_is_below)

            # 3. Split back using SVD (right part canonical)
            mpo_im1_temp, mpo_i_final = split_tensor_into_half_canonical_mpo_pair(
                merged_T, canonical_mode='right', max_bondim=max_bondim, svd_cutoff=svd_cutoff
            )
            
           # 5. Process the left part (mpo_im1_temp)
            if i - 1 == 0:
                # This is the leftmost tensor, do NOT canonicalize. It absorbed the final R implicitly during split.
                mpo_res_tensors[i - 1] = mpo_im1_temp
                R_to_carry_left = None # No more R factor needed
            else:
                # Canonicalize the left part to get final tensor and NEW R factor
                mpo_im1_final, R_new = canonicalize_local_tensor(mpo_im1_temp, mode='right')

                mpo_res_tensors[i - 1] = mpo_im1_final
                R_to_carry_left = R_new
            
            mpo_res_tensors[i] = mpo_i_final

            # 6. Decrement loop counter
            i -= 2

        else:
            # --- Single-Qubit Gate or No Gate Case ---
            current_tensor = initial_tensors[i]

            # 1. Absorb R factor (**CORRECTED EINSUM**)
            # R_factor shape (r, new_r). current_tensor shape (l,p,p,r). Contract 'r'.
            # Output shape (l,p,p,new_r)
            tensor_after_absorb = jnp.einsum('iabj, jk -> iabk', current_tensor, R_to_carry_left, optimize='optimal')

            # 2. Apply 1Q gate if present, else do nothing
            tensor_after_gate = tensor_after_absorb
            if gate_1q is not None:
                tensor_after_gate = merge_one_mpo_and_gate(mpo=tensor_after_absorb, gate=gate_1q.tensor, gate_is_below=layer_is_below)

            
            # 3. Canonicalize or Store Final Tensor
            if i == 0:
                # This is the leftmost tensor, do NOT canonicalize.
                mpo_res_tensors[i] = tensor_after_gate
                R_to_carry_left = None # No more R factor needed
            else:
                # Canonicalize the result -> get final tensor and NEW R factor
                Q_final, R_new = canonicalize_local_tensor(tensor_after_gate, mode='right')
                mpo_res_tensors[i] = Q_final
                R_to_carry_left = R_new

            # 4. Decrement loop counter
            i -= 1

     # NOTE: Cannot handle n_sites=1 case.Add error if n_sites<2. 
    
    # --- Final Check and MPO Creation ---
    if any(t is None for t in mpo_res_tensors):
         missing_indices = [idx for idx, t in enumerate(mpo_res_tensors) if t is None]
         raise RuntimeError(f"MPO contraction resulted in None tensors at indices: {missing_indices}. Check loop logic.")

    final_mpo = MPO(tensors=mpo_res_tensors)
    # Mark as only partially canonical
    final_mpo.is_right_canonical = True # Although site 0 is not canonical

    return final_mpo

# TODO: we should be canonicalizing also in case the gate is not present there (i.e. gate is None)!!
# add this to both contraction left_to_right as well as right_to_left.
# TODO: add a test that covers this. 
def contract_mpo_with_layer_left_to_right(
    mpo_init: MPO,
    layer: GateLayer,
    layer_is_below: bool,
    max_bondim: Optional[int] = None,
    svd_cutoff: float = 1e-12
) -> MPO:
    """
    Contracts an MPO with a circuit layer from left to right.

    Handles both 1-qubit and 2-qubit gates within the layer. Uses QR
    decomposition for single sites (i < n_sites-1) and SVD + QR for
    two-qubit gate merges (i < n_sites-2) to maintain left-canonical
    form during the sweep. Does NOT canonicalize the final tensor at site n-1.

    Args:
        mpo_init: The initial MPO object.
        layer: The GateLayer object containing gates to contract.
        layer_is_below: True if the gate layer is logically below the MPO
                       (contracts with MPO's p_in), False if above
                       (contracts with MPO's p_out).
        max_bondim: Maximum bond dimension for SVD truncation.
        svd_cutoff: Cutoff for SVD truncation.

    Returns:
        A new MPO object representing the contracted result. Tensors at
        indices 0 to n_sites-2 are approximately left-canonical.
    """
    n_sites = mpo_init.n_sites     # need to handle minimum 3 sites

    initial_tensors = mpo_init.tensors
    dtype = initial_tensors[0].dtype

    # --- Prepare Gate Lookup ---
    # Store gates based on the *leftmost* site they act on
    gate_map_left, _ = gate_map(layer=layer, n_sites=n_sites)
    # --- Left-to-Right Sweep ---
    mpo_res_tensors: List[jnp.ndarray] = [None] * n_sites
    # R_factor: shape (new_left_bond_dim, old_left_bond_dim) from perspective of mpo_i+1
    left_bond_dim = initial_tensors[0].shape[0]
    R_to_carry_right = jnp.eye(left_bond_dim, dtype=dtype)

    i = 0
    while i < n_sites:
        gate_acting_here = gate_map_left.get(i)
        gate_1q = None
        gate_2q = None

        if gate_acting_here is not None:
            if gate_acting_here.is_two_qubit():
                # Check if gate acts on (i, i+1)
                if i + 1 < n_sites and tuple(sorted(gate_acting_here.qubits)) == (i, i + 1):
                    gate_2q = gate_acting_here
                else:
                     print("Warning: Two-qubit gate found with unexpected qubit ordering or boundary.")
            elif gate_acting_here.is_single_qubit() and gate_acting_here.qubits == (i,):
                gate_1q = gate_acting_here

        # --- Process Site(s) ---
        if gate_2q is not None:
            # --- Two-Qubit Gate Case (acting on i, i+1) ---
            mpo_i = initial_tensors[i]
            mpo_ip1 = initial_tensors[i + 1]

            # 1. Absorb R factor into mpo_i
            # R_factor shape (new_l, l). mpo_i shape (l,p,p,r). Contract 'l'.
            # Output shape (new_l,p,p,r)
            mpo_i_prime = jnp.einsum('ij, jabk -> iabk', R_to_carry_right, mpo_i, optimize='optimal')

            # 2. Merge MPOs and the Gate
            # Pass `gate_is_below` correctly to `merge_two_mpos_and_gate`'s `gate_is_below` parameter
            merged_T = merge_two_mpos_and_gate(mpo_i_prime, mpo_ip1, gate_2q.tensor, gate_is_below=layer_is_below)

            # 3. Split back using SVD (left part canonical)
            mpo_i_final, mpo_ip1_temp = split_tensor_into_half_canonical_mpo_pair(
                merged_T, canonical_mode='left', max_bondim=max_bondim, svd_cutoff=svd_cutoff
            )

            # 4. Store the canonical left part
            mpo_res_tensors[i] = mpo_i_final

            # 5. Process the right part (mpo_ip1_temp)
            if i + 1 == n_sites - 1:
                # This is the rightmost tensor, do NOT canonicalize.
                mpo_res_tensors[i + 1] = mpo_ip1_temp
                R_to_carry_right = None # No more R factor needed
            else:
                # Canonicalize the right part to get final tensor and NEW R factor
                mpo_ip1_final, R_new = canonicalize_local_tensor(mpo_ip1_temp, mode='left')
                mpo_res_tensors[i + 1] = mpo_ip1_final
                R_to_carry_right = R_new

            # 6. Increment loop counter
            i += 2

        else:
            # --- Single-Qubit Gate or No Gate Case (acting on i) ---
            current_tensor = initial_tensors[i]

            # 1. Absorb R factor
            # R_factor shape (new_l, l). current_tensor shape (l,p,p,r). Contract 'l'.
            # Output shape (new_l,p,p,r)
            tensor_after_absorb = jnp.einsum('ij, jabk -> iabk', R_to_carry_right, current_tensor, optimize='optimal')

            # 2. Apply 1Q gate if present
            tensor_after_gate = tensor_after_absorb
            if gate_1q is not None:
                tensor_after_gate = merge_one_mpo_and_gate(mpo=tensor_after_absorb, gate=gate_1q.tensor, gate_is_below=layer_is_below)                

            # 3. Canonicalize or Store Final Tensor
            if i == n_sites - 1:
                # This is the rightmost tensor, do NOT canonicalize.
                mpo_res_tensors[i] = tensor_after_gate
                R_to_carry_right = None # No more R factor needed
            else:
                # Canonicalize the result -> get final tensor and NEW R factor
                Q_final, R_new = canonicalize_local_tensor(tensor_after_gate, mode='left')
                mpo_res_tensors[i] = Q_final
                R_to_carry_right = R_new

            # 4. Increment loop counter
            i += 1

    # --- Final Check and MPO Creation ---
    if any(t is None for t in mpo_res_tensors):
         missing_indices = [idx for idx, t in enumerate(mpo_res_tensors) if t is None]
         raise RuntimeError(f"MPO contraction resulted in None tensors at indices: {missing_indices}. Check loop logic.")

    final_mpo = MPO(tensors=mpo_res_tensors)
    # Mark as only partially canonical
    final_mpo.is_left_canonical = True # Although last site is not canonical.

    return final_mpo

def contract_mpo_with_layer(
    mpo_init: MPO,
    layer: GateLayer,
    layer_is_below: bool,
    max_bondim: Optional[int] = None,
    svd_cutoff: float = 1e-12,
    direction: str = None
):
    """
    Applies contract_mpo_with_layer_right_to_left or contract_mpo_with_layer_left_to_right depending on the specified direction.
    """
    if direction == 'left_to_right':
        return contract_mpo_with_layer_left_to_right(mpo_init, layer, layer_is_below, max_bondim, svd_cutoff)
    elif direction == 'right_to_left':
        return contract_mpo_with_layer_right_to_left(mpo_init, layer, layer_is_below, max_bondim, svd_cutoff)
    else:
        raise ValueError("Please specify either either as 'left_to_right' or 'right_to_left'.")