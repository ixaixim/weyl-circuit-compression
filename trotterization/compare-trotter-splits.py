"""
Spectral error ||U_exact - U_Trotter||_2 of three second-order Trotter splittings of the Ising
model (N=10, t=2, dense matrices) as a function of the time step:
  - "Riemannian":            even-odd splitting, field terms split between both halves (RO ansatz),
  - "HW-Aware (split ZZ)":   field layer between the even and odd R_ZZ layers,
  - "HW-Aware (unified ZZ)": all R_ZZ layers, then the field layer (PO ansatz).

Usage: python compare-trotter-splits.py [--out CSV]
"""
import weyl_compression.jax_config

import argparse
import csv
import os
import time
import numpy as np
import jax.numpy as jnp
from jax.scipy.linalg import expm

from weyl_compression.hamiltonian.xyz_model import XYZModel
from weyl_compression.circuit.trotter.field_middle import field_middle_trotter_circuit
from weyl_compression.circuit.trotter.even_odd import even_odd_trotter_circuit
from weyl_compression.circuit.trotter.hardware_aware_ising import ising_hardware_aware_trotter_circuit

def run_comparison(out_path):
    # 1. Setup Parameters
    n_sites = 10
    Jx, Jy, Jz = 0.0, 0.0, 1.0 # Ising model
    hx, hy, hz = 0.75, 0.0, 0.6
    total_time = 2.0
    
    # Orders to compare
    orders = [2] 
    
    print(f"Comparing Trotter Methods for Ising Model, N={n_sites}, T={total_time}")
    print(f"Couplings: J={Jz}, h=({hx},{hz})")

    # 2. Exact Evolution
    print("Computing exact evolution...")
    model = XYZModel(n_sites=n_sites, Jx=Jx, Jy=Jy, Jz=Jz, 
                     hx=hx, hy=hy, hz=hz, dtype=jnp.complex128)
    H_full = model.build_hamiltonian_matrix()
    U_exact = expm(-1j * total_time * H_full)

    # 3. Sweep parameters
    n_steps_list = np.unique(np.logspace(0.6, 2.0, 15).astype(int))
    dt_list = total_time / n_steps_list
    
    results = {
        "Riemannian": {o: [] for o in orders},
        "HW-Aware (split ZZ)": {o: [] for o in orders},
        "HW-Aware (unified ZZ)": {o: [] for o in orders}
    }

    print(f"Sweeping over {len(n_steps_list)} time steps...")
    sweep_start = time.time()
    for i, n_steps in enumerate(n_steps_list):
        dt = total_time / n_steps
        print(f"[{i+1}/{len(n_steps_list)}] n_steps={n_steps}, dt={dt:.4g} "
              f"(elapsed {time.time() - sweep_start:.1f}s)")

        for order in orders:
            # --- Standard Method ---
            circ_std = even_odd_trotter_circuit(
                n_sites=n_sites, Jx=Jx, Jy=Jy, Jz=Jz, hx=hx, hy=hy, hz=hz,
                order=order, dt=dt, reps=n_steps, dtype=jnp.complex128
            )
            U_std = circ_std.to_matrix()
            err_std = float(np.linalg.norm(U_exact - U_std, ord=2))
            results["Riemannian"][order].append(err_std)
            
            # --- XYZ Hardware Friendly Method ---
            circ_xyz_hw = field_middle_trotter_circuit(
                n_sites=n_sites, Jx=Jx, Jy=Jy, Jz=Jz, hx=hx, hy=hy, hz=hz,
                order=order, dt=dt, reps=n_steps, collapse=True, dtype=jnp.complex128
            )
            U_xyz_hw = circ_xyz_hw.to_matrix()
            err_xyz_hw = float(np.linalg.norm(U_exact - U_xyz_hw, ord=2))
            results["HW-Aware (split ZZ)"][order].append(err_xyz_hw)

            # --- Ising Hardware Friendly Method ---
            # New specialized method
            circ_ising_hw = ising_hardware_aware_trotter_circuit(
                n_sites=n_sites, J=Jz, hx=hx, hz=hz,
                dt=dt, reps=n_steps, dtype=jnp.complex128
            )
            U_ising_hw = circ_ising_hw.to_matrix()
            err_ising_hw = float(np.linalg.norm(U_exact - U_ising_hw, ord=2))
            results["HW-Aware (unified ZZ)"][order].append(err_ising_hw)

            print(f"    order={order}: Riemannian={err_std:.3e}, "
                  f"split-ZZ={err_xyz_hw:.3e}, unified-ZZ={err_ising_hw:.3e}")

    # 4. Save results to CSV
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "n_sites", "total_time", "method", "order", "n_steps", "dt", "error"
        ])
        writer.writeheader()
        for method in results:
            for order in orders:
                for n_steps, dt, error in zip(n_steps_list, dt_list, results[method][order]):
                    writer.writerow({
                        "n_sites": n_sites,
                        "total_time": total_time,
                        "method": method,
                        "order": order,
                        "n_steps": n_steps,
                        "dt": dt,
                        "error": error,
                    })
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "plotting-results",
                                                      "data", "trotter_splits_ising_n10.csv"))
    run_comparison(parser.parse_args().out)
