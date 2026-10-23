"""
Generate the data figures of the paper from the compact data in plotting-results/data/ and save
them to plotting-results/gfx/ (same file names as in the paper).

  - run_riemannian-opt_20260520_185744.pkl (Ising), run_riemannian-opt_20260616_020031.pkl (XXZ):
                                                    Riemannian optimization runs (N = 50, 7 Trotter repetitions);
                                                    the Weyl coordinates of their gates are computed here
  - trotter_splits_ising_n10.csv:                   spectral error of three Trotter splittings
                                                    (from ../trotterization/compare-trotter-splits.py)
  - {ising,heisenberg}_comparison_n50.pkl:          loss vs. native depth, RO vs. PO (values plotted in the paper)
  - riemannian_loss_history_heisenberg_n50.pkl:     Riemannian Adam loss history
The degrees-of-freedom figure is computed from closed-form gate counts.

Usage: python plotting-results.py [--data-dir DIR] [--gfx-dir DIR]
"""
import argparse
import csv
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter
from qiskit.synthesis import TwoQubitWeylDecomposition

HERE = Path(__file__).resolve().parent
data_dir = HERE / "data"
gfx_dir = HERE / "gfx"

plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["figure.figsize"] = 4, 3


def load(name):
    with open(data_dir / name, "rb") as fh:
        return pickle.load(fh)


def save(name, **kwargs):
    # A fixed (empty) creation date keeps the PDF files reproducible byte for byte.
    plt.savefig(gfx_dir / name, metadata={"CreationDate": None}, **kwargs)
    print(f"Saved {gfx_dir / name}")


# -----------------------------------------------------------------------------
# Weyl coordinates of Riemannian-optimized gates (N = 50, 7 Trotter repetitions)
# -----------------------------------------------------------------------------

COLORS = ['#4C72B0', '#DD8452', '#55A868']
PARAM_LABELS = [r'$\alpha_x$', r'$\alpha_y$', r'$\alpha_z$']
LABEL_FONTSIZE = 13
TICK_FONTSIZE = 11
YLABEL_FONTSIZE = 11
LEGEND_FONTSIZE = 9

# Riemannian runs (N=50, 7 Trotter repetitions) shown in the Weyl-coordinate figures.
WEYL_RUNS = {
    "ising": "run_riemannian-opt_20260520_185744.pkl",
    "heisenberg": "run_riemannian-opt_20260616_020031.pkl",
}


def weyl_coordinates(run_file):
    """
    Weyl coordinates (a, b, c) = (alpha_x, alpha_y, alpha_z) of every two-qubit gate of a run file,
    from the KAK decomposition (qiskit.synthesis.TwoQubitWeylDecomposition, default settings).
    """
    circuit = load(run_file)["circuit"]
    coords = []
    for layer in sorted(circuit["layers"], key=lambda l: l["layer_index"]):
        for gate in sorted(layer["gates"], key=lambda g: g["qubits"][0]):
            if len(gate["qubits"]) == 2:
                m = gate["matrix"]
                matrix = np.asarray(m["real"], dtype=np.complex128) + 1j * np.asarray(m["imag"], dtype=np.complex128)
                decomp = TwoQubitWeylDecomposition(matrix)
                coords.append((decomp.a, decomp.b, decomp.c))
    return [np.array(values) for values in zip(*coords)]


def weyl_filename(run_file):
    return f"riemannian-opt_{Path(run_file).stem}_weyl_parameters.pdf"


def pi_formatter(x, pos):
    if np.isclose(x, 0):
        return "0"
    return f"{x/np.pi:.3f}π"


def _style_violin(vp, color):
    for part in ('bodies', 'cmeans', 'cmedians', 'cbars', 'cmaxes', 'cmins'):
        if part in vp:
            items = vp[part] if part == 'bodies' else [vp[part]]
            for item in items:
                item.set_color(color)
                item.set_alpha(0.72 if part == 'bodies' else 1.0)


def plot_weyl_ising():
    # Ising: log-space mean ± 1σ dot plot
    aa, bb, cc = weyl_coordinates(WEYL_RUNS["ising"])

    all_vals = [np.abs(aa), np.abs(bb), np.abs(cc)]
    all_flat = np.concatenate(all_vals)
    clip_min = all_flat[all_flat > 0].min() * 0.1
    abs_vals = [np.clip(v, clip_min, None) for v in all_vals]

    fig, ax = plt.subplots()

    for x, vals, color in zip([1, 2, 3], abs_vals, COLORS):
        log_v = np.log10(vals)
        m, s = np.mean(log_v), np.std(log_v)
        mean_lin = 10 ** m
        err_lo = mean_lin - 10 ** (m - s)
        err_hi = 10 ** (m + s) - mean_lin
        ax.errorbar(x, mean_lin, yerr=[[err_lo], [err_hi]],
                    fmt='o', color=color, markersize=9,
                    capsize=6, capthick=1.6, elinewidth=1.6)

    ax.set_yscale('log')
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(PARAM_LABELS, fontsize=LABEL_FONTSIZE)
    ax.set_xlim(0.4, 3.6)
    ax.set_ylabel(r"$|\alpha|$  (log-space mean $\pm 1\sigma$)", fontsize=YLABEL_FONTSIZE)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2e}" if x > 0 else "0"))
    ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.35)

    save(weyl_filename(WEYL_RUNS["ising"]))
    plt.close()


def plot_weyl_heisenberg():
    # Heisenberg XXZ: violin plots with the reference line 2·median(alpha_z)
    aa, bb, cc = weyl_coordinates(WEYL_RUNS["heisenberg"])

    formatter = FuncFormatter(pi_formatter)
    y_min = np.min(cc)
    y_max = np.max(aa)

    fig = plt.figure()
    gs = fig.add_gridspec(1, 3, wspace=0.3)

    for col, (vals, label, color) in enumerate(zip([aa, bb, cc], PARAM_LABELS, COLORS)):
        ax = fig.add_subplot(gs[0, col])
        ax.set_ylim(y_min, y_max)

        vp = ax.violinplot([vals], showmeans=True, showmedians=True)
        _style_violin(vp, color)
        ax.set_xticks([1])
        ax.set_xticklabels([label], fontsize=LABEL_FONTSIZE)
        ax.set_ylabel("Value", fontsize=YLABEL_FONTSIZE)
        ax.yaxis.set_major_formatter(formatter)
        ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.35)
        if col > 0:
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.set_ylabel("")

        # Dashed reference at 2·median(αz) on αx and αy panels
        if col < 2:
            ref = 2.0 * np.median(cc)
            ax.axhline(ref, color='crimson', linestyle='--', linewidth=1.8,
                       alpha=0.85, label=r'$2\,\tilde{\alpha}_z$')

    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", fontsize=LEGEND_FONTSIZE)
    save(weyl_filename(WEYL_RUNS["heisenberg"]))
    plt.close()


# -----------------------------------------------------------------------------
# Trotter splittings of the Ising model (N = 10, t = 2)
# -----------------------------------------------------------------------------

def plot_trotter_splits():
    STYLES = {
        "Riemannian": dict(marker='o', linestyle='-', color='royalblue'),
        "HW-Aware (split ZZ)": dict(marker='s', linestyle='--', color='seagreen'),
        "HW-Aware (unified ZZ)": dict(marker='^', linestyle='-.', color='darkorange'),
    }

    # data[method][order] -> list of (n_steps, dt, error), sorted by n_steps
    data = defaultdict(lambda: defaultdict(list))
    with open(data_dir / "trotter_splits_ising_n10.csv", newline="") as f:
        for row in csv.DictReader(f):
            data[row["method"]][int(row["order"])].append(
                (int(row["n_steps"]), float(row["dt"]), float(row["error"]))
            )

    fig, ax = plt.subplots()

    for method, per_order in data.items():
        style = STYLES.get(method, {})
        for order, rows in per_order.items():
            rows.sort(key=lambda r: r[0])
            dt_list = [dt for _, dt, _ in rows]
            err_list = [err for _, _, err in rows]
            label = method.replace("HW-Aware", "HW").replace(" (", " ").replace(")", "")
            ax.loglog(dt_list, err_list, label=f"{label}, order {order}", **style)

    ax.set_xticks([0.02, 0.05, 0.1, 0.5],
                  labels=["0.02", "0.05", "0.1", "0.5"])
    ax.xaxis.set_minor_formatter(NullFormatter())

    plt.xlabel(r'Time step $\Delta t$')
    plt.ylabel(r'Spectral Error $\|U_{exact} - U_{approx}\|_2$')
    fig.legend(loc="outside upper center", ncol=2, fontsize=7.5,
               handlelength=1.5, labelspacing=0.4, columnspacing=1.0)
    plt.grid(True, which="both", ls="--", alpha=0.4)

    save("compare_trotter_methods_ising.pdf", bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------------------------
# Degrees of freedom of the Ising circuits
# -----------------------------------------------------------------------------

def calculate_dofs(n_sites, reps):
    # Riemannian Formula (SU(4) gates)
    # (reps+1) even layers of n_sites//2 gates + reps odd layers of n_sites//2-1 gates
    # 15 params/gate * total gate count
    riem_gates = (n_sites // 2) * (reps + 1) + (n_sites // 2 - 1) * reps
    riem_dof = 15 * riem_gates

    # Ising Formula
    # (reps+1) even AND (reps+1) odd RZZ interaction layers (1 param each),
    # matching the actual hardware-friendly circuit topology (symmetric even/odd
    # layer counts, unlike the Riemannian layout), plus reps field layers of
    # n_sites single-qubit rotations (3 params each)
    rzz_gates = (n_sites // 2) * (reps + 1) + (n_sites // 2 - 1) * (reps + 1)
    field_gates = reps * n_sites
    ising_dof = 1 * rzz_gates + 3 * field_gates

    return riem_dof, ising_dof


def plot_dof_scaling():
    # Values quoted in the paper for N = 50, r = 5
    assert calculate_dofs(50, 5) == (4050, 1044)

    # Data generation
    n_range = np.arange(4, 52, 2)
    reps_to_plot = [1, 5, 10]
    alphas = [0.35, 0.65, 1.0]  # Trotter repetitions: 1, 5, 10 (lighter = fewer reps)

    fig = plt.figure()

    for i, r in enumerate(reps_to_plot):
        dofs = [calculate_dofs(n, r) for n in n_range]
        riem, ising = zip(*dofs)

        # Plot Riemannian
        plt.plot(n_range, riem, label=f'Riemannian, r={r}',
                 color='royalblue', marker='o', alpha=alphas[i])
        # Plot Ising
        plt.plot(n_range, ising, label=f'Hardware-aware, r={r}',
                 color='darkorange', marker='^', markerfacecolor='none', alpha=alphas[i])

    plt.xlabel('Number of spins, $N$')
    plt.ylabel('Total Degrees of Freedom')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    # r denotes the number of Trotter repetitions.
    # Matplotlib fills legend columns first; group methods to align repetitions by row.
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = handles[::2] + handles[1::2]
    labels = labels[::2] + labels[1::2]
    fig.legend(handles, labels, loc='outside upper center', ncol=2, fontsize=8,
               handlelength=1.5, labelspacing=0.4, columnspacing=1.0)

    save("dof_scaling.pdf", bbox_inches='tight')
    plt.close()


# -----------------------------------------------------------------------------
# Loss vs. native entangling-gate depth (N = 50) and Riemannian Adam loss history
# -----------------------------------------------------------------------------

def curve(data, method, key):
    points = data["curves"][method]["points"]
    return [p["native_depth"] for p in points], [p[key] for p in points]


def plot_ising_comparison():
    ising = load("ising_comparison_n50.pkl")

    fig, ax = plt.subplots()

    ax.plot(*curve(ising, "riemannian", "optimized_loss"), 'o-', color='royalblue', linewidth=2,
            markersize=8, label='Riemannian (optimized)')
    ax.plot(*curve(ising, "hardware-aware", "optimized_loss"), 's-', color='darkorange', linewidth=2,
            markersize=8, label='Hardware-aware (optimized)')
    ax.plot(*curve(ising, "riemannian", "initial_loss"), 'o--', color='royalblue', linewidth=1.5,
            markersize=6, alpha=0.5, label='Riemannian (initial)')
    ax.plot(*curve(ising, "hardware-aware", "initial_loss"), 's--', color='darkorange', linewidth=1.5,
            markersize=6, alpha=0.5, label='Hardware-aware (initial)')

    # Formatting
    ax.set_yscale('log')
    ax.set_xlabel('Number of RZZ Layers')
    ax.set_ylabel('Loss')
    fig.legend(loc='outside upper center', ncol=2, fontsize=7.5,
               handlelength=1.5, labelspacing=0.4, columnspacing=1.0)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

    save('ising_comparison_plot.pdf', bbox_inches='tight')
    plt.close()


def plot_heisenberg_comparison():
    heisenberg = load("heisenberg_comparison_n50.pkl")

    fig, ax = plt.subplots()

    ax.plot(*curve(heisenberg, "riemannian", "optimized_loss"), 'o-', color='royalblue', linewidth=2,
            markersize=8, label='Riemannian (optimized)')
    ax.plot(*curve(heisenberg, "hardware-aware", "optimized_loss"), 's-', color='darkorange', linewidth=2,
            markersize=8, label='Hardware-aware (optimized)')
    ax.plot(*curve(heisenberg, "riemannian", "initial_loss"), 'o--', color='royalblue', linewidth=1.5,
            markersize=6, alpha=0.5, label='Riemannian (initial)')
    ax.plot(*curve(heisenberg, "hardware-aware", "initial_loss"), 's--', color='darkorange', linewidth=1.5,
            markersize=6, alpha=0.5, label='Hardware-aware (initial)')

    # Formatting
    ax.set_yscale('log')
    ax.set_xlabel('Number of Two-Qubit Gate Layers')
    ax.set_ylabel('Loss')
    fig.legend(loc='outside upper center', ncol=2, fontsize=7.5,
               handlelength=1.5, labelspacing=0.4, columnspacing=1.0)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

    save('heisenberg_comparison_plot_50q.pdf', bbox_inches='tight')
    plt.close()


def plot_loss_history():
    loss_history = load("riemannian_loss_history_heisenberg_n50.pkl")["loss_history"]

    fig, ax = plt.subplots()
    ax.plot(loss_history, color="royalblue", linewidth=1.5)

    ax.set_yscale('log')
    ax.set_xlabel('Optimization Step')
    ax.set_ylabel('Loss (Log Scale)')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

    save('riemannian_loss_history_50q.pdf', bbox_inches='tight')
    plt.close()


def main():
    global data_dir, gfx_dir
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=data_dir, help="compact data (default: plotting-results/data/)")
    parser.add_argument("--gfx-dir", type=Path, default=gfx_dir, help="output folder (default: plotting-results/gfx/)")
    args = parser.parse_args()
    data_dir, gfx_dir = args.data_dir, args.gfx_dir
    gfx_dir.mkdir(parents=True, exist_ok=True)

    plot_weyl_ising()
    plot_weyl_heisenberg()
    plot_trotter_splits()
    plot_dof_scaling()
    plot_ising_comparison()
    plot_heisenberg_comparison()
    plot_loss_history()


if __name__ == "__main__":
    main()
