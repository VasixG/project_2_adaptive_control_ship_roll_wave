from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from core.types import Array
from experiments.ship_roll_setup import ShipRollExperiment
from simulation.simulator import SimulationResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = PROJECT_ROOT / "figures"


def make_figure_dir(*parts: str) -> Path:
    path = FIGURE_DIR.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def valid_mask(result: SimulationResult) -> Array:
    return np.isfinite(result.x[:, 0]) & np.isfinite(result.x[:, 1]) & np.isfinite(result.W)


def mode_indicator(result: SimulationResult) -> Array:
    values = np.full(len(result.mode), np.nan, dtype=float)
    mapping = {
        "local": 0.0,
        "linear": 0.0,
        "local-only": 0.0,
        "outer": 1.0,
    }
    for idx, mode in enumerate(result.mode):
        values[idx] = mapping.get(mode, np.nan)
    return values


def ellipse_points(P: Array, level: float, n: int = 500) -> tuple[Array, Array]:
    theta = np.linspace(0.0, 2.0 * np.pi, n)
    circle = np.vstack((np.cos(theta), np.sin(theta)))
    values, vectors = np.linalg.eigh(P)
    transform = vectors @ np.diag(1.0 / np.sqrt(values))
    points = np.sqrt(level) * (transform @ circle)
    return points[0], points[1]


def ellipse_fill(P: Array, level: float, n_r: int = 80, n_t: int = 240) -> tuple[Array, Array]:
    values, vectors = np.linalg.eigh(P)
    transform = vectors @ np.diag(1.0 / np.sqrt(values))
    theta = np.linspace(0.0, 2.0 * np.pi, n_t)
    radius = np.linspace(0.0, np.sqrt(level), n_r)
    theta_grid, radius_grid = np.meshgrid(theta, radius)
    circle = np.vstack(
        (
            radius_grid.ravel() * np.cos(theta_grid.ravel()),
            radius_grid.ravel() * np.sin(theta_grid.ravel()),
        )
    )
    points = transform @ circle
    return points[0].reshape(radius_grid.shape), points[1].reshape(radius_grid.shape)


def add_time_colored_phase(
    ax,
    x: Array,
    y: Array,
    t: Array,
    label: str,
    cmap: str = "plasma",
):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(float(np.min(t)), float(np.max(t)))
    line_collection = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2.2, label=label)
    line_collection.set_array(t[:-1])
    artist = ax.add_collection(line_collection)
    return artist


def plot_phase_geometry(ax, experiment: ShipRollExperiment) -> None:
    P = experiment.lyapunov.P
    r_x, r_y = ellipse_points(P, experiment.levels.r_gamma)
    abar_x, abar_y = ellipse_points(P, experiment.levels.a_bar)
    aloc_x, aloc_y = ellipse_points(P, experiment.levels.a_loc)
    fill_x, fill_y = ellipse_fill(P, experiment.levels.r_gamma)
    ax.contourf(fill_x, fill_y, np.ones_like(fill_x), levels=[0.5, 1.5], alpha=0.16)
    ax.plot(r_x, r_y, "--", linewidth=2.0, label=r"$W(x)=r_\gamma$")
    ax.plot(abar_x, abar_y, "-.", linewidth=2.0, label=r"$W(x)=\bar a$")
    ax.plot(aloc_x, aloc_y, ":", linewidth=2.4, label=r"$W(x)=a_{\rm loc}$")


def plot_singular_line(ax, experiment: ShipRollExperiment, x_limits: tuple[float, float]) -> None:
    P = experiment.lyapunov.P
    p21 = P[1, 0]
    p22 = P[1, 1]
    theta_values = np.linspace(x_limits[0], x_limits[1], 400)
    p_values = -(p21 / p22) * theta_values
    ax.plot(theta_values, p_values, "k--", linewidth=1.3, label=r"$\nabla W(x)^T B=0$")


def print_design_summary(experiment: ShipRollExperiment, result: SimulationResult) -> None:
    levels = experiment.levels
    print("Hybrid adaptive ship-roll experiment")
    print(f"  a_loc={levels.a_loc:.6f}")
    print(f"  eta_margin={levels.eta_margin:.6f}")
    print(f"  a_bar={levels.a_bar:.6f}")
    print(f"  C_theta={levels.c_theta:.6f}")
    print(f"  gamma_eta={levels.gamma_eta:.6f}")
    print(f"  gamma={levels.gamma:.6f}")
    print(f"  r_gamma={levels.r_gamma:.6f}")
    print(f"  ||theta_star||={np.linalg.norm(experiment.disturbance.theta_star):.6f}")
    print(f"  R_theta={experiment.config.r_theta:.6f}")
    print("A_cl =")
    print(experiment.lyapunov_data.Acl)
    print("P =")
    print(experiment.lyapunov_data.P)
    print(f"Eigenvalues(A_cl)={experiment.lyapunov_data.eig_Acl}")
    if result.switch_time is None:
        print("  switching set was not reached")
    else:
        idx = result.switch_index
        print(f"  tau_gamma={result.switch_time:.6f}")
        print(f"  W(tau_gamma)={result.W[idx]:.6f}")
        print(f"  V_gamma(tau_gamma)={result.V_gamma[idx]:.6f}")
        print(f"  V_gamma(tau_gamma) <= a_bar: {result.V_gamma[idx] <= levels.a_bar + 1e-8}")
    print(f"  max parameter energy={np.nanmax(result.parameter_energy):.6f}")
    print(f"  C_theta/gamma={levels.c_theta / levels.gamma:.6f}")
    outer_mask = np.asarray(result.mode) == "outer"
    if np.any(outer_mask):
        print(f"  min |gradW^T B| in outer mode={np.nanmin(np.abs(result.input_gradient[outer_mask])):.6e}")
        print(f"  max |u| in outer mode={np.nanmax(np.abs(result.u[outer_mask])):.6f}")
    print(f"  max |u| total={np.nanmax(np.abs(result.u)):.6f}")
    print(f"  max |d_param - d_wave|={np.nanmax(np.abs(result.d_true - result.d_wave)):.6e}")


def print_comparison_summary(results: dict[str, SimulationResult]) -> None:
    print("Controller comparison")
    for name, result in results.items():
        valid = valid_mask(result)
        final_idx = np.where(valid)[0][-1]
        tail = valid & (result.t >= result.t[-1] - 10.0)
        rms_theta_tail = np.sqrt(np.nanmean(result.x[tail, 0] ** 2))
        rms_error_tail = np.sqrt(np.nanmean(result.disturbance_error[tail] ** 2))
        print(f"  {name}:")
        print(f"    final state={result.x[final_idx]}")
        print(f"    final W={result.W[final_idx]:.6f}")
        print(f"    tail RMS theta={rms_theta_tail:.6f} rad")
        print(f"    tail RMS disturbance error={rms_error_tail:.6f}")
        print(f"    max |u|={np.nanmax(np.abs(result.u)):.6f}")
        print(f"    switch_time={result.switch_time}")


def save_hybrid_adaptive_plots(
    experiment: ShipRollExperiment,
    result: SimulationResult,
    output_dir: Path,
) -> None:
    valid = valid_mask(result)
    t = result.t[valid]
    x = result.x[valid]
    mode_values = mode_indicator(result)
    levels = experiment.levels

    fig, axs = plt.subplots(3, 3, figsize=(18, 14), num="Hybrid adaptive ship-roll control")

    ax = axs[0, 0]
    plot_phase_geometry(ax, experiment)
    phase_artist = add_time_colored_phase(ax, x[:, 0], x[:, 1], t, "trajectory")
    x_margin = 0.25
    x_limits = (float(np.min(x[:, 0]) - x_margin), float(np.max(x[:, 0]) + x_margin))
    plot_singular_line(ax, experiment, x_limits)
    ax.scatter([x[0, 0]], [x[0, 1]], marker="o", color="green", s=120, label="Start", zorder=5)
    ax.scatter([x[-1, 0]], [x[-1, 1]], marker="x", color="red", s=120, label="Finish", zorder=5)
    ax.scatter([0.0], [0.0], marker="+", color="black", s=90, label="Target")
    if result.switch_index is not None:
        idx = result.switch_index
        ax.scatter(
            [result.x[idx, 0]],
            [result.x[idx, 1]],
            marker="X",
            color="tab:green",
            s=110,
            label=r"Switch at $\tau_\gamma$",
            zorder=6,
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Roll angle $\theta$ (rad)")
    ax.set_ylabel(r"Roll rate $p$ (rad/s)")
    ax.set_title("Phase Portrait")
    ax.grid(True)
    ax.legend(loc="best")
    colorbar = fig.colorbar(phase_artist, ax=ax)
    colorbar.set_label("Time (s)")

    ax = axs[0, 1]
    ax.plot(t, x[:, 0], label=r"$\theta$")
    ax.plot(t, x[:, 1], label=r"$p$")
    if result.switch_time is not None:
        ax.axvline(result.switch_time, color="black", linestyle=":", label=r"$\tau_\gamma$")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("State")
    ax.set_title("State Trajectories")
    ax.grid(True)
    ax.legend()

    ax = axs[0, 2]
    ax.plot(t, result.u[valid], color="tab:orange", label=r"$u$")
    if result.switch_time is not None:
        ax.axvline(result.switch_time, color="black", linestyle=":", label=r"$\tau_\gamma$")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Control torque")
    ax.set_title("Control Input")
    ax.grid(True)
    ax.legend()

    ax = axs[1, 0]
    ax.plot(t, result.W[valid], linewidth=2, label=r"$W(x)$")
    ax.axhline(levels.r_gamma, linestyle="--", linewidth=2, label=r"$r_\gamma$")
    ax.axhline(levels.a_bar, linestyle="-.", linewidth=2, label=r"$\bar a$")
    ax.axhline(levels.a_loc, linestyle=":", linewidth=2.4, label=r"$a_{\rm loc}$")
    if result.switch_time is not None:
        ax.axvline(result.switch_time, color="black", linestyle=":", label=r"$\tau_\gamma$")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$W(x)$")
    ax.set_title("Implementable Switching Variable")
    ax.grid(True)
    ax.legend()

    ax = axs[1, 1]
    ax.plot(t, result.V_gamma[valid], linewidth=2, label=r"$V_\gamma$")
    ax.axhline(levels.a_bar, linestyle="--", linewidth=2, label=r"$\bar a$")
    if result.switch_time is not None:
        ax.axvline(result.switch_time, color="black", linestyle=":", label=r"$\tau_\gamma$")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$V_\gamma$")
    ax.set_title("Extended Lyapunov Function")
    ax.grid(True)
    ax.legend()

    ax = axs[1, 2]
    ax.plot(t, result.parameter_energy[valid], linewidth=2, label=r"$\frac{1}{2\gamma}\|\tilde\theta\|^2$")
    ax.axhline(levels.c_theta / levels.gamma, linestyle="--", linewidth=2, label=r"$C_\Theta/\gamma$")
    if result.switch_time is not None:
        ax.axvline(result.switch_time, color="black", linestyle=":", label=r"$\tau_\gamma$")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy")
    ax.set_title("Parameter-Energy Bound")
    ax.grid(True)
    ax.legend()

    ax = axs[2, 0]
    ax.plot(t, result.d_true[valid], linewidth=2, label=r"$d(t)=\varphi(t)^T\theta^\star$")
    ax.plot(t, result.d_wave[valid], "--", linewidth=1.5, label=r"$d(t)=\mu(\eta_R-\eta_L)$")
    ax.plot(t, result.d_hat[valid], linewidth=2, label=r"$\hat d(t)=\varphi(t)^T\hat\theta$")
    if result.switch_time is not None:
        ax.axvline(result.switch_time, color="black", linestyle=":", label=r"$\tau_\gamma$")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Wave moment")
    ax.set_title("Wave Moment and Adaptive Estimate")
    ax.grid(True)
    ax.legend()

    ax = axs[2, 1]
    ax.plot(t, result.disturbance_error[valid], linewidth=2, label=r"$d(t)-\hat d(t)$")
    if result.switch_time is not None:
        ax.axvline(result.switch_time, color="black", linestyle=":", label=r"$\tau_\gamma$")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Disturbance estimation error")
    ax.set_title("Disturbance Tracking Error")
    ax.grid(True)
    ax.legend()

    ax = axs[2, 2]
    ax.plot(result.t, mode_values, linewidth=2, label="Hybrid mode")
    if result.switch_time is not None:
        ax.axvline(result.switch_time, color="black", linestyle=":", label=r"$\tau_\gamma$")
    ax.set_ylim([-0.1, 1.1])
    ax.set_yticks([0.0, 1.0], ["local adaptive", "outer adaptive"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mode")
    ax.set_title("Hybrid Switching Logic")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "plots.png", dpi=150)
    plt.close(fig)


def save_singularity_diagnostic(
    experiment: ShipRollExperiment,
    result: SimulationResult,
    output_dir: Path,
) -> None:
    valid = valid_mask(result)
    t = result.t[valid]
    g = result.input_gradient[valid]
    u = result.u[valid]
    threshold = experiment.config.outer_g_min

    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True, num="Singularity diagnostic")

    axs[0].plot(t, g, linewidth=2, label=r"$\nabla W(x)^T B$")
    axs[0].axhline(0.0, color="black", linewidth=1)
    axs[0].axhline(threshold, linestyle="--", linewidth=1, label="guard threshold")
    axs[0].axhline(-threshold, linestyle="--", linewidth=1)
    if result.switch_time is not None:
        axs[0].axvline(result.switch_time, color="black", linestyle=":", label=r"$\tau_\gamma$")
    axs[0].set_ylabel(r"$\nabla W(x)^T B$")
    axs[0].set_title("Outer-Control Denominator Diagnostic")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].semilogy(t, np.maximum(np.abs(g), 1e-12), linewidth=2, label=r"$|\nabla W(x)^T B|$")
    axs[1].axhline(threshold, linestyle="--", linewidth=1, label="guard threshold")
    if result.switch_time is not None:
        axs[1].axvline(result.switch_time, color="black", linestyle=":", label=r"$\tau_\gamma$")
    axs[1].set_ylabel("Magnitude")
    axs[1].grid(True)
    axs[1].legend()

    axs[2].plot(t, u, linewidth=2, label=r"$u(t)$")
    if result.switch_time is not None:
        axs[2].axvline(result.switch_time, color="black", linestyle=":", label=r"$\tau_\gamma$")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Control torque")
    axs[2].grid(True)
    axs[2].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "singularity.png", dpi=150)
    plt.close(fig)


def save_comparison_plots(
    experiment: ShipRollExperiment,
    results: dict[str, SimulationResult],
    output_dir: Path,
) -> None:
    fig, axs = plt.subplots(4, 3, figsize=(18, 18), num="Controller comparison")
    colors = {
        "hybrid_adaptive": "tab:blue",
        "local_linear": "tab:red",
        "adaptive_local_only": "tab:green",
        "hybrid_nominal": "tab:purple",
    }

    ax = axs[0, 0]
    plot_phase_geometry(ax, experiment)
    all_theta = []
    for name, result in results.items():
        valid = valid_mask(result)
        x = result.x[valid]
        all_theta.extend(x[:, 0].tolist())
        ax.plot(x[:, 0], x[:, 1], linewidth=2, color=colors[name], label=result.controller_display_name)
        ax.scatter([x[0, 0]], [x[0, 1]], marker="o", color=colors[name], s=45)
        ax.scatter([x[-1, 0]], [x[-1, 1]], marker="x", color=colors[name], s=70)
    if all_theta:
        theta_min = min(all_theta) - 0.25
        theta_max = max(all_theta) + 0.25
        plot_singular_line(ax, experiment, (theta_min, theta_max))
    ax.scatter([0.0], [0.0], marker="+", color="black", s=90, label="Target")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Roll angle $\theta$ (rad)")
    ax.set_ylabel(r"Roll rate $p$ (rad/s)")
    ax.set_title("Phase Portrait Comparison")
    ax.grid(True)
    ax.legend(fontsize=8)

    ax = axs[0, 1]
    for name, result in results.items():
        valid = valid_mask(result)
        ax.plot(result.t[valid], result.W[valid], linewidth=2, color=colors[name], label=result.controller_display_name)
    ax.axhline(experiment.levels.r_gamma, linestyle="--", color="black", linewidth=1.5, label=r"$r_\gamma$")
    ax.axhline(experiment.levels.a_loc, linestyle=":", color="black", linewidth=1.8, label=r"$a_{\rm loc}$")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$W(x)$")
    ax.set_title("Lyapunov State Energy")
    ax.grid(True)
    ax.legend(fontsize=8)

    ax = axs[0, 2]
    for name, result in results.items():
        valid = valid_mask(result)
        state_norm = np.linalg.norm(result.x[valid], axis=1)
        ax.plot(result.t[valid], state_norm, linewidth=2, color=colors[name], label=result.controller_display_name)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\|x\|$")
    ax.set_title("State Norm")
    ax.grid(True)
    ax.legend(fontsize=8)

    ax = axs[1, 0]
    for name, result in results.items():
        valid = valid_mask(result)
        ax.plot(result.t[valid], result.x[valid, 0], linewidth=2, color=colors[name], label=result.controller_display_name)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Roll angle $\theta$ (rad)")
    ax.set_title("Roll Angle")
    ax.grid(True)
    ax.legend(fontsize=8)

    ax = axs[1, 1]
    for name, result in results.items():
        valid = valid_mask(result)
        ax.plot(result.t[valid], result.u[valid], linewidth=1.7, color=colors[name], label=result.controller_display_name)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Control torque")
    ax.set_title("Control Effort")
    ax.grid(True)
    ax.legend(fontsize=8)

    ax = axs[1, 2]
    for name, result in results.items():
        valid = valid_mask(result)
        ax.step(
            result.t[valid],
            mode_indicator(result)[valid],
            where="post",
            linewidth=1.7,
            color=colors[name],
            label=result.controller_display_name,
        )
    ax.set_ylim([-0.1, 1.1])
    ax.set_yticks([0.0, 1.0], ["local/linear", "outer"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mode")
    ax.set_title("Controller Mode")
    ax.grid(True)
    ax.legend(fontsize=8)

    ax = axs[2, 0]
    reference_result = results["hybrid_adaptive"]
    valid_reference = valid_mask(reference_result)
    ax.plot(
        reference_result.t[valid_reference],
        reference_result.d_true[valid_reference],
        color="black",
        linestyle="--",
        linewidth=1.8,
        label=r"true $d(t)$",
    )
    for name in ["hybrid_adaptive", "adaptive_local_only"]:
        result = results[name]
        valid = valid_mask(result)
        ax.plot(
            result.t[valid],
            result.d_hat[valid],
            linewidth=1.8,
            color=colors[name],
            label=rf"{result.controller_display_name} $\hat d(t)$",
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Wave moment")
    ax.set_title("Wave Moment Approximation")
    ax.grid(True)
    ax.legend(fontsize=8)

    ax = axs[2, 1]
    for name, result in results.items():
        valid = valid_mask(result)
        ax.plot(
            result.t[valid],
            result.disturbance_error[valid],
            linewidth=1.7,
            color=colors[name],
            label=result.controller_display_name,
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$d(t)-\hat d(t)$")
    ax.set_title("Disturbance Tracking Error")
    ax.grid(True)
    ax.legend(fontsize=8)

    ax = axs[2, 2]
    for name in ["hybrid_adaptive", "adaptive_local_only"]:
        result = results[name]
        valid = valid_mask(result)
        ax.plot(
            result.t[valid],
            result.theta_tilde_norm[valid],
            linewidth=1.8,
            color=colors[name],
            label=result.controller_display_name,
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\|\tilde\theta\|$")
    ax.set_title("Parameter Error Norm")
    ax.grid(True)
    ax.legend(fontsize=8)

    ax = axs[3, 0]
    for name, result in results.items():
        valid = valid_mask(result)
        ax.plot(result.t[valid], result.V_gamma[valid], linewidth=2, color=colors[name], label=result.controller_display_name)
    ax.axhline(experiment.levels.a_bar, linestyle="--", color="black", linewidth=1.5, label=r"$\bar a$")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$V_\gamma$")
    ax.set_title("Extended Lyapunov Quantity")
    ax.grid(True)
    ax.legend(fontsize=8)

    ax = axs[3, 1]
    for name, result in results.items():
        valid = valid_mask(result)
        ax.plot(
            result.t[valid],
            np.abs(result.input_gradient[valid]),
            linewidth=1.7,
            color=colors[name],
            label=result.controller_display_name,
        )
    ax.axhline(experiment.config.outer_g_min, linestyle="--", color="black", linewidth=1.2, label="guard threshold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$|\nabla W(x)^T B|$")
    ax.set_title("Singular Denominator Distance")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend(fontsize=8)

    ax = axs[3, 2]
    names = list(results.keys())
    tail_rms_theta = []
    tail_rms_error = []
    labels = []
    for name in names:
        result = results[name]
        valid = valid_mask(result)
        tail = valid & (result.t >= result.t[-1] - 10.0)
        tail_rms_theta.append(np.sqrt(np.nanmean(result.x[tail, 0] ** 2)))
        tail_rms_error.append(np.sqrt(np.nanmean(result.disturbance_error[tail] ** 2)))
        labels.append(result.controller_display_name)
    x_pos = np.arange(len(names))
    width = 0.36
    ax.bar(x_pos - width / 2.0, tail_rms_theta, width, label=r"Tail RMS $\theta$ (rad)")
    ax.bar(x_pos + width / 2.0, tail_rms_error, width, label="Tail RMS disturbance error")
    ax.set_xticks(x_pos, labels, rotation=20, ha="right")
    ax.set_title("Tail Performance Metrics")
    ax.grid(True, axis="y")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "plots.png", dpi=150)
    plt.close(fig)
