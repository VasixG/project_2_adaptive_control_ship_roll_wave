from __future__ import annotations

from experiments.ship_roll_setup import (
    build_experiment,
    local_decay_diagnostic,
    run_all_controllers,
    run_controller,
)
from visualization import (
    make_figure_dir,
    print_comparison_summary,
    print_design_summary,
    save_comparison_plots,
    save_hybrid_adaptive_plots,
    save_singularity_diagnostic,
)


def hybrid_main() -> None:
    experiment = build_experiment()
    result = run_controller(experiment, "hybrid_adaptive")
    print_design_summary(experiment, result)
    diagnostic = local_decay_diagnostic(experiment)
    print("Local nominal Lyapunov decay diagnostic")
    for key, value in diagnostic.items():
        print(f"  {key}={value}")

    hybrid_dir = make_figure_dir("hybrid_adaptive_control")
    diagnostic_dir = make_figure_dir("diagnostics")
    save_hybrid_adaptive_plots(experiment, result, hybrid_dir)
    save_singularity_diagnostic(experiment, result, diagnostic_dir)
    print(f"Saved main plots to {hybrid_dir / 'plots.png'}")
    print(f"Saved singularity diagnostic to {diagnostic_dir / 'singularity.png'}")


def comparison_main() -> None:
    experiment = build_experiment()
    results = run_all_controllers(experiment)
    print_comparison_summary(results)
    comparison_dir = make_figure_dir("comparison")
    save_comparison_plots(experiment, results, comparison_dir)
    print(f"Saved comparison plots to {comparison_dir / 'plots.png'}")


def all_main() -> None:
    hybrid_main()
    comparison_main()

