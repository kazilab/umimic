"""CLI entry point for U-MIMIC experiments."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, UTC
from pathlib import Path

LOGGER = logging.getLogger("umimic.cli")


def _add_logging_args(parser: argparse.ArgumentParser) -> None:
    """Attach logging-related CLI args to a sub-command parser."""
    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to write command run logs. Defaults to a timestamped file.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity level.",
    )


def _default_log_path(args: argparse.Namespace) -> Path:
    """Resolve default command log path."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    if hasattr(args, "output"):
        out_dir = Path(args.output)
    else:
        out_dir = Path("results") / "logs"

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{args.command}_{stamp}.log"


def _configure_logging(args: argparse.Namespace) -> Path:
    """Configure stdout + file logging and return log file path.

    Uses a package-specific logger ("umimic") instead of the root logger
    so that library consumers' logging configuration is not overwritten.
    """
    log_path = Path(args.log_file) if args.log_file else _default_log_path(args)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, args.log_level.upper(), logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")

    pkg_logger = logging.getLogger("umimic")
    pkg_logger.setLevel(level)
    # Remove any previously attached handlers to avoid duplicate output
    pkg_logger.handlers.clear()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    pkg_logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    pkg_logger.addHandler(file_handler)

    # Prevent log messages from propagating to the root logger
    pkg_logger.propagate = False

    return log_path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="U-MIMIC: Unified Mechanistic Inference from Multimodal Imaging and Counts",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Simulate command
    sim_parser = subparsers.add_parser("simulate", help="Run forward simulation")
    sim_parser.add_argument("--config", type=str, help="Path to YAML config")
    sim_parser.add_argument("--output", type=str, default="results/simulation",
                           help="Output directory")
    sim_parser.add_argument("--drug-type", type=str, default="cytotoxic",
                           choices=["cytotoxic", "cytostatic", "mixed"],
                           help="Drug mechanism type")
    _add_logging_args(sim_parser)

    # Fit command
    fit_parser = subparsers.add_parser("fit", help="Fit model to data")
    fit_parser.add_argument("--config", type=str, required=True, help="Config path")
    fit_parser.add_argument("--data", type=str, help="Data file path")
    fit_parser.add_argument("--output", type=str, default="results/inference",
                           help="Output directory")
    _add_logging_args(fit_parser)

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate synthetic data")
    gen_parser.add_argument("--config", type=str, help="Config path")
    gen_parser.add_argument("--output", type=str, default="results/synthetic",
                           help="Output directory")
    _add_logging_args(gen_parser)

    # Dashboard command
    dash_parser = subparsers.add_parser("dashboard", help="Launch Streamlit dashboard")
    dash_parser.add_argument("--port", type=int, default=8501)
    _add_logging_args(dash_parser)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    log_path = _configure_logging(args)
    LOGGER.info("Starting '%s' command", args.command)
    LOGGER.info("Run arguments: %s", vars(args))
    LOGGER.info("Log file: %s", log_path)

    try:
        if args.command == "simulate":
            _run_simulate(args)
        elif args.command == "fit":
            _run_fit(args)
        elif args.command == "generate":
            _run_generate(args)
        elif args.command == "dashboard":
            _run_dashboard(args)
        LOGGER.info("Completed '%s' command successfully", args.command)
    except FileNotFoundError as exc:
        LOGGER.error("File not found: %s", exc)
        sys.exit(1)
    except (ValueError, TypeError) as exc:
        LOGGER.error("Invalid input: %s", exc)
        sys.exit(1)
    except ImportError as exc:
        LOGGER.error(
            "Missing dependency: %s. Install optional extras with: "
            "pip install umimic[all]",
            exc,
        )
        sys.exit(1)
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user")
        sys.exit(130)
    except Exception:
        LOGGER.exception("Command '%s' failed with an unexpected error", args.command)
        raise


def _run_simulate(args):
    """Run simulation command."""
    from umimic.pipeline.config import load_config, ExperimentConfig
    from umimic.pipeline.experiment import Experiment
    from umimic.dynamics.rates import RateSet

    if args.config:
        config = load_config(args.config)
        LOGGER.info("Loaded config from %s", args.config)
    else:
        config = ExperimentConfig()
        LOGGER.info("Using default ExperimentConfig")

    exp = Experiment(config)

    if args.drug_type == "cytotoxic":
        rs = RateSet.cytotoxic_drug()
    elif args.drug_type == "cytostatic":
        rs = RateSet.cytostatic_drug()
    else:
        rs = RateSet.cytotoxic_drug()

    concentrations = config.dosing.concentrations or [0, 0.1, 0.3, 1, 3, 10, 30]
    results = exp.simulate(rate_set=rs, method="ode", concentrations=concentrations)
    LOGGER.info("Simulated %d concentration conditions", len(results))

    # Save plots
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Saving simulation artifacts to %s", out_dir)

    from umimic.visualization.trajectories import plot_dose_response_trajectories
    from umimic.visualization.dose_response import plot_rate_dose_response, plot_net_growth_curve

    fig = plot_dose_response_trajectories(results, title=f"Dose-Response ({args.drug_type})")
    fig.savefig(out_dir / "trajectories.png", dpi=150, bbox_inches="tight")

    fig2 = plot_rate_dose_response(rs)
    fig2.savefig(out_dir / "rate_dose_response.png", dpi=150, bbox_inches="tight")

    fig3 = plot_net_growth_curve(rs)
    fig3.savefig(out_dir / "net_growth.png", dpi=150, bbox_inches="tight")

    print(f"Simulation results saved to {out_dir}")
    LOGGER.info("Saved simulation plots to %s", out_dir)


def _run_fit(args):
    """Run inference command."""
    from umimic.pipeline.config import load_config
    from umimic.pipeline.experiment import Experiment
    from umimic.pipeline.results import save_result

    config = load_config(args.config)
    exp = Experiment(config)
    LOGGER.info("Loaded config from %s", args.config)

    # Generate or load data
    if args.data:
        from umimic.data.loaders import load_csv
        dataset = load_csv(args.data, config.data)
        LOGGER.info("Loaded dataset from %s with %d series", args.data, dataset.n_series)
    else:
        dataset = exp.generate_synthetic()
        LOGGER.info("Generated synthetic dataset with %d series", dataset.n_series)

    result = exp.fit(dataset)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_result(result, out_dir / "result.json")
    LOGGER.info("Saved inference result to %s", out_dir / "result.json")

    print(f"Inference complete. Results saved to {out_dir / 'result.json'}")
    print(f"Point estimates: {result.point_estimates}")
    LOGGER.info("Inference point estimates: %s", result.point_estimates)


def _run_generate(args):
    """Run synthetic data generation command."""
    from umimic.pipeline.config import load_config, ExperimentConfig
    from umimic.pipeline.experiment import Experiment

    if args.config:
        config = load_config(args.config)
        LOGGER.info("Loaded config from %s", args.config)
    else:
        config = ExperimentConfig()
        LOGGER.info("Using default ExperimentConfig")

    exp = Experiment(config)
    dataset = exp.generate_synthetic()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Using output directory %s", out_dir)

    print(f"Generated {dataset.n_series} synthetic time series")
    print(f"Concentrations: {dataset.concentrations}")
    LOGGER.info(
        "Generated synthetic dataset with %d series at concentrations=%s",
        dataset.n_series,
        dataset.concentrations,
    )


def _run_dashboard(args):
    """Launch Streamlit dashboard."""
    import subprocess
    import shutil

    # Look for dashboard relative to the package, then fall back to CWD
    candidates = [
        Path(__file__).resolve().parent.parent / "dashboard" / "app.py",
        Path(__file__).resolve().parent.parent.parent / "dashboard" / "app.py",
        Path.cwd() / "dashboard" / "app.py",
    ]
    dashboard_path = next((p for p in candidates if p.exists()), None)

    if dashboard_path is None:
        searched = "\n  ".join(str(p) for p in candidates)
        LOGGER.error(
            "Dashboard app.py not found. Searched:\n  %s\n"
            "Create a dashboard/app.py or pass a path via --app-path.",
            searched,
        )
        sys.exit(1)

    if shutil.which("streamlit") is None:
        LOGGER.error(
            "Streamlit is not installed. Install it with: pip install umimic[dashboard]"
        )
        sys.exit(1)

    LOGGER.info("Launching dashboard from %s on port %d", dashboard_path, args.port)
    subprocess.run(
        ["streamlit", "run", str(dashboard_path), "--server.port", str(args.port)],
    )


if __name__ == "__main__":
    main()
