"""CLI entry point for U-MIMIC experiments."""

from __future__ import annotations

import argparse
import json
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
    from umimic import __version__

    parser = argparse.ArgumentParser(
        description="U-MIMIC: Unified Mechanistic Inference from Multimodal Imaging and Counts",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"umimic {__version__}",
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

    # NOTE: the `dashboard` command was removed in this release. No dashboard
    # application ships with the package, so the command could only ever fail;
    # advertising it (and a [dashboard] extra) implied a feature that did not
    # exist. It will return if and when an app is actually included.

    # Config utility command
    config_parser = subparsers.add_parser("config", help="Configuration utilities")
    config_subparsers = config_parser.add_subparsers(
        dest="config_command", help="Config utility commands"
    )
    cfg_ranges_parser = config_subparsers.add_parser(
        "suggest-coupling-ranges",
        help="Print recommended coupling parameter ranges",
    )
    cfg_ranges_parser.add_argument(
        "--format",
        dest="output_format",
        choices=["json", "yaml"],
        default="json",
        help="Output format for suggested ranges.",
    )
    _add_logging_args(cfg_ranges_parser)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)
    if args.command == "config" and getattr(args, "config_command", None) is None:
        config_parser.print_help()
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
        elif args.command == "config":
            _run_config(args)
        LOGGER.info("Completed '%s' command successfully", args.command)
    except FileNotFoundError as exc:
        LOGGER.error("File not found: %s", exc)
        sys.exit(1)
    except (ValueError, TypeError) as exc:
        LOGGER.error("Invalid input: %s", exc)
        sys.exit(1)
    except ImportError as exc:
        # Only third-party packages are optional extras. A missing umimic.*
        # module is a broken installation, not something an extra can fix, and
        # must not be reported as one.
        name = getattr(exc, "name", "") or ""
        if name.startswith("umimic"):
            LOGGER.error(
                "Internal module %s could not be imported. This indicates a "
                "corrupt or incomplete umimic installation, not a missing "
                "optional dependency. Reinstall the package. (%s)",
                name,
                exc,
            )
        else:
            LOGGER.error(
                "Missing optional dependency: %s. Install optional extras "
                "with: pip install umimic[all]",
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

    builders = {
        "cytotoxic": RateSet.cytotoxic_drug,
        "cytostatic": RateSet.cytostatic_drug,
        "mixed": RateSet.mixed_drug,
    }
    rs = builders[args.drug_type]()

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

    from umimic.data.loaders import save_dataset

    exp = Experiment(config)
    dataset = exp.generate_synthetic()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Using output directory %s", out_dir)

    # Write the dataset in the documented interchange format so that
    # `umimic fit --data <this file>` reads it back without conversion.
    data_path = save_dataset(dataset, out_dir / "dataset.csv")

    # Record the configuration and seed alongside the data for reproducibility.
    from umimic.pipeline.config import save_config

    config_path = out_dir / "config.yaml"
    save_config(config, config_path)

    print(f"Generated {dataset.n_series} synthetic time series")
    print(f"Concentrations: {dataset.concentrations}")
    print(f"Dataset written to {data_path}")
    print(f"Configuration written to {config_path}")
    LOGGER.info(
        "Generated synthetic dataset with %d series at concentrations=%s -> %s",
        dataset.n_series,
        dataset.concentrations,
        data_path,
    )


def _run_config(args):
    """Run configuration utility commands."""
    from umimic.pipeline.config import CouplingConfig

    if args.config_command == "suggest-coupling-ranges":
        payload = CouplingConfig.recommended_parameter_ranges()
        if args.output_format == "yaml":
            import yaml

            yaml_payload = json.loads(json.dumps(payload))
            print(yaml.safe_dump(yaml_payload, sort_keys=True))
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
        LOGGER.info("Printed recommended coupling parameter ranges")
        return

    raise ValueError(f"Unknown config command: {args.config_command}")


if __name__ == "__main__":
    main()
