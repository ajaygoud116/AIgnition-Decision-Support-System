"""CLI entry point for the forecasting pipeline."""
import argparse
from pathlib import Path

from src.pipeline.runner import PipelineRunner
from src.utils.config import Config


def main() -> None:
    parser = argparse.ArgumentParser(description="AIgnition Forecasting Pipeline")
    parser.add_argument(
        "--data-dir", default="data",
        help="Directory containing input CSV files (default: data)",
    )
    parser.add_argument(
        "--model-path", default="pickle/model.pkl",
        help="Path to save/load the pickled model (default: pickle/model.pkl)",
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="Directory for output reports (default: output)",
    )
    parser.add_argument(
        "--force-retrain", action="store_true",
        help="Force model retraining even if pickle exists",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to config YAML file (default: config.yaml)",
    )
    args = parser.parse_args()

    config = Config(Path(args.config) if args.config else Path("config.yaml"))
    runner = PipelineRunner(config)

    runner.run(
        data_dir=Path(args.data_dir),
        model_path=Path(args.model_path),
        output_dir=Path(args.output_dir),
        force_retrain=args.force_retrain,
    )


if __name__ == "__main__":
    main()
