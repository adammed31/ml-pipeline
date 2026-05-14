import argparse
import logging
import sys

import pandas as pd

from src.training import ModelTrainer
from src.utils import load_config, setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train supervised ML models with MLflow tracking.")
    parser.add_argument("--data", required=True, help="Path to the CSV dataset")
    parser.add_argument("--target", required=True, help="Name of the target column")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--experiment", default="ml_experiment")
    parser.add_argument("--model-dir", default="models")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("Loading dataset from %s", args.data)
    df = pd.read_csv(args.data)
    logger.info("Dataset shape: %s", df.shape)

    trainer = ModelTrainer(experiment_name=args.experiment)
    results, best_model, preprocessor = trainer.train(df, args.target, test_size=args.test_size)

    print("\nResults:")
    print(results.to_string(index=False))
    print(f"\nBest model: {type(best_model).__name__}")

    trainer.save(args.model_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
