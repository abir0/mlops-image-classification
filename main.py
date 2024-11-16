import logging
import os
import yaml
import argparse
import time

from src.data.data_loader import get_data_loaders
from src.models.classifier import FontClassifier
from src.training.trainer import ModelTrainer
from src.monitoring.metrics import MetricsLogger, logger


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/training.log"), logging.StreamHandler()],
)


def train(config):
    # Initialize data loaders
    train_loader, val_loader, test_loader = get_data_loaders(config)

    # Initialize model
    model = FontClassifier(config)

    # Initialize trainer
    trainer = ModelTrainer(model, config)

    # Train model
    trainer.train(train_loader, val_loader)

    # Evaluate model on test set
    test_loss, test_accuracy, test_metrics = trainer.evaluate(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Test F1 Score: {test_metrics['val_f1']:.4f}")


def monitor(config):
    # Initialize metrics logger
    metrics_logger = MetricsLogger()

    while True:
        try:
            # Get summary statistics for the last hour (3600 seconds)
            stats = metrics_logger.get_summary_statistics(time_window=3600)

            if stats:
                logger.info("Monitoring Statistics (Last Hour):")
                logger.info(f"Total Inferences: {stats['count']}")
                logger.info(
                    f"Average Inference Time: {stats['avg_inference_time']:.4f}s"
                )
                logger.info(f"Average Confidence: {stats['avg_confidence']:.4f}")

                if "accuracy" in stats:
                    logger.info(f"Accuracy: {stats['accuracy']:.4f}")

            # Sleep for monitoring interval (e.g., every 5 minutes)
            time.sleep(config.get("monitoring", {}).get("interval", 300))

        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in monitoring: {str(e)}")
            time.sleep(60)  # Wait a minute before retrying


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "monitor"],
        help="Mode to run the script in: train or monitor",
    )
    args = parser.parse_args()

    # Load configuration
    with open("src/config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Create necessary directories
    os.makedirs(config["logging"]["log_dir"], exist_ok=True)
    os.makedirs(config["training"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["monitoring"]["mlflow_tracking_uri"], exist_ok=True)

    # Run the appropriate mode
    if args.mode == "train":
        train(config)
    elif args.mode == "monitor":
        monitor(config)


if __name__ == "__main__":
    main()
