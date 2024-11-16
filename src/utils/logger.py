import logging
import sys
from datetime import datetime
import os
import json


class CustomLogger:
    def __init__(self, config):
        self.config = config
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = self.config["logging"]["log_dir"]
        os.makedirs(log_dir, exist_ok=True)

        # Create timestamp for log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"training_{timestamp}.log")

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info("Logging initialized")

    def log_hyperparameters(self, hyperparams):
        """Log hyperparameters"""
        self.logger.info(f"Hyperparameters: {json.dumps(hyperparams, indent=2)}")

    def log_metrics(self, metrics, step=None):
        """Log metrics"""
        metrics_str = json.dumps(metrics, indent=2)
        if step is not None:
            self.logger.info(f"Step {step} metrics: {metrics_str}")
        else:
            self.logger.info(f"Metrics: {metrics_str}")

    def log_model_summary(self, model):
        """Log model summary"""
        model.summary(print_fn=self.logger.info)

    def log_error(self, error):
        """Log error"""
        self.logger.error(f"Error occurred: {str(error)}")

    def log_data_stats(self, stats):
        """Log data statistics"""
        self.logger.info(f"Data statistics: {json.dumps(stats, indent=2)}")
