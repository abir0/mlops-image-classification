from datetime import datetime
import json
import os
import logging
import time
from typing import Dict, Optional

import mlflow


logger = logging.getLogger(__name__)


class MetricsLogger:
    def __init__(self):
        # Set MLflow tracking URI
        mlflow.set_tracking_uri("mlruns")

        # Create or get experiment
        experiment_name = "font_classification"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)

        # Set the experiment
        mlflow.set_experiment(experiment_name)

        logger.info(f"Initialized metrics logging with experiment: {experiment_name}")

        self.metrics_file = "metrics/inference_metrics.jsonl"
        os.makedirs("metrics", exist_ok=True)

        # Initialize running statistics
        self.total_inferences = 0
        self.total_time = 0
        self.total_confidence = 0
        self.last_reset_time = time.time()

    def log_inference(
        self,
        inference_time: float,
        confidence: float,
        prediction: Optional[int] = None,
        actual: Optional[int] = None,
    ):
        """Log a single inference event with metrics."""
        timestamp = datetime.now().isoformat()

        metrics = {
            "timestamp": timestamp,
            "inference_time": inference_time,
            "confidence": confidence,
        }

        if prediction is not None and actual is not None:
            metrics.update(
                {
                    "prediction": prediction,
                    "actual": actual,
                    "correct": prediction == actual,
                }
            )

        # Update running statistics
        self.total_inferences += 1
        self.total_time += inference_time
        self.total_confidence += confidence

        # Log to file
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        # Log to MLflow
        with mlflow.start_run(run_name="inference", nested=True):
            mlflow.log_metric("inference_time", inference_time)
            mlflow.log_metric("confidence", confidence)

    def get_summary_statistics(self, time_window: Optional[float] = None) -> Dict:
        """Get summary statistics for inferences within the specified time window."""
        current_time = time.time()

        metrics = []
        with open(self.metrics_file, "r") as f:
            for line in f:
                metric = json.loads(line)
                timestamp = datetime.fromisoformat(metric["timestamp"]).timestamp()

                if time_window is None or (current_time - timestamp) <= time_window:
                    metrics.append(metric)

        if not metrics:
            return {}

        inference_times = [m["inference_time"] for m in metrics]
        confidences = [m["confidence"] for m in metrics]

        summary = {
            "count": len(metrics),
            "avg_inference_time": sum(inference_times) / len(inference_times),
            "max_inference_time": max(inference_times),
            "min_inference_time": min(inference_times),
            "avg_confidence": sum(confidences) / len(confidences),
        }

        # Add accuracy if available
        correct_predictions = [m for m in metrics if m.get("correct", False)]
        if correct_predictions:
            summary["accuracy"] = len(correct_predictions) / len(metrics)

        return summary

    def reset_statistics(self):
        """Reset running statistics."""
        self.total_inferences = 0
        self.total_time = 0
        self.total_confidence = 0
        self.last_reset_time = time.time()
