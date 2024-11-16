import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
import mlflow
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support


logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, model: nn.Module, config: Dict):
        """Initialize the trainer with a pre-configured model instance"""
        self.model = model
        self.config = config
        self.device = torch.device(config["training"]["device"])
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config["model"]["learning_rate"],
            weight_decay=config["model"]["weight_decay"],
        )

        mlflow.set_tracking_uri(config["monitoring"]["mlflow_tracking_uri"])
        # Create or get existing experiment
        experiment_name = config["monitoring"].get("experiment_name", "default")
        try:
            mlflow.create_experiment(experiment_name)
        except Exception:
            pass  # Experiment already exists
        mlflow.set_experiment(experiment_name)

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        # Add position=0, leave=True to fix progress bar
        pbar = tqdm(train_loader, desc="Training", position=0, leave=True)
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            if batch_idx % self.config["monitoring"]["log_interval"] == 0:
                mlflow.log_metrics(
                    {
                        "batch_loss": loss.item(),
                        "batch_accuracy": 100.0 * correct / total,
                    },
                    step=batch_idx,
                )

            # Update progress bar with formatted strings
            pbar.set_postfix_str(
                f"Loss: {total_loss/(batch_idx+1):.3f}, Accuracy: {100.0*correct/total:.2f}%"
            )

        epoch_loss = total_loss / len(train_loader)
        epoch_accuracy = 100.0 * correct / total
        return epoch_loss, epoch_accuracy

    def evaluate(self, val_loader: DataLoader) -> Tuple[float, float, Dict]:
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, target in tqdm(
                val_loader, desc="Evaluating", position=0, leave=True
            ):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                val_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        # Calculate additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average="weighted", zero_division=0
        )

        metrics = {
            "val_loss": val_loss / len(val_loader),
            "val_accuracy": 100.0 * correct / total,
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": f1,
        }

        return metrics["val_loss"], metrics["val_accuracy"], metrics

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        best_val_loss = float("inf")
        patience_counter = 0

        with mlflow.start_run():
            # Log model parameters
            mlflow.log_params(
                {
                    "architecture": self.config["model"]["architecture"],
                    "learning_rate": self.config["model"]["learning_rate"],
                    "batch_size": self.config["data"]["batch_size"],
                    "epochs": self.config["model"]["epochs"],
                }
            )

            for epoch in range(self.config["model"]["epochs"]):
                logger.info(f"Epoch {epoch+1}/{self.config['model']['epochs']}")

                # Training phase
                train_loss, train_accuracy = self.train_epoch(train_loader)

                # Validation phase
                val_loss, val_accuracy, metrics = self.evaluate(val_loader)

                # Log metrics
                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "train_accuracy": train_accuracy,
                        "val_loss": val_loss,
                        "val_accuracy": val_accuracy,
                        "val_precision": metrics["val_precision"],
                        "val_recall": metrics["val_recall"],
                        "val_f1": metrics["val_f1"],
                    },
                    step=epoch,
                )

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.model.save_model(
                        f"{self.config['training']['checkpoint_dir']}/best_model.pth"
                    )
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Early stopping
                if (
                    patience_counter
                    >= self.config["training"]["early_stopping_patience"]
                ):
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break

            # Log final model
            mlflow.pytorch.log_model(self.model, "model")
