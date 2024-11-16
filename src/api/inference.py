import io
import time
import logging
from typing import Dict
import yaml

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import torch
import torchvision.transforms as transforms

from src.models.classifier import FontClassifier
from src.monitoring.metrics import MetricsLogger


logger = logging.getLogger(__name__)

app = FastAPI()


class InferenceService:
    def __init__(self, config: Dict):
        try:
            self.config = config
            self.device = torch.device(config["training"]["device"])
            self.model = FontClassifier.load_model(
                config["api"]["model_path"], device=self.device
            )
            self.model.eval()

            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        (config["data"]["img_size"], config["data"]["img_size"])
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            self.metrics_logger = MetricsLogger()
            logger.info("InferenceService initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing InferenceService: {str(e)}")
            raise

    async def predict(self, image: Image.Image) -> Dict:
        try:
            start_time = time.time()

            # Transform image
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Make prediction
            with torch.no_grad():
                output = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][prediction].item()

            # Log metrics
            inference_time = time.time() - start_time
            self.metrics_logger.log_inference(
                inference_time=inference_time, confidence=confidence
            )

            return {
                "predicted_class": prediction,
                "confidence": confidence,
                "inference_time": inference_time,
            }

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))


inference_service = None


@app.on_event("startup")
async def startup_event():
    global inference_service
    # Load config from YAML file
    with open("src/config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    inference_service = InferenceService(config)
    logger.info("Inference service initialized")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Make prediction
        result = await inference_service.predict(image)
        return result

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
