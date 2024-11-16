# Font Classification MLOps Project

This project implements an end-to-end MLOps pipeline for font classification using deep learning.

## Project Structure

```
mlops-image-classification/
├── src/
│   ├── config/
│   │   └── config.yaml
│   ├── data/
│   │   ├── data_loader.py
│   │   └── preprocessor.py
│   ├── models/
│   │   └── classifier.py
│   ├── training/
│   │   └── trainer.py
│   ├── monitoring/
│   │   └── metrics.py
│   ├── utils/
│   │   └── logger.py
│   ├── api/
│   │   └── inference.py
│   └── tests/
│       └── test_model.py
├── dataset/
├── notebooks/
├── scripts/
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── README.md
└── main.py
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mlops-image-classification.git
cd mlops-image-classification
```

2. Build and run with Docker Compose:
```bash
docker compose up --d
```

## Components

- **Data Pipeline**: Handles data loading, preprocessing, and augmentation
- **Model**: ResNet50-based classifier for font recognition
- **Training**: Implements training loop with monitoring and logging
- **API**: FastAPI-based inference service
- **Monitoring**: MLflow tracking and metrics monitoring
- **Testing**: Unit tests for all components

## API Usage

Make predictions using the API:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/image.jpg"
```

## Monitoring

- MLflow UI: http://localhost:5000

## Testing

Run tests:
```bash
python -m pytest src/tests/
```

