# Font Classification MLOps Project

A production-ready MLOps pipeline for font classification using deep learning, built with PyTorch and MLflow.

## Overview

This project implements an end-to-end MLOps pipeline for classifying fonts in images. It uses a ResNet18-based architecture and includes comprehensive monitoring, logging, and containerized deployment.

## Features

- **Data Pipeline**
  - Automated data preprocessing and augmentation
  - Train/validation/test split management
  - Real-time data statistics logging
  - Support for multiple image formats

- **Model Architecture**
  - ResNet18-based classifier (pretrained on ImageNet)
  - Configurable architecture through YAML
  - Support for transfer learning
  - Early stopping and model checkpointing

- **Training Pipeline**
  - Distributed training support
  - Comprehensive metric logging
  - MLflow experiment tracking
  - Early stopping with configurable patience
  - Automated hyperparameter logging

- **Monitoring & Logging**
  - Real-time inference monitoring
  - Performance metrics tracking
  - MLflow dashboard integration
  - Detailed logging of training and inference

- **API Service**
  - FastAPI-based REST API
  - Async inference support
  - Swagger documentation
  - Production-ready containerization

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

2. Set up environment (choose one):

Using pip:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Using conda:
```bash
conda env create -f environment.yml
conda activate mlops_env
```

3. Using Docker:
```bash
docker compose up -d
```

## Usage

### Training

1. Configure training parameters in `src/config/config.yaml`
2. Start training:
```bash
python main.py --mode train
```

### API Service

1. Start the API server:
```bash
python main.py --mode api
```

2. Make predictions:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/image.jpg"
```

### Monitoring

1. Access MLflow UI:
```bash
mlflow ui --port 5000
```
Visit http://localhost:5000 to view experiments

2. Start monitoring service:
```bash
python main.py --mode monitor
```

## Testing

Run the test suite:
```bash
python -m pytest src/tests/
```

## Configuration

Key configuration parameters in `src/config/config.yaml`:

- Data processing: image size, batch size, data splits
- Model: architecture, learning rate, weight decay
- Training: device, checkpointing, early stopping
- API: host, port, model path
- Monitoring: MLflow settings, logging intervals

## Docker Services

- `mlflow`: MLflow tracking server
- `training`: Model training service
- `api`: Inference API service
- `monitoring`: Metrics monitoring service

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Author

Abir Hassan

