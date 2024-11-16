import os
from PIL import Image

import numpy as np
import pytest
import torch
import torch.nn as nn

from models.classifier import FontClassifier
from data.data_loader import FontDataset, get_data_loaders


@pytest.fixture
def config():
    return {
        "model": {
            "architecture": "resnet18",
            "num_classes": 50,
            "pretrained": True,
            "learning_rate": 0.001,
        },
        "data": {"img_size": 224, "batch_size": 32, "num_workers": 4},
        "training": {"device": "cuda" if torch.cuda.is_available() else "cpu"},
    }


@pytest.fixture
def model(config):
    return FontClassifier(config)


def test_model_structure(model):
    assert isinstance(model, nn.Module)
    assert hasattr(model, "forward")
    assert callable(model.forward)


def test_model_output_shape(model, config):
    batch_size = 4
    input_tensor = torch.randn(
        batch_size, 3, config["data"]["img_size"], config["data"]["img_size"]
    )
    output = model(input_tensor)
    assert output.shape == (batch_size, config["model"]["num_classes"])


def test_model_save_load(model, config, tmp_path):
    # Save model
    save_path = os.path.join(tmp_path, "test_model.pth")
    model.save_model(save_path)
    assert os.path.exists(save_path)

    # Load model
    loaded_model = FontClassifier.load_model(save_path, device="cpu")
    assert isinstance(loaded_model, FontClassifier)

    # Compare outputs
    input_tensor = torch.randn(
        1, 3, config["data"]["img_size"], config["data"]["img_size"]
    )
    with torch.no_grad():
        original_output = model(input_tensor)
        loaded_output = loaded_model(input_tensor)

    assert torch.allclose(original_output, loaded_output)


def test_dataset(tmp_path):
    # Create mock dataset structure
    class_names = ["font1", "font2"]
    for class_name in class_names:
        os.makedirs(os.path.join(tmp_path, class_name), exist_ok=True)
        # Create dummy image
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img.save(os.path.join(tmp_path, class_name, "test.png"))

    dataset = FontDataset(str(tmp_path))
    assert len(dataset) == len(class_names)

    # Test __getitem__
    img, label = dataset[0]
    assert isinstance(img, Image.Image)
    assert isinstance(label, int)
    assert 0 <= label < len(class_names)


def test_data_loaders(config, tmp_path):
    # Create mock dataset
    for split in ["train", "val", "test"]:
        split_path = os.path.join(tmp_path, split)
        os.makedirs(split_path, exist_ok=True)
        for i in range(2):
            class_path = os.path.join(split_path, f"font{i}")
            os.makedirs(class_path, exist_ok=True)
            img = Image.fromarray(
                np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            )
            img.save(os.path.join(class_path, "test.png"))

    config["data"].update(
        {
            "train_path": os.path.join(tmp_path, "train"),
            "val_path": os.path.join(tmp_path, "val"),
            "test_path": os.path.join(tmp_path, "test"),
        }
    )

    train_loader, val_loader, test_loader = get_data_loaders(config)
    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0
