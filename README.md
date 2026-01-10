# 🐟 Smart Vision-Based Fish Biomass Estimation System

A comprehensive deep learning system for automated fish detection, segmentation, and biomass estimation using state-of-the-art CNN architectures.

## 🌟 Features

- **Multiple Detection Models**: YOLOv8, Faster R-CNN, EfficientDet
- **Advanced Segmentation**: U-Net, Mask R-CNN, DeepLabV3+
- **Biomass Estimation**: CNN-based regression with multiple approaches
- **Real-time Processing**: Video and webcam support with fish tracking
- **Transfer Learning**: Pre-trained weights for faster convergence
- **Comprehensive Pipeline**: From data preprocessing to inference
- **Visualization Tools**: Interactive results display and analysis

## 📋 Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference](#inference)
- [Model Architecture](#model-architecture)
- [Configuration](#configuration)
- [API Usage](#api-usage)
- [Results](#results)

## 🚀 Installation

### Requirements

- Python 3.8+
- CUDA 11.0+ (for GPU support)
- 8GB+ RAM (16GB recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/SuvRules/Fish-biomass-estimation.git
cd Fish-biomass-estimation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📊 Dataset Preparation

### Dataset Structure

```
data/
├── raw/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── videos/
├── annotations/
│   ├── train.json  # COCO format
│   ├── val.json
│   └── test.json
└── processed/
    └── augmented/
```

### Create Dummy Dataset for Testing

```python
from src.dataset import create_dummy_dataset

# Create dummy training data
create_dummy_dataset('data/train', num_samples=100)
create_dummy_dataset('data/val', num_samples=20)
```

## 🎓 Training

### Quick Start - Train CNN Models

```bash
# Create dummy dataset first
python -c "from src.dataset import create_dummy_dataset; create_dummy_dataset('data/train', 100); create_dummy_dataset('data/val', 20)"

# Train biomass estimation CNN
python src/train_cnn.py --config configs/cnn_config.yaml --model biomass_cnn

# Train segmentation CNN
python src/train_cnn.py --config configs/cnn_config.yaml --model segmentation_cnn

# Train detection CNN
python src/train_cnn.py --config configs/cnn_config.yaml --model detection_cnn

# Train multi-task CNN
python src/train_cnn.py --config configs/cnn_config.yaml --model multitask_cnn
```

### Using the Example Script

```bash
python train_example.py
```

## 🔮 Inference

### Test Models

```python
import torch
from src.models.cnn_models import FishBiomassCNN, FishSegmentationCNN

# Load biomass estimation model
model = FishBiomassCNN(input_channels=3, output_features=3)
model.eval()

# Test with random input
x = torch.randn(1, 3, 224, 224)
output = model(x)
print(f"Predicted [length, width, weight]: {output}")

# Load segmentation model
seg_model = FishSegmentationCNN(input_channels=3, num_classes=1)
seg_output = seg_model(x)
print(f"Segmentation output shape: {seg_output.shape}")
```

## 🏗️ Model Architecture

### Available CNN Models

#### 1. FishDetectionCNN
Custom CNN for fish detection and localization with bounding boxes.

```python
from src.models.cnn_models import FishDetectionCNN

model = FishDetectionCNN(num_classes=10, input_channels=3)
```

**Architecture:**
- Encoder with 5 convolutional blocks
- Detection head with bbox regression
- Classification and objectness prediction

#### 2. FishSegmentationCNN
U-Net architecture for precise fish segmentation.

```python
from src.models.cnn_models import FishSegmentationCNN

model = FishSegmentationCNN(input_channels=3, num_classes=1)
```

**Architecture:**
- Encoder-Decoder structure with skip connections
- 4 encoder blocks + bottleneck + 4 decoder blocks
- Pixel-wise segmentation output

#### 3. EnhancedFishSegmentationCNN
U-Net with CBAM attention mechanisms for improved segmentation.

```python
from src.models.cnn_models import EnhancedFishSegmentationCNN

model = EnhancedFishSegmentationCNN(input_channels=3, num_classes=1)
```

**Features:**
- Channel and Spatial Attention Blocks (CBAM)
- Better feature extraction
- Improved boundary detection

#### 4. FishBiomassCNN
Direct biomass estimation using CNN regression.

```python
from src.models.cnn_models import FishBiomassCNN

model = FishBiomassCNN(input_channels=3, output_features=3)
```

**Architecture:**
- VGG-style feature extraction backbone
- Global average pooling
- Fully connected regression head
- Outputs: [length, width, weight]

#### 5. MultiTaskCNN
Simultaneous detection, segmentation, and biomass estimation.

```python
from src.models.cnn_models import MultiTaskCNN

model = MultiTaskCNN(input_channels=3, num_classes=10)
```

**Features:**
- Shared backbone for all tasks
- Separate task-specific heads
- End-to-end multi-task learning

### Model Parameters

| Model | Parameters | Input Size | Output |
|-------|-----------|------------|--------|
| FishDetectionCNN | ~45M | 224x224 | Boxes, Classes, Scores |
| FishSegmentationCNN | ~31M | 224x224 | Segmentation Mask |
| EnhancedSegmentationCNN | ~35M | 224x224 | Segmentation Mask |
| FishBiomassCNN | ~134M | 224x224 | [length, width, weight] |
| MultiTaskCNN | ~52M | 224x224 | All outputs |

## ⚙️ Configuration

### Main Configuration File (`configs/cnn_config.yaml`)

```yaml
model:
  name: biomass_cnn
  input_channels: 3
  num_classes: 10
  estimation:
    output_features: 3

training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.001
  optimizer: adamw
  scheduler: cosine
  mixed_precision: true
```

See `configs/cnn_config.yaml` for complete configuration options.

## 📈 Model Testing

```python
# Test all models
python src/models/cnn_models.py
```

This will:
- Test all 5 CNN architectures
- Display output shapes
- Show parameter counts
- Verify forward pass

## 🔧 Project Structure

```
Fish-biomass-estimation/
├── README.md
├── requirements.txt
├── train_example.py          # Quick start training script
├── configs/
│   └── cnn_config.yaml       # Model configuration
├── src/
│   ├── models/
│   │   └── cnn_models.py     # CNN architectures
│   ├── train_cnn.py          # Training script
│   ├── dataset.py            # Dataset loader
│   ├── losses.py             # Loss functions
│   └── utils.py              # Utility functions
├── data/
│   ├── train/                # Training data
│   ├── val/                  # Validation data
│   └── test/                 # Test data
├── checkpoints/              # Saved models
└── logs/                     # Training logs
```

## 📝 Custom Dataset

To use your own dataset, create annotations in this format:

```json
{
  "annotations": [
    {
      "image_path": "path/to/image.jpg",
      "bbox": [[x1, y1, x2, y2]],
      "class_id": [0],
      "length": 25.5,
      "width": 8.2,
      "weight": 150.3
    }
  ]
}
```

Save as `annotations.json` in your data directory.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- PyTorch Team
- Deep Learning Community
- Fish Biology Research Community

## 📧 Contact

SuvRules - [@SuvRules](https://github.com/SuvRules)

Project Link: [https://github.com/SuvRules/Fish-biomass-estimation](https://github.com/SuvRules/Fish-biomass-estimation)

---

**⭐ Star this repository if you find it helpful!**