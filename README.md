
# Simple U-Net for Image Segmentation

## Hugging Face Model & Demo
This model is also hosted on Hugging Face Spaces, providing an easy-to-use, interactive demo. Visit the following link to test the model in your browser:
[https://huggingface.co/spaces/hemant-bhambhu/Image_segmentation_unet_model](Image_segmentation_unet_model)


## Quick Start

### Requirements
- Python 3.6+
- PyTorch 2.2.1
- torchvision 0.17.1
- pytorch-lightning 2.2.1
- gradio 4.21.0

### Setup
```bash
git https://github.com/hemant1456/UNET_implementation.git
cd UNET_implementation
pip install -r requirements.txt
```

### Train
```bash
python main.py
```

### Demo
```bash
python app.py
```

## Overview
This project is a straightforward implementation of the U-Net architecture for educational purposes. It includes a training script, model definition, and a Gradio demo app for easy testing.

- `main.py`: Entry point for training the model on the Oxford-IIIT Pet Dataset.
- `UNET_model.py`: The U-Net model architecture.
- `app.py`: A Gradio web app for demonstrating the model on your images.

## Acknowledgments
- U-Net: Convolutional Networks for Biomedical Image Segmentation by Olaf Ronneberger, Philipp Fischer, and Thomas Brox.

Feel free to use this project for learning and experimentation!

