# Real vs. AI Face Classifier

This project trains a neural network to classify face images as either real or AI-generated. It uses a pre-trained EfficientNet-B0 model in PyTorch, fine-tunes the top layers, and includes a web interface and an API for testing images.

## What's in this project?

* **`train.py`**: The training loop. It uses an AdamW optimizer, a Cosine Annealing learning rate scheduler, and saves the best model weights and training curves to an `outputs/` folder.
* **`dataset.py`**: Handles loading and preparing the images for the model.
* **`app_gui.py`**: A simple drag-and-drop web interface built with Gradio so you can test the trained model visually.
* **`app_api.py`**: An API script to run predictions on images from other applications.

## How it works

**Model**: It uses EfficientNet-B0. The lower feature-extraction layers are frozen to keep the pre-trained weights, while the last two feature layers and the final classifier head are trained on your dataset.

## Setup

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
   cd YOUR_REPOSITORY_NAME
   
2. Install the required Python packages:
   ```bash
    pip install torch torchvision matplotlib gradio
   
3. Place your data in a folder named dataset/
