import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# CONFIGURATION
IMAGE_PATH = r"dataset\test\real\00053.jpg"  
MODEL_WEIGHTS_PATH = "outputs/model_epoch_2.pth"

CLASS_NAMES = ["Fake", "Real"] 


# SETUP MODEL AND PIPELINE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Recreate the exact same transforms used in validation
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Rebuild the model architecture
model = models.efficientnet_b0()
model.classifier[1] = nn.Linear(1280, len(CLASS_NAMES))

# Load trained weights
if not os.path.exists(MODEL_WEIGHTS_PATH):
    raise FileNotFoundError(f"Could not find weights file at {MODEL_WEIGHTS_PATH}. Did you run training first?")

model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
model = model.to(device)
model.eval()

# PREDICTION FUNCTION
def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    input_tensor = test_transform(image)
    input_batch = input_tensor.unsqueeze(0)
    input_batch = input_batch.to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(input_batch)
        
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        confidence, predicted_idx = torch.max(probabilities, 0)
        
    predicted_class = CLASS_NAMES[predicted_idx.item()]
    print(f"\n--- Prediction Result ---")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Result: {predicted_class}")
    print(f"Confidence: {confidence.item() * 100:.2f}%")
    
    print(f"\nFull Breakdown:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"  {class_name}: {probabilities[i].item() * 100:.2f}%")

if __name__ == "__main__":
    predict_image(IMAGE_PATH)