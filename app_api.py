import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn

MODEL_WEIGHTS_PATH = r"outputs\best_model.pth"
CLASS_NAMES = ["Fake", "Real"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Image.MAX_IMAGE_PIXELS = 200_000_000 

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize architecture and weights
model = models.efficientnet_b0()
model.classifier[1] = nn.Linear(1280, len(CLASS_NAMES))
try:
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    print(f"Successfully loaded custom weights: {MODEL_WEIGHTS_PATH}")
except Exception as e:
    print(f"Warning: Could not load weights ({e}). Running with default initialization.")

model.to(device)
model.eval()

app = FastAPI(title="AI Face Authentication System API")

# REST API ENDPOINT
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        request_object_content = await file.read()
        
        img_stream = io.BytesIO(request_object_content)
        image = Image.open(img_stream)
        
        orig_width, orig_height = image.size
        resolution_category = "Large (>=500px)" if (orig_width >= 500 or orig_height >= 500) else "Small (<500px)"

        MAX_BACKEND_SIZE = 1024
        if max(orig_width, orig_height) > MAX_BACKEND_SIZE:
            image.thumbnail((MAX_BACKEND_SIZE, MAX_BACKEND_SIZE), Image.Resampling.LANCZOS)
            
        image = image.convert("RGB")

        input_tensor = test_transform(image)
        input_batch = input_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_batch)
            
            probabilities = torch.nn.functional.softmax(outputs, dim=-1)[0]
            
            confidence, predicted_idx = torch.max(probabilities, dim=0)

        predicted_class = CLASS_NAMES[predicted_idx.item()]
        
        return {
            "status": "success",
            "input_metadata": {
                "dimensions": f"{orig_width}x{orig_height}",
                "category": resolution_category
            },
            "prediction": predicted_class,
            "confidence": round(confidence.item() * 100, 2),
            "probabilities": {
                CLASS_NAMES[i]: round(probabilities[i].item() * 100, 2) for i in range(len(CLASS_NAMES))
            }
        }
    except Exception as err:
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(err)})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)  