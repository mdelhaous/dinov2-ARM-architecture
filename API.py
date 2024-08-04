from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
import io
import requests  # Import the requests library
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Dictionary to map model names to their respective identifiers
MODEL_MAP = {
    'dinov2_vitl14': 'dinov2_vitl14',
    'dinov2_vits14': 'dinov2_vits14',
    'dinov2_vitb14': 'dinov2_vitb14',
    'dinov2_vitg14': 'dinov2_vitg14',
}

# Load the model based on the provided model name
def load_model(model_name: str):
    if model_name not in MODEL_MAP:
        raise ValueError(f"Model {model_name} is not supported.")
    model = torch.hub.load('facebookresearch/dinov2', MODEL_MAP[model_name])
    model.eval()
    return model

# Preprocess the image
def preprocess_image(image):
    input_image = Image.open(io.BytesIO(image)).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model
    return input_batch

# Perform inference
def infer(model, input_batch):
    with torch.no_grad():
        output = model(input_batch)
    return output

@app.post("/infer/")
async def infer_image(file: UploadFile = File(...), model_name: str = Form(...)):
    try:
        if model_name not in MODEL_MAP:
            raise HTTPException(status_code=400, detail="Invalid model name provided.")

        image_bytes = await file.read()
        input_batch = preprocess_image(image_bytes)
        model = load_model(model_name)
        output = infer(model, input_batch)

        return JSONResponse(content={"inference_output": output.tolist()})
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/infer-url/")
async def infer_image_url(url: str = Form(...), model_name: str = Form(...)):
    try:
        if model_name not in MODEL_MAP:
            raise HTTPException(status_code=400, detail="Invalid model name provided.")

        response = requests.get(url)
        if response.status_code != 200:
            logger.error(f"Failed to download image from URL: {url}, Status code: {response.status_code}")
            raise HTTPException(status_code=400, detail="Failed to download image from the provided URL.")

        image_bytes = response.content

        try:
            input_batch = preprocess_image(image_bytes)
        except UnidentifiedImageError:
            logger.error(f"The provided URL does not contain a valid image: {url}")
            raise HTTPException(status_code=400, detail="The provided URL does not contain a valid image.")

        model = load_model(model_name)
        output = infer(model, input_batch)

        return JSONResponse(content={"inference_output": output.tolist()})
    except Exception as e:
        logger.error(f"Error processing image from URL: {url}, Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)