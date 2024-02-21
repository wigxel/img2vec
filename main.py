from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import numpy as np
import json

from contextlib import asynccontextmanager

from torchvision.models import resnet50, ResNet50_Weights

from fastapi.responses import RedirectResponse
from fastapi import status

from utils import vectorize_image, cosine_similarity

import shutil

model_dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    model_dict["resnet50"] = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model_dict["resnet50"].eval()
    model_dict["extraction_layer"] = model_dict["resnet50"]._modules.get('avgpool')
    yield
    # Clean up the ML models and release the resources
    model_dict.clear()

app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/search")
async def query_endpoint(request: Request, file: UploadFile):
    embedding = vectorize_image(file.file, model_dict)
    with open("vector_store.json", 'r') as file:
        data = json.load(file)
    
    results = {}

    for i in data.keys():
        results[i] = cosine_similarity(np.array(data[i]), embedding)

    results = sorted(results.items(), key=lambda item: item[1], reverse=True)[:3]

    results = dict(results)

    return templates.TemplateResponse(request=request, context = {"results": results }, name="result.html")

@app.post("/vectorize/")
async def upload_file(file: UploadFile):
    filename = file.filename
    with open(f"static/images/{filename}", "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    embedding = vectorize_image(file.file, model_dict)
    with open("vector_store.json", 'r') as file:
        data = json.load(file)

    # Add the new image data to the JSON data
    data[filename] = embedding.tolist()

    # Write the updated data back to the JSON file
    with open("vector_store.json", 'w') as file:
        json.dump(data, file, indent=4)

    return {"message": "Image Successfully Vectorized"}