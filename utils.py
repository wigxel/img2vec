import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image as PILImage


IMAGE_SHAPE = (224, 224)
LAYER_OUTPUT_SIZE = 2048


# Define the transformations
scaler = transforms.Resize(IMAGE_SHAPE)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

# Compose the transformations
transform = transforms.Compose([
    scaler,
    to_tensor,
    normalize
])

def vectorize_image(image_path, model_dict):
  img = PILImage.open(image_path).convert('RGB')

  image = transform(img).unsqueeze(0)
  embedding = torch.zeros(1, LAYER_OUTPUT_SIZE, 1, 1)

  def copy_data(m, i, o):
    embedding.copy_(o.data)

  h = model_dict["extraction_layer"].register_forward_hook(copy_data)
  model_dict["resnet50"](image)
  h.remove()

  return embedding.numpy()[0, :, 0, 0]

# Function to check similarity between two images
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)

    return dot_product / (magnitude1 * magnitude2)

