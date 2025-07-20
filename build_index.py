import os
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import pickle

# Initialize model with safetensors
model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32",
    use_safetensors=True
)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Recursively collect image paths
base_dir = "D:/tank_chatbot/dataset"
image_paths = []

for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            image_paths.append(os.path.join(root, file))

# Store embeddings and paths
image_embeddings = []
image_names = []

print(f"📦 Found {len(image_paths)} images. Building index...")

for path in tqdm(image_paths, desc="Embedding images"):
    try:
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            embedding = model.get_image_features(**inputs)
        image_embeddings.append(embedding[0].cpu())
        image_names.append(path)
    except Exception as e:
        print(f"❌ Failed to process {path}: {e}")

# Save index
os.makedirs("D:/tank_chatbot/index", exist_ok=True)
with open("D:/tank_chatbot/index/image_index.pkl", "wb") as f:
    pickle.dump({
        "embeddings": torch.stack(image_embeddings),
        "paths": image_names
    }, f)

print("✅ Done. Index saved to D:/tank_chatbot/index/image_index.pkl")
