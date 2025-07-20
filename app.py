import os
import gradio as gr
import torch
from PIL import Image
import pickle
from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionPipeline
import torch.nn.functional as F

# ---- CONFIG ----
DEVICE = "cpu"  # Use 'cuda' if GPU available and torch.cuda.is_available()
INDEX_PATH = "D:/tank_chatbot/index/image_index.pkl"
IMAGE_ROOT = "D:/tank_chatbot/dataset"
CLIP_MODEL_PATH = "openai/clip-vit-base-patch32"

# ✅ Update this to your actual snapshot folder for stable diffusion
SD_MODEL_PATH = os.path.abspath("D:/tank_chatbot/models/stable_diffusion/models--runwayml--stable-diffusion-v1-5")
print(f"📁 Using Stable Diffusion path: {SD_MODEL_PATH}")
print("📂 Contents of SD_MODEL_PATH:", os.listdir(SD_MODEL_PATH))

# ---- Load CLIP ----
print("🔄 Loading CLIP model...")
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_PATH, local_files_only=True).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_PATH, local_files_only=True)

# ---- Load Stable Diffusion ----
print("🎨 Loading Stable Diffusion model...")
sd_pipeline = StableDiffusionPipeline.from_pretrained(
    SD_MODEL_PATH,
    torch_dtype=torch.float32,
    use_safetensors=True,  # Optional: set to False if not using safetensors
    local_files_only=True
).to(DEVICE)

# ---- Load Embeddings and Paths ----
with open(INDEX_PATH, "rb") as f:
    index_data = pickle.load(f)

embeddings = index_data["embeddings"]
paths = index_data["paths"]
print("📦 Keys in index_data:", index_data.keys())

# ---- Image Search Function ----
def search_similar_images(query_img):
    # Preprocess image
    inputs = clip_processor(images=query_img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)
    
    embedding = F.normalize(embedding, dim=1)
    embeddings_norm = F.normalize(embeddings, dim=1)

    similarity = torch.matmul(embedding, embeddings_norm.T)  # shape: (1, N)
    topk = torch.topk(similarity, k=5)

    indices = topk.indices[0].tolist()
    results = [Image.open(paths[i]) for i in indices]
    return results

# ---- Text-to-Image Function ----
def generate_image_from_text(prompt):
    image = sd_pipeline(prompt).images[0]
    return image

# ---- Gradio Interface ----
search_interface = gr.Interface(
    fn=search_similar_images,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(label=f"Result {i+1}") for i in range(5)],
    title="🔍 Image Similarity Search",
    description="Upload a tank/vehicle image to retrieve visually similar images."
)

generate_interface = gr.Interface(
    fn=generate_image_from_text,
    inputs=gr.Textbox(lines=2, placeholder="Enter a military scene or tank prompt..."),
    outputs=gr.Image(),
    title="🎨 Text to Image Generator",
    description="Generate military-themed images using Stable Diffusion."
)

# Combine both tabs
app = gr.TabbedInterface(
    interface_list=[search_interface, generate_interface],
    tab_names=["🔍 Image Search", "🎨 Text-to-Image"]
)

# ---- Launch App ----
if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)
