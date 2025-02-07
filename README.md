# AI/ML Internship Task
This project includes synthetic image generation, preprocessing, and a Flux-based forward pass.

## Installation
Run:
-- Install diffusers (Stable Diffusion), torch, transformers, PIL, and OpenCV 
pip install diffusers torch torchvision transformers pillow opencv-python matplotlib
--Install Julia
--Install Flux library in Julia
using Pkg
Pkg.add("Flux")
Pkg.add("Images")
Pkg.add("FileIO")
--Generate Synthetic Images using Stable Diffusion
--Generate images based on a text prompt
-- Steps:
1.Load the StableDiffusionPipeline from diffusers
2.Provide a creative text prompt
3.Generate and save three images
-- code : 
import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

# Load the model
pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

# Define the prompt
prompt = "a serene sunset over a futuristic city"

# Generate images
for i in range(3):
    image = pipeline(prompt).images[0]  # Generate image
    image.save(f"generated_image_{i+1}.png")  # Save to disk
    plt.imshow(image)
    plt.axis("off")
    plt.show()
    --- Outcome: You will have three images saved as generated_image_1.png, generated_image_2.png, etc.
    --Preprocess the Images
📌 Objective: Resize, normalize, and optionally convert images to grayscale.
 Steps:
1.Load images using OpenCV or PIL
2.Resize to 224×224
3.Normalize pixel values (0 to 1)
4.Save/display the preprocessed images
code:
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Load the image
image_path = "generated_image_1.png"
image = Image.open(image_path)

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # Resize
    transforms.ToTensor(),           # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize (-1 to 1)
])

preprocessed_image = transform(image)

# Convert tensor back to image for verification
image_array = preprocessed_image.numpy().transpose(1, 2, 0)
cv2.imshow("Preprocessed Image", image_array)
cv2.waitKey(0)
cv2.destroyAllWindows()
-- Outcome:  now we resized and normalized images ready for model input.
--- Build a Simple Neural Network in Flux (Julia)
📌 Objective: Pass a preprocessed image through a minimal Flux model.

 Steps:
1.Define a simple CNN in Flux
2.Load a preprocessed image
3.Run a forward pass
using Flux, Images, FileIO
code :
# Define a simple CNN
model = Chain(
    Conv((3,3), 3=>16, relu),  # Conv layer
    MaxPool((2,2)),            # Pooling
    Conv((3,3), 16=>32, relu),  # Conv layer
    MaxPool((2,2)),
    flatten,                    # Flatten the output
    Dense(32*54*54, 10, relu),   # Dense layer
    Dense(10, 2),                # Output layer
    softmax
)

# Load an image
img = load("generated_image_1.png")
img = imresize(img, (224, 224))  # Resize
img_tensor = Float32.(channelview(img)) # Convert to tensor

# Perform forward pass
output = model(img_tensor)
println("Model output: ", output)
-- Outcome: The model will process the image and print predictions
