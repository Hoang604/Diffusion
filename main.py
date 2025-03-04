import tensorflow as tf
from diffusion_model import DiffusionModel  # Update import path to your diffusion model file
from unet_tiny import model as unet_model   # Make sure this matches your reduced U-Net filename

# Create diffusion model with MATCHING IMAGE SIZE to U-Net
diffusion = DiffusionModel(
    img_size=128,  # Changed from 256 to match your U-Net input size
    img_channels=3,
    timesteps=1000,
    beta_start=1e-4,
    beta_end=0.02
)

# Create diffusion-compatible U-Net model
diffusion_model = diffusion.modify_unet_for_diffusion(unet_model)

# Load your dataset
# Use an actual path to your dataset
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    '/home/hoang/python/diffusion/dataset',  # Update to your actual dataset path
    image_size=(128, 128),
    batch_size=16,  # Reduced batch size for better memory usage
    label_mode=None  # No labels needed for diffusion
)

# Preprocess images
dataset = dataset.map(lambda x: x / 255.0)  # Normalize to [0, 1]

# Train the model
diffusion.train(diffusion_model, dataset, epochs=30, batch_size=8)  # Reduced batch size again

# Generate images
generated_images = diffusion.generate_images(diffusion_model, num_images=4)

# Visualize results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(generated_images[i])
    plt.title(f"Generated Image {i+1}")
    plt.axis('off')
plt.savefig('generated_images.png')
plt.show()