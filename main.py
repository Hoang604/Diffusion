import tensorflow as tf
import os
import numpy as np
import time

# GPU memory growth setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Setup basic directories
BASE_DIR = '/home/hoang/python/diffusion'
SAVE_DIR = os.path.join(BASE_DIR, 'saved_models')
os.makedirs(SAVE_DIR, exist_ok=True)

# Import required modules
from unet import create_diffusion_unet
from model import DiffusionModel

# Use float32 precision
tf.keras.mixed_precision.set_global_policy('float32')

# Simple gradient accumulator
class GradientAccumulator:
    def __init__(self, model, steps=4):
        self.model = model
        self.steps = steps
        self.optimizer = model.optimizer
        self.accumulated_gradients = [
            tf.Variable(tf.zeros_like(var, dtype=tf.float32))
            for var in model.trainable_variables
        ]
        self.current_step = 0
        
    def reset_gradients(self):
        for i, grad in enumerate(self.accumulated_gradients):
            self.accumulated_gradients[i].assign(tf.zeros_like(grad))
        
    def apply_gradients(self):
        self.optimizer.apply_gradients(
            zip(self.accumulated_gradients, self.model.trainable_variables)
        )
        self.reset_gradients()
        
    def train_step(self, x_batch, t_batch, noise):
        with tf.GradientTape() as tape:
            predicted_noise = self.model([x_batch, t_batch], training=True)
            loss = tf.reduce_mean(tf.square(predicted_noise - noise))
            scaled_loss = loss / self.steps
        
        gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
        
        for i, grad in enumerate(gradients):
            if grad is not None:
                self.accumulated_gradients[i].assign_add(grad)
        
        self.current_step += 1
        
        if self.current_step >= self.steps:
            self.apply_gradients()
            self.current_step = 0
        
        return loss

# Basic training function
def train_with_accumulation(model, diffusion, dataset, epochs, accumulation_steps):
    accumulator = GradientAccumulator(model=model, steps=accumulation_steps)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0
        batch_count = 0
        
        for batch_idx, batch in enumerate(dataset):
            t = tf.random.uniform(
                shape=[tf.shape(batch)[0]], 
                minval=0, 
                maxval=diffusion.timesteps, 
                dtype=tf.int32
            )
            
            x_t, noise_added = diffusion.q_sample(batch, t)
            loss = accumulator.train_step(x_t, t, noise_added)
            
            epoch_loss += loss.numpy()
            batch_count += 1
            
            if batch_idx % 20 == 0:
                print(f"Batch {batch_idx} - Loss: {loss:.4f}")
        
        avg_loss = epoch_loss / batch_count
        print(f"Epoch {epoch+1} complete - Average Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % 10 == 0:
            model.save_weights(os.path.join(SAVE_DIR, f"diffusion_model_epoch_{epoch+1}.h5"))

# Main function
def main():
    # Basic configuration
    IMG_SIZE = 128
    IMG_CHANNELS = 3
    BATCH_SIZE = 4
    EPOCHS = 30
    TIMESTEPS = 1000
    BETA_START = 1e-4
    BETA_END = 0.02
    
    # Create diffusion model
    diffusion = DiffusionModel(
        img_size=IMG_SIZE,
        img_channels=IMG_CHANNELS,
        timesteps=TIMESTEPS,
        beta_start=BETA_START,
        beta_end=BETA_END
    )
    
    # Create optimizer and model
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=5e-4,
        weight_decay=1e-6
    )
    
    diffusion_model = create_diffusion_unet(
        img_size=IMG_SIZE, 
        img_channels=IMG_CHANNELS,
        base_channels=32
    )
    
    diffusion_model.compile(
        optimizer=optimizer, 
        loss='mean_squared_error'
    )
    
    # Create dataset
    dataset_path = '/home/hoang/python/diffusion/dataset/images'
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_path,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode=None,
        shuffle=True
    )
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    # Train
    train_with_accumulation(
        model=diffusion_model,
        diffusion=diffusion,
        dataset=dataset,
        epochs=EPOCHS,
        accumulation_steps=4
    )
    
    # Save final model
    diffusion_model.save_weights(os.path.join(SAVE_DIR, "diffusion_final_weights.h5"))
    print("Training completed successfully!")

if __name__ == "__main__":
    main()