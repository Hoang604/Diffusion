import tensorflow as tf
import numpy as np
from keras.api.layers import *
from keras import Model
import matplotlib.pyplot as plt
from tqdm import tqdm

from unet_smaller import model as unet_base  # Import your existing U-Net model

class DiffusionModel:
    def __init__(
        self,
        img_size=256,
        img_channels=3,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02
    ):
        """
        Initialize the diffusion model
        
        Args:
            img_size: Image dimensions (assumed square)
            img_channels: Number of image channels (3 for RGB)
            timesteps: Number of diffusion steps
            beta_start: Starting variance schedule value
            beta_end: Ending variance schedule value
        """
        self.img_size = img_size
        self.img_channels = img_channels
        self.timesteps = timesteps
        
        # Define noise schedule (linear between beta_start and beta_end)
        self.betas = np.linspace(beta_start, beta_end, timesteps)
        
        # Pre-calculate diffusion parameters for efficiency
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = np.sqrt(1.0 / self.alphas)
        
        # Parameters for sampling
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
    def modify_unet_for_diffusion(self, base_model):
        """
        Modify the existing U-Net model for diffusion tasks by adding timestep embedding
        
        Args:
            base_model: The base U-Net model
        """
        # Create a new model with the same U-Net structure but adding timestep input
        inputs = Input(shape=(self.img_size, self.img_size, self.img_channels))
        time_input = Input(shape=(1,))
        
        # Time embedding with sinusoidal positional encoding (similar to Transformers)
        time_embed_dim = 256
        positions = tf.range(0, time_embed_dim // 2, dtype=tf.float32)
        frequencies = 1.0 / (10000.0**(2.0 * (positions / (time_embed_dim // 2))))
        
        def time_embedding(t):
            embedding = tf.cast(t, dtype=tf.float32)
            embedding = tf.reshape(embedding, [-1, 1])
            embedding = tf.matmul(embedding, frequencies[None, :])
            embedding = tf.concat([tf.sin(embedding), tf.cos(embedding)], axis=-1)
            embedding = tf.reshape(embedding, [-1, time_embed_dim])
            return embedding
            
        # Time embedding layer
        time_embed = Lambda(lambda t: time_embedding(t))(time_input)
        time_embed = Dense(time_embed_dim, activation="swish")(time_embed)
        time_embed = Dense(time_embed_dim, activation="swish")(time_embed)
        
        # Pass the input through the existing U-Net layers, but inject time embedding
        # at key points through the network
        
        # Extract layers from your base U-Net
        # Here we're making a simplified adaptation - a complete implementation would 
        # inject time information at multiple layers
        
        # Create a new U-Net model with the same architecture but accepting time embedding
        # This is a simplified example - you'll need to adjust based on your U-Net structure
        
        # For a complete implementation, you would need to reimplement your U-Net
        # adding time embedding at various stages
        
        # For demonstration, we'll use an approach that works with the existing U-Net
        x = base_model.layers[1](inputs)  # Skip the input layer
        
        # Continue with the U-Net processing
        for layer in base_model.layers[2:-1]:  # All layers except input and output
            x = layer(x)
        
        # Final output
        outputs = Conv2D(self.img_channels, (1, 1), activation='sigmoid')(x)
        
        diffusion_model = Model(inputs=[inputs, time_input], outputs=outputs)
        
        # Compile the model
        diffusion_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='mean_squared_error'
        )
        
        return diffusion_model
        
    def q_sample(self, x_0, t):
        """
        Forward diffusion process: add noise to image according to timestep t
        
        Args:
            x_0: Original image with values in [0, 1]
            t: Timestep
        
        Returns:
            Noisy version of x_0 at timestep t
        """
        t = tf.cast(t, dtype=tf.int32)
        noise = tf.random.normal(shape=tf.shape(x_0))
        
        sqrt_alphas_cumprod_t = tf.gather(self.sqrt_alphas_cumprod, t)
        sqrt_one_minus_alphas_cumprod_t = tf.gather(self.sqrt_one_minus_alphas_cumprod, t)
        
        # Apply reshaping to match image dimensions
        sqrt_alphas_cumprod_t = tf.reshape(sqrt_alphas_cumprod_t, [-1, 1, 1, 1])
        sqrt_one_minus_alphas_cumprod_t = tf.reshape(sqrt_one_minus_alphas_cumprod_t, [-1, 1, 1, 1])
        
        # Formula: x_t = √(α_t) * x_0 + √(1-α_t) * ε
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise
    
    def train(self, model, dataset, epochs=30, batch_size=32):
        """
        Train the diffusion model
        
        Args:
            model: The neural network model (modified U-Net)
            dataset: Training dataset of images
            epochs: Number of training epochs
            batch_size: Batch size
        """
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            progress_bar = tqdm(total=len(dataset))
            
            for x_batch in dataset.batch(batch_size):
                # Normalize images to [-1, 1]
                x_batch = (tf.cast(x_batch, tf.float32) / 127.5) - 1.0
                
                # Sample random timesteps
                t = tf.random.uniform(
                    shape=[batch_size], 
                    minval=0, 
                    maxval=self.timesteps, 
                    dtype=tf.int32
                )
                
                # Add noise according to timestep
                x_t, noise_added = self.q_sample(x_batch, t)
                
                # Train the model to predict the noise
                loss = model.train_on_batch([x_t, t], noise_added)
                
                progress_bar.update(1)
                progress_bar.set_description(f"Loss: {loss:.4f}")
            
            # Save model checkpoint
            if (epoch + 1) % 5 == 0:
                model.save_weights(f"diffusion_model_epoch_{epoch+1}.h5")
    
    def p_sample(self, model, x_t, t):
        """
        Sample from the model at timestep t
        
        Args:
            model: The neural network model
            x_t: Current noisy image
            t: Current timestep
            
        Returns:
            Predicted less noisy image at timestep t-1
        """
        t_tensor = tf.ones(x_t.shape[0], dtype=tf.int32) * t
        
        # Predict the noise component
        predicted_noise = model([x_t, t_tensor], training=False)
        
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alphas_cumprod[t]
        beta_t = self.betas[t]
        
        # Calculate the mean for sampling
        coeff = beta_t / tf.sqrt(1 - alpha_cumprod_t)
        x_0_pred = (1 / tf.sqrt(alpha_t)) * (x_t - coeff * predicted_noise)
        x_0_pred = tf.clip_by_value(x_0_pred, -1.0, 1.0)
        
        # Calculate the variance
        variance = 0.0
        if t > 0:
            variance = self.posterior_variance[t]
            
        # Sample from posterior
        noise = tf.random.normal(shape=x_t.shape)
        x_t_minus_1 = x_0_pred + tf.sqrt(variance) * noise
        
        return x_t_minus_1
    
    def generate_images(self, model, num_images=4):
        """
        Generate images using the diffusion model
        
        Args:
            model: The neural network model
            num_images: Number of images to generate
            
        Returns:
            Generated images
        """
        # Start with pure noise
        x_t = tf.random.normal(shape=[num_images, self.img_size, self.img_size, self.img_channels])
        
        # Sample step by step, from t=T to t=0
        for t in tqdm(range(self.timesteps - 1, -1, -1)):
            x_t = self.p_sample(model, x_t, t)
            
        # Convert from [-1,1] range to [0,1]
        generated_images = (x_t + 1.0) / 2.0
        generated_images = tf.clip_by_value(generated_images, 0.0, 1.0)
        
        return generated_images
    
    def visualize_diffusion_steps(self, model, x_0, num_steps_to_show=10):
        """
        Visualize the diffusion process from clean to noisy and back
        
        Args:
            model: The neural network model
            x_0: Original clean image
            num_steps_to_show: Number of intermediate steps to display
            
        Returns:
            Visualization of diffusion process
        """
        # Normalize image to [-1,1]
        x_0 = tf.cast(x_0, tf.float32) / 127.5 - 1.0
        x_0 = tf.expand_dims(x_0, axis=0)  # Add batch dimension
        
        # Select timesteps to visualize
        step_size = self.timesteps // num_steps_to_show
        timesteps_to_show = list(range(0, self.timesteps, step_size))
        
        # Forward process (adding noise)
        forward_images = []
        for t in timesteps_to_show:
            noisy_image, _ = self.q_sample(x_0, [t])
            forward_images.append(noisy_image[0])
        
        # Reverse process (denoising)
        reverse_images = []
        x_t = tf.random.normal(shape=x_0.shape)  # Start with noise
        
        for t in tqdm(range(self.timesteps - 1, -1, -1)):
            x_t = self.p_sample(model, x_t, t)
            if t in timesteps_to_show[::-1]:
                reverse_images.append(x_t[0])
        
        # Visualize
        num_images = len(forward_images) + len(reverse_images)
        plt.figure(figsize=(num_images * 2, 4))
        
        # Plot forward process
        for i, img in enumerate(forward_images):
            plt.subplot(2, num_images // 2, i + 1)
            plt.imshow((img.numpy() + 1) / 2)  # Convert from [-1,1] to [0,1]
            plt.title(f"t={timesteps_to_show[i]}")
            plt.axis('off')
        
        # Plot reverse process
        for i, img in enumerate(reverse_images):
            plt.subplot(2, num_images // 2, i + len(forward_images) + 1)
            plt.imshow((img.numpy() + 1) / 2)  # Convert from [-1,1] to [0,1]
            plt.title(f"t={timesteps_to_show[len(reverse_images)-i-1]}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('diffusion_visualization.png')
        plt.show()