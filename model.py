import tensorflow as tf
import numpy as np
from keras.api.layers import *
from keras import Model
import matplotlib.pyplot as plt
from tqdm import tqdm

from unet import model as unet_base  # Import your existing U-Net model

class DiffusionModel:
    def __init__(
        self,
        img_size=256,
        img_channels=3,
        timesteps=1000,
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
        # Ensure all numpy arrays are float32
        # Define cosine noise schedule (better than linear for many cases)
        def cosine_beta_schedule(timesteps, s=0.008):
            """
            Create a cosine noise schedule as proposed in the improved DDPM paper
            """
            steps = timesteps + 1
            x = np.linspace(0, timesteps, steps)
            alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return np.clip(betas, 0, 0.999).astype(np.float32)
            
        self.betas = cosine_beta_schedule(timesteps)
        
        # Pre-calculate diffusion parameters for efficiency
        self.alphas = (1.0 - self.betas).astype(np.float32)
        self.alphas_cumprod = np.cumprod(self.alphas).astype(np.float32)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1]).astype(np.float32)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod).astype(np.float32)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod).astype(np.float32)
        self.sqrt_recip_alphas = np.sqrt(1.0 / self.alphas).astype(np.float32)
        
        # Parameters for sampling
        self.posterior_variance = (self.betas * 
            (1.0 - self.alphas_cumprod_prev) / 
            (1.0 - self.alphas_cumprod)).astype(np.float32)
    
    def q_sample(self, x_0, t):
        """
        Add noise to images following the forward diffusion process.
        
        Args:
            x_0: Input images, shape [batch_size, height, width, channels]
            t: Timesteps, shape [batch_size]
            
        Returns:
            x_t: Noisy images at timestep t
            noise: The noise added to the images
        """
        # Cast to float32 for computation
        x_0 = tf.cast(x_0, tf.float32) / 255.0  # Normalize to [0, 1]
        x_0 = x_0 * 2.0 - 1.0  # Scale to [-1, 1]
        
        # Tạo noise ngẫu nhiên
        noise = tf.random.normal(shape=tf.shape(x_0), dtype=tf.float32)

        # Điều chỉnh t để phù hợp với kích thước batch thực tế
        actual_batch_size = tf.shape(x_0)[0]
        t = t[:actual_batch_size]  # Thêm dòng này để cắt t
        
        # Truy cập các tham số đã tính từ bảng tra
        sqrt_alphas_cumprod_t = tf.gather(self.sqrt_alphas_cumprod, t)
        sqrt_one_minus_alphas_cumprod_t = tf.gather(self.sqrt_one_minus_alphas_cumprod, t)
        
        # Reshape để phù hợp với kích thước ảnh
        sqrt_alphas_cumprod_t = tf.reshape(sqrt_alphas_cumprod_t, [-1, 1, 1, 1])
        sqrt_one_minus_alphas_cumprod_t = tf.reshape(sqrt_one_minus_alphas_cumprod_t, [-1, 1, 1, 1])
        
        # Phương trình khuếch tán chuyển tiếp
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        
        # Làm sạch bộ nhớ, chỉ giữ lại kết quả cuối cùng
        tf.keras.backend.clear_session()
        
        return x_t, noise
        
    def train(self, dataset, model=unet_base, epochs=30, batch_size=8, callbacks=None):
        """
        Train the diffusion model
    
        Args:
            dataset: Training dataset of images
            model: The neural network model
            epochs: Number of training epochs
            batch_size: Batch size
            callbacks: List of Keras callbacks
        """
        if callbacks is None:
            callbacks = []
            
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            progress_bar = tqdm(total=len(dataset))
            epoch_losses = []
            
            for x_batch in dataset.batch(batch_size):
                # Get the actual batch size (might be smaller for the last batch)
                actual_batch_size = tf.shape(x_batch)[0]
                
                # Sample random timesteps matching the actual batch size
                t = tf.random.uniform(
                    shape=[actual_batch_size], 
                    minval=0, 
                    maxval=self.timesteps, 
                    dtype=tf.int32
                )
                
                # Add noise according to timestep
                x_t, noise_added = self.q_sample(x_batch, t)
                
                # Train the model to predict the noise
                loss = model.train_on_batch([x_t, t], noise_added)
                epoch_losses.append(loss)
                
                progress_bar.update(1)
                progress_bar.set_description(f"Loss: {loss:.4f}")
            
            # Call callbacks at the end of each epoch
            logs = {'loss': np.mean(epoch_losses)}
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs=logs)
    
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
        # Ensure everything is float32
        x_t = tf.cast(x_t, tf.float32)
        batch_size = tf.shape(x_t)[0]
        t_tensor = tf.ones(batch_size, dtype=tf.int32) * t
        
        # Predict the noise component
        predicted_noise = model([x_t, t_tensor], training=False)
        
        # Cast parameters to float32
        alpha_t = tf.cast(self.alphas[t], tf.float32)
        alpha_cumprod_t = tf.cast(self.alphas_cumprod[t], tf.float32)
        beta_t = tf.cast(self.betas[t], tf.float32)
        
        # Calculate the mean for sampling
        coeff = beta_t / tf.sqrt(1 - alpha_cumprod_t)
        x_0_pred = (1 / tf.sqrt(alpha_t)) * (x_t - coeff * predicted_noise)
        x_0_pred = tf.clip_by_value(x_0_pred, -1.0, 1.0)
        
        # Calculate the variance
        variance = 0.0
        if t > 0:
            variance = tf.cast(self.posterior_variance[t], tf.float32)
            
        # Sample from posterior
        noise = tf.random.normal(shape=x_t.shape, dtype=tf.float32)
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