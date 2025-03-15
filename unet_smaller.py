import keras
from keras.api.layers import *
import tensorflow as tf

# Tạo khối cơ bản của encoder với số filter giảm
def encoder_block(inputs, filters, kernel_size=(3, 3), dropout_rate=0.1, batch_norm=True):
    conv = Conv2D(filters, kernel_size, activation=None, padding='same')(inputs)
    if batch_norm:
        conv = BatchNormalization()(conv)
    conv = LeakyReLU(negative_slope=0.1)(conv)
    conv = Conv2D(filters, kernel_size, activation=None, padding='same')(conv)
    if batch_norm:
        conv = BatchNormalization()(conv)
    conv = LeakyReLU(negative_slope=0.1)(conv)
    if dropout_rate > 0:
        conv = Dropout(dropout_rate)(conv)
    return conv

# Simplified attention gate
def attention_gate(x, g, filters):
    # Giảm số feature channels của x và g
    theta_x = Conv2D(filters//2, (1, 1), padding='same')(x)
    phi_g = Conv2D(filters//2, (1, 1), padding='same')(g)
    
    # Tính phi_g + theta_x
    f = Activation('relu')(Add()([theta_x, phi_g]))

    # Tạo attention map
    psi_f = Conv2D(1, (1, 1), padding='same')(f)
    attention = Activation('sigmoid')(psi_f)
    
    return Multiply()([x, attention])

# Function to create a diffusion-ready U-Net
def create_diffusion_unet(img_size=128, img_channels=3, time_embed_dim=256):
    # Image input
    inputs = Input(shape=(img_size, img_size, img_channels))
    
    # Time embedding input
    time_input = Input(shape=(1,))
    
    # Time embedding with sinusoidal positional encoding
    positions = tf.range(0, time_embed_dim // 2, dtype=tf.float32)
    frequencies = 1.0 / (10000.0**(2.0 * (positions / (time_embed_dim // 2))))
    
    def time_embedding(t):
        embedding = tf.cast(t, dtype=tf.float32)
        embedding = tf.reshape(embedding, [-1, 1])
        embedding = tf.matmul(embedding, frequencies[None, :])
        embedding = tf.concat([tf.sin(embedding), tf.cos(embedding)], axis=-1)
        embedding = tf.reshape(embedding, [-1, time_embed_dim])
        return embedding
    
    # Process time embedding
    time_embed = Lambda(lambda t: time_embedding(t))(time_input)
    time_embed = Dense(time_embed_dim, activation="swish")(time_embed)
    time_embed = Dense(time_embed_dim, activation="swish")(time_embed)
    
    # Chuẩn hóa đầu vào
    x = Lambda(lambda x: (x / 255.0) * 2 - 1)(inputs)  # Normalize to [-1, 1]
    
    # Encoder với số filter giảm nhiều
    conv1 = encoder_block(x, 32)
    pool1 = MaxPooling2D((2, 2))(conv1)
    
    # Inject time embedding after first pool (64x64)
    time_proj1 = Dense(32, activation="swish")(time_embed)
    time_proj1 = Reshape((1, 1, 32))(time_proj1)
    time_proj1 = Lambda(lambda x: tf.tile(x, [1, 64, 64, 1]))(time_proj1)
    pool1 = Add()([pool1, time_proj1])
    
    conv2 = encoder_block(pool1, 64)
    pool2 = MaxPooling2D((2, 2))(conv2)
    
    # Inject time embedding after second pool (32x32)
    time_proj2 = Dense(64, activation="swish")(time_embed)
    time_proj2 = Reshape((1, 1, 64))(time_proj2)
    time_proj2 = Lambda(lambda x: tf.tile(x, [1, 32, 32, 1]))(time_proj2)
    pool2 = Add()([pool2, time_proj2])
    
    conv3 = encoder_block(pool2, 128)
    pool3 = MaxPooling2D((2, 2))(conv3)
    
    # Inject time embedding after third pool (16x16)
    time_proj3 = Dense(128, activation="swish")(time_embed)
    time_proj3 = Reshape((1, 1, 128))(time_proj3)
    time_proj3 = Lambda(lambda x: tf.tile(x, [1, 16, 16, 1]))(time_proj3)
    pool3 = Add()([pool3, time_proj3])
    
    # Bottleneck
    conv4 = encoder_block(pool3, 256, dropout_rate=0.3)
    
    # Inject time embedding at bottleneck
    time_proj4 = Dense(256, activation="swish")(time_embed)
    time_proj4 = Reshape((1, 1, 256))(time_proj4)
    time_proj4 = Lambda(lambda x: tf.tile(x, [1, 16, 16, 1]))(time_proj4)
    conv4 = Add()([conv4, time_proj4])
    
    # Decoder với Attention Gates (giảm 1 level)
    up5 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv4)
    attention5 = attention_gate(conv3, up5, 128)
    merge5 = concatenate([attention5, up5], axis=3)
    conv5 = encoder_block(merge5, 128)
    
    up6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv5)
    attention6 = attention_gate(conv2, up6, 64)
    merge6 = concatenate([attention6, up6], axis=3)
    conv6 = encoder_block(merge6, 64)
    
    up7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv6)
    attention7 = attention_gate(conv1, up7, 32)
    merge7 = concatenate([attention7, up7], axis=3)
    conv7 = encoder_block(merge7, 32, dropout_rate=0.1)
    
    # For diffusion models: Output is noise prediction (same channels as input, NO activation)
    outputs = Conv2D(img_channels, (1, 1), activation=None)(conv7)
    
    # Create the diffusion model
    diffusion_model = keras.Model(inputs=[inputs, time_input], outputs=outputs)
    
    # Compile with MSE loss (for noise prediction)
    diffusion_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='mean_squared_error'
    )
    
    return diffusion_model

# Create the original U-Net model (unchanged, for regular tasks)
# Đầu vào (hỗ trợ ảnh RGB 128x128 - smaller input size)
inputs = Input(shape=(128, 128, 3))

# Chuẩn hóa đầu vào
scale = Lambda(lambda x: (x / 255.0) * 2 - 1)(inputs)  # Normalize to [-1, 1]

# Encoder với số filter giảm nhiều
conv1 = encoder_block(scale, 32)  # Reduced from 64
pool1 = MaxPooling2D((2, 2))(conv1)

conv2 = encoder_block(pool1, 64)  # Reduced from 128
pool2 = MaxPooling2D((2, 2))(conv2)

conv3 = encoder_block(pool2, 128)  # Reduced from 256
pool3 = MaxPooling2D((2, 2))(conv3)

# Giảm 1 level so với mạng ban đầu
conv4 = encoder_block(pool3, 256, dropout_rate=0.3)  # Bottleneck, reduced from 512

# Decoder với Attention Gates (giảm 1 level)
up5 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv4)
attention5 = attention_gate(conv3, up5, 128)
merge5 = concatenate([attention5, up5], axis=3)
conv5 = encoder_block(merge5, 128)

up6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv5)
attention6 = attention_gate(conv2, up6, 64)
merge6 = concatenate([attention6, up6], axis=3)
conv6 = encoder_block(merge6, 64)

up7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv6)
attention7 = attention_gate(conv1, up7, 32)
merge7 = concatenate([attention7, up7], axis=3)
conv7 = encoder_block(merge7, 32, dropout_rate=0.1)

# Đầu ra (for regular U-Net tasks like segmentation)
outputs = Conv2D(3, (1, 1), activation='sigmoid')(conv7)

# Tạo model
model = keras.Model(inputs=inputs, outputs=outputs)

# Biên dịch model với optimizer Adam và learning rate thấp hơn
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Recall(), keras.metrics.Precision()]
)

# Create a diffusion-ready U-Net model using the new function
diffusion_unet = create_diffusion_unet(img_size=128, img_channels=3)

# Only print the summary of the diffusion model (comment out if not needed)
diffusion_unet.summary()