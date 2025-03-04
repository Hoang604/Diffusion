import keras
from keras.api.layers import *

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

# Đầu ra
outputs = Conv2D(3, (1, 1), activation='sigmoid')(conv7)

# Tạo model
model = keras.Model(inputs=inputs, outputs=outputs)

# Biên dịch model với optimizer Adam và learning rate thấp hơn
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Recall(), keras.metrics.Precision()]
)

model.summary()