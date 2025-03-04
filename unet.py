import keras
from keras.api.layers import *


# Tạo khối cơ bản của encoder
def encoder_block(inputs, filters, kernel_size=(3, 3), dropout_rate=0.2, batch_norm=True):
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

# Tạo attention gate, nhằm mục đích sửa lại x (là feature map của encoder) để tập trung vào vùng quan trọng
def attention_gate(x, g, filters):
    # Giảm số feature channels của x và g
    theta_x = Conv2D(filters, (1, 1), padding='same')(x)
    phi_g = Conv2D(filters, (1, 1), padding='same')(g)
    
    # Tính phi_g + theta_x
    f = Activation('relu')(Add()([theta_x, phi_g]))

    # Tạo attention map (có giá trị từ 0 đến 1, với mục đích tập trung vào vùng quan trọng)
    psi_f = Conv2D(1, (1, 1), padding='same')(f)

    # Tính attention map
    attention = Activation('sigmoid')(psi_f)
    
    # Tính x * attention (những vùng quan trọng - có giá trị trong attention map cao gần như giữ nguyên, còn lại giảm về 0)
    return Multiply()([x, attention])

# Đầu vào (hỗ trợ ảnh RGB 256x256)
inputs = Input(shape=(256, 256, 3))

# Chuẩn hóa đầu vào
scale = Lambda(lambda x: (x / 255.0) * 2 - 1)(inputs)  # Normalize to [-1, 1]

# Encoder với số filter lớn hơn
conv1 = encoder_block(scale, 64)
pool1 = MaxPooling2D((2, 2))(conv1)

conv2 = encoder_block(pool1, 128)
pool2 = MaxPooling2D((2, 2))(conv2)

conv3 = encoder_block(pool2, 256)
pool3 = MaxPooling2D((2, 2))(conv3)

conv4 = encoder_block(pool3, 512)
pool4 = MaxPooling2D((2, 2))(conv4)

conv5 = encoder_block(pool4, 1024)
pool5 = MaxPooling2D((2, 2))(conv5)  # Thêm một khối sâu hơn

# Khối đáy
conv6 = encoder_block(pool5, 2048, dropout_rate=0.5)

# Decoder với Attention Gates
up7 = Conv2DTranspose(1024, (2, 2), strides=(2, 2), padding='same')(conv6)
attention7 = attention_gate(conv5, up7, 1024)
merge7 = concatenate([attention7, up7], axis=3)
conv7 = encoder_block(merge7, 1024)

up8 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv7)
attention8 = attention_gate(conv4, up8, 512)
merge8 = concatenate([attention8, up8], axis=3)
conv8 = encoder_block(merge8, 512)

up9 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv8)
attention9 = attention_gate(conv3, up9, 256)
merge9 = concatenate([attention9, up9], axis=3)
conv9 = encoder_block(merge9, 256)

up10 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv9)
attention10 = attention_gate(conv2, up10, 128)
merge10 = concatenate([attention10, up10], axis=3)
conv10 = encoder_block(merge10, 128)

up11 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv10)
attention11 = attention_gate(conv1, up11, 64)
merge11 = concatenate([attention11, up11], axis=3)
conv11 = encoder_block(merge11, 64, dropout_rate=0.1)

# Đầu ra
outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv11)

# Tạo model
model = keras.Model(inputs=inputs, outputs=outputs)

# Biên dịch model với optimizer Adam và learning rate thấp hơn
model.compile(
    optimizer= keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Recall(), keras.metrics.Precision()]
)

model.summary()