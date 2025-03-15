import keras
from keras.api.layers import *
import tensorflow as tf

# Tạo khối ResNet cải tiến với skip connection (tích chập chính bao gồm 2 khối convolution + skip connection)
def resnet_block(inputs, filters, kernel_size=(3, 3), dropout_rate=0.1, batch_norm=True):
    skip = inputs
    
    # Nếu số kênh đầu vào khác với số filter, điều chỉnh skip connection
    if inputs.shape[-1] != filters:
        skip = Conv2D(filters, (1, 1), padding='same')(inputs)
    
    # Khối tích chập chính
    conv = Conv2D(filters, kernel_size, padding='same')(inputs)
    if batch_norm:
        conv = BatchNormalization()(conv)
    conv = LeakyReLU(negative_slope=0.2)(conv)
    
    conv = Conv2D(filters, kernel_size, padding='same')(conv)
    if batch_norm:
        conv = BatchNormalization()(conv)
    
    # Kết hợp với skip connection
    output = Add()([conv, skip])
    # nếu x < 0, thì x = 0.2 * x, ngược lại x = x
    output = LeakyReLU(negative_slope=0.2)(output)
    
    if dropout_rate > 0:
        output = Dropout(dropout_rate)(output)
    
    return output

# Sửa lại hàm self_attention để phù hợp với mixed precision
def self_attention(inputs, filters):
    batch_size, height, width, channels = inputs.shape
    
    # Xác định kiểu dữ liệu của input để sử dụng nhất quán
    input_dtype = inputs.dtype
    
    # Giảm kích thước kênh để tiết kiệm bộ nhớ
    reduced_channels = max(filters // 8, 16)  # c' = c // 8
    
    # Projection matrices
    # Cách đọc ma trận
    # ví dụ với f: [bs, h, w, c], ta sẽ đọc từ phải qua trái, nghĩa là:
    # ma trận f, có thành phần nhỏ nhất là ma trận [w, c], gọi là khối I
    # có h khối I như thế trong khối lớn hơn gọi là khối K
    # có bs khối K như thế trong ma trận lớn nhất gọi là ma trận F

    # thay vì nhìn tensor dưới dây bao gồm các khối (batch), mỗi khối 
    # bao gồm các channels khác nhau xếp chồng lên nhau theo phương ngang (như cách ta
    # hay nhìn convolution layer), ta sẽ nhìn nó theo phương vuông góc, tức là xem
    # với các ma trận kích thước bằng [rộng, channels], xếp chồng lên nhau height lần
    f = Conv2D(reduced_channels, 1, padding='same')(inputs)  # [bs, h, w, c'] ~ Query matrix
    g = Conv2D(reduced_channels, 1, padding='same')(inputs)  # [bs, h, w, c'] ~ Key matrix
    h = Conv2D(filters, 1, padding='same')(inputs)  # [bs, h, w, c] ~ Value matrix
    
    # Reshape và chuyển vị
    # Khi Reshape như dưới đây, chiều 0, tức là batch size, sẽ được giữ nguyên
    # Chiều sau cùng bằng reduce_channels, hoặc filters (giữ nguyên)
    # Chiều còn lại được tính toán tự động
    # Cụ thể, các khối I nằm trong một khối K sẽ được nối với nhau, với axis = 0, hay xếp chồng lên nhau
    f_flat = Reshape((-1, reduced_channels))(f)  # [bs, h*w, c'] - Query
    g_flat = Reshape((-1, reduced_channels))(g)  # [bs, h*w, c'] - Key
    h_flat = Reshape((-1, filters))(h)  # [bs, h*w, c] - Value
    
    # Nhân ma trận để tính attention map
    s = Lambda(
        lambda x: tf.matmul(x[0], tf.transpose(x[1], [0, 2, 1])),
        output_shape=(None, None)
    )([f_flat, g_flat])  # [bs, h*w, h*w]
    
    # Softmax để có attention weights
    beta = Activation('softmax')(s)
    
    # Áp dụng attention weights vào h
    o = Lambda(
        lambda x: tf.matmul(x[0], x[1]),
        output_shape=(None, filters)
    )([beta, h_flat])  # [bs, h*w, c]
    
    # Reshape lại về kích thước ban đầu
    o = Reshape((height, width, filters))(o)
    
    # Khởi tạo gamma với kiểu dữ liệu phù hợp với input
    gamma = tf.Variable(0.1, trainable=True, dtype=input_dtype)
    
    # Tính scaled attention - chuyển đổi kiểu dữ liệu cho phép toán
    scaled_attention = Lambda(
        # sclaled_attention = gamma * o
        lambda x: tf.cast(gamma, x.dtype) * x,
        output_shape=(height, width, filters)
    )(o)
    
    # Skip connection, tức là đang thay đổi input theo attention
    output = Add()([inputs, scaled_attention])
    
    return output


# Cải tiến: FiLM Conditioning - Feature-wise Linear Modulation
def film_conditioning(features, condition):
    # Lấy shape từ input features
    _, height, width, channels = features.shape
    
    # Chuẩn hóa đặc trưng trước khi áp dụng FiLM
    features_norm = LayerNormalization()(features)
    
    # MLP sâu hơn cho conditioning với kích hoạt phi tuyến
    hidden = Dense(channels * 2, activation='swish')(condition)
    hidden = Dense(channels * 2, activation='swish')(hidden)
    
    # Tách gamma và beta từ hidden state
    gamma_beta = Dense(channels * 2, activation=None)(hidden)
    gamma, beta = tf.split(gamma_beta, 2, axis=-1)
    
    # Reshape và scale gamma để ổn định hóa
    gamma = Reshape((1, 1, channels))(gamma)
    beta = Reshape((1, 1, channels))(beta)
    
    # Scale gamma với tanh để tránh giá trị quá lớn/nhỏ
    gamma = Lambda(lambda x: tf.nn.tanh(x) + 1.0)(gamma)  # giới hạn trong [0, 2]
    
    # Áp dụng modulation và thêm skip connection
    output = features_norm * gamma + beta
    output = Add()([output, features])  # skip connection
    
    return output

# Function to create an improved diffusion-ready U-Net
def create_diffusion_unet(img_size=128, img_channels=3, time_embed_dim=256, base_channels=32):
    # Image input
    inputs = Input(shape=(img_size, img_size, img_channels))
    
    # Time embedding input - cải thiện độ phân giải
    time_input = Input(shape=(1,))
    
    # Time embedding with improved sinusoidal positional encoding
    positions = tf.range(0, time_embed_dim // 2, dtype=tf.float32)
    frequencies = 1.0 / (10000.0**(2.0 * (positions / (time_embed_dim // 2))))
    
    def time_embedding(t):
        embedding = tf.cast(t, dtype=tf.float32)
        embedding = tf.reshape(embedding, [-1, 1])
        embedding = tf.matmul(embedding, frequencies[None, :])
        embedding = tf.concat([tf.sin(embedding), tf.cos(embedding)], axis=-1)
        embedding = tf.reshape(embedding, [-1, time_embed_dim])
        return embedding
    
    # Process time embedding with deeper network
    time_embed = Lambda(
        lambda t: time_embedding(t),
        output_shape=(time_embed_dim,)
    )(time_input)
    time_embed = Dense(time_embed_dim, activation="swish")(time_embed)
    time_embed = Dense(time_embed_dim, activation="swish")(time_embed)
    
    # Chuẩn hóa đầu vào
    x = Lambda(lambda x: (x / 255.0) * 2 - 1)(inputs)  # Normalize to [-1, 1]
    
    # Số kênh đặc trưng tăng dần (giảm từ gốc để phù hợp với GPU)
    channels = [base_channels, base_channels*2, base_channels*4, base_channels*8, base_channels*16]
    
    # --- ENCODER ---
    skips = []
    
    # Initial convolution
    x = Conv2D(base_channels, (3, 3), padding='same')(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    
    # Encoder blocks
    for i, ch in enumerate(channels):
        # Save skip connection
        skips.append(x)
        
        # Down blocks with residual connections (giảm số block)
        x = resnet_block(x, ch)
        
        # Inject time embedding via FiLM, truyền dense thay vì time_embed
        # để chỉnh cho condition có số kênh phù hợp
        x = film_conditioning(x, Dense(ch*2)(time_embed))
        
        # Add self-attention chỉ ở tầng thứ 2 (giảm số lượng self-attention)
        if i == 0:
            x = self_attention(x, ch)  # Thêm attention ở tầng đầu
        elif i == len(channels) - 1:
            x = self_attention(x, ch)  
            x = self_attention(x, ch)
            x = self_attention(x, ch)  # Thêm triple attention ở tầng cuối
        else:
            x = self_attention(x, ch)
            if i == len(channels) - 2:  # Tầng gần cuối
                x = self_attention(x, ch)  # Double attention
            
        # Downsample except for last block
        if i < len(channels) - 1:
            x = Conv2D(ch, (3, 3), strides=2, padding='same')(x)
            x = LeakyReLU(negative_slope=0.2)(x)
    
    # --- BOTTLENECK ---
    # Bottleneck with improved capacity
    x = resnet_block(x, channels[-1], dropout_rate=0.3)  # Giảm số kênh
    x = film_conditioning(x, Dense(channels[-1]*2)(time_embed))
    
    # --- DECODER ---
    # Decoder blocks
    for i, ch in enumerate(reversed(channels)):
        # Upscale features
        if i > 0:  # Không upscale cho lớp đầu tiên của decoder
            x = Conv2DTranspose(ch, (4, 4), strides=2, padding='same')(x)
            x = LeakyReLU(negative_slope=0.2)(x)
        
        # Concat with skip connection
        skip = skips[-(i+1)]
        x = concatenate([x, skip], axis=-1)
        
        # Residual blocks (giảm số block)
        x = resnet_block(x, ch, dropout_rate=0.1)
        
        # Inject time embedding for each scale
        x = film_conditioning(x, Dense(ch*2)(time_embed))
        
        # Decoder
        if i == 0:  # Tầng sâu nhất
            x = self_attention(x, ch)
            x = self_attention(x, ch)
            x = self_attention(x, ch)  # Triple attention
        elif i == len(channels) - 1:
            x = self_attention(x, ch)  # Thêm attention ở tầng cuối 
        else:
            x = self_attention(x, ch)
            if i == 1:  # Tầng gần nhất với tầng sâu nhất
                x = self_attention(x, ch)  # Double attention
    
    # Final processing block
    x = resnet_block(x, base_channels)
    x = Conv2D(base_channels, (3, 3), padding='same')(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    
    # Output layer - dự đoán nhiễu
    outputs = Conv2D(img_channels, (3, 3), padding='same', activation=None)(x)
    
    # Create the diffusion model
    diffusion_model = keras.Model(inputs=[inputs, time_input], outputs=outputs)
    
    # Compile with MSE loss
    diffusion_model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=1e-4, 
            weight_decay=1e-4,
            beta_1=0.9,
            beta_2=0.99
        ),
        loss='mean_squared_error'
    )
    
    return diffusion_model


model = create_diffusion_unet()