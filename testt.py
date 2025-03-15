import tensorflow as tf

# Ví dụ 1: Thu thập các phần tử từ một vector
params1 = tf.constant([10, 20, 30, 40, 50])
indices1 = tf.constant([1, 3, 0])
output1 = tf.gather(params1, indices1)
print(output1)  # Output: tf.Tensor([20 40 10], shape=(3,), dtype=int32)

# Ví dụ 2: Thu thập các hàng từ một ma trận
params2 = tf.constant([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])
indices2 = tf.constant([0, 2])
output2 = tf.gather(params2, indices2)
print(output2)
# Output:
# tf.Tensor(
# [[1 2 3]
#  [7 8 9]], shape=(2, 3), dtype=int32)

# Ví dụ 3: Thu thập các cột từ một ma trận (axis=1)
params3 = tf.constant([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])
indices3 = tf.constant([0, 2])
output3 = tf.gather(params3, indices3, axis=1)
print(output3)
# Output:
# tf.Tensor(
# [[1 3]
#  [4 6]
#  [7 9]], shape=(3, 2), dtype=int32)

# Ví dụ 4: Sử dụng batch_dims
params4 = tf.constant([[[1, 2], [3, 4]],
                       [[5, 6], [7, 8]]])  # Shape: (2, 2, 2)
indices4 = tf.constant([1, 0])          # Shape: (2,)
output4 = tf.gather(params4, indices4, batch_dims=1)
print(output4)
# Output:
# tf.Tensor(
#  [[[3 4]
#   [1 2]]
#
#  [[7 8]
#   [5 6]]], shape=(2, 2, 2), dtype=int32)
# Trong ví dụ này batch_dims = 1, có nghĩa là chiều đầu (batch_size = 2) sẽ được giữ nguyên
# và gather sẽ áp dụng cho chiều thứ hai, và các phần tử có indices = [1, 0] ở chiều thứ hai sẽ được gather

# Ví dụ 5: indices là 1 tensor 2D
params5 = tf.constant([10, 20, 30, 40, 50])
indices5 = tf.constant([[0, 2], [1, 3]]) # shape (2, 2)
output5 = tf.gather(params5, indices5)
print(output5)
# Output: tf.Tensor(
# [[10 30]
#  [20 40]], shape=(2, 2), dtype=int32)

# Ví dụ 6: Minh họa trong embedding lookup.
# Rất phổ biến trong NLP, chẳng hạn, bạn có một từ điển các vector embedding.
# Bạn có câu "cat sat on mat" được biểu diễn bằng index: [2, 5, 8, 1].
# Bạn muốn lấy embedding của những từ này.
embeddings = tf.constant([
    [0.1, 0.2, 0.3],  # 0
    [0.4, 0.5, 0.6],  # 1
    [0.7, 0.8, 0.9],  # 2: "cat"
    [1.0, 1.1, 1.2],  # 3
    [1.3, 1.4, 1.5],  # 4
    [1.6, 1.7, 1.8],  # 5: "sat"
    [1.9, 2.0, 2.1],  # 6
    [2.2, 2.3, 2.4],  # 7
    [2.5, 2.6, 2.7],  # 8: "on"
    [2.8, 2.9, 3.0]   # 9: "mat"
    ])

word_indices = tf.constant([2, 5, 8, 9])
word_embeddings = tf.gather(embeddings, word_indices)
print(word_embeddings)
# Output:
# tf.Tensor(
# [[0.7 0.8 0.9]
#  [1.6 1.7 1.8]
#  [2.5 2.6 2.7]
#  [2.8 2.9 3. ]], shape=(4, 3), dtype=float32)