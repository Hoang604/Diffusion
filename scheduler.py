import tensorflow as tf
import numpy as np

class CosineAnnealingScheduler:
    def __init__(self, initial_lr, min_lr, total_epochs):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs
        
    def __call__(self, epoch):
        # Cosine annealing schedule
        cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch / self.total_epochs))
        return self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay

class CustomLRSchedulerCallback(tf.keras.callbacks.Callback):
    def __init__(self, scheduler, optimizer=None):
        super(CustomLRSchedulerCallback, self).__init__()
        self.scheduler = scheduler
        self._optimizer = optimizer  # Lưu trữ optimizer riêng
    
    def on_epoch_begin(self, epoch, logs=None):
        # Set the new learning rate
        new_lr = self.scheduler(epoch)
        
        # Lấy optimizer từ các nguồn khác nhau
        optimizer = None
        
        # Cách 1: Sử dụng optimizer đã truyền vào
        if self._optimizer is not None:
            optimizer = self._optimizer
        
        # Cách 2: Lấy từ model nếu có
        elif hasattr(self, 'model') and self.model is not None and hasattr(self.model, 'optimizer'):
            optimizer = self.model.optimizer
        
        if optimizer is not None:
            try:
                # Cách 1: Sử dụng _set_hyper (cách an toàn nhất)
                if hasattr(optimizer, '_set_hyper'):
                    optimizer._set_hyper('learning_rate', new_lr)
                    print(f"\nLearning rate for epoch {epoch+1} set to: {new_lr:.6f}")
                # Cách 2: Sử dụng set_value với kiểu dữ liệu tương thích
                elif hasattr(optimizer, 'learning_rate'):
                    # Lấy kiểu dữ liệu của learning_rate
                    lr_var = optimizer.learning_rate
                    # Chuyển đổi giá trị mới sang kiểu dữ liệu phù hợp
                    new_lr_value = tf.cast(new_lr, lr_var.dtype)
                    tf.keras.backend.set_value(optimizer.learning_rate, new_lr_value)
                    print(f"\nLearning rate for epoch {epoch+1} set to: {new_lr:.6f}")
                # Cách 3: Thử với thuộc tính lr (cho phiên bản cũ)
                elif hasattr(optimizer, 'lr'):
                    lr_var = optimizer.lr
                    new_lr_value = tf.cast(new_lr, lr_var.dtype)
                    tf.keras.backend.set_value(optimizer.lr, new_lr_value)
                    print(f"\nLearning rate for epoch {epoch+1} set to: {new_lr:.6f}")
                else:
                    print(f"\nWarning: Could not set learning rate - appropriate attribute not found")
            except Exception as e:
                print(f"\nError setting learning rate: {e}")
                # Phương pháp dự phòng - tạo mới optimizer với learning rate mới
                try:
                    if hasattr(optimizer, 'get_config'):
                        config = optimizer.get_config()
                        # Cập nhật learning_rate trong config
                        if 'learning_rate' in config:
                            config['learning_rate'] = new_lr
                        elif 'lr' in config:
                            config['lr'] = new_lr
                        # Tạo optimizer mới với config mới
                        # self._optimizer = type(optimizer).from_config(config)
                        print(f"\nCreated new optimizer with learning rate: {new_lr:.6f}")
                except Exception as e2:
                    print(f"\nFailed to recreate optimizer: {e2}")
        else:
            print(f"\nWarning: Cannot set learning rate - optimizer not found")