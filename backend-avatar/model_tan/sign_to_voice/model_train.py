# model_train.py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Toy training script: makes a model that maps sequences of keypoints -> 5 classes
def make_dummy_data(num_samples=200, seq_len=20, keypoint_dim=63):
    X = np.random.randn(num_samples, seq_len, keypoint_dim).astype(np.float32)
    y = np.random.randint(0, 5, size=(num_samples,))
    return X, y

def build_model(seq_len=20, keypoint_dim=63, num_classes=5):
    model = Sequential([
        LSTM(128, return_sequences=False, input_shape=(seq_len, keypoint_dim)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    seq_len = 20
    keypoint_dim = 21*3  # MediaPipe hands single-hand flattened (21 landmarks * 3 coords)
    num_classes = 5
    X, y = make_dummy_data(500, seq_len, keypoint_dim)
    model = build_model(seq_len, keypoint_dim, num_classes)
    model.fit(X, y, epochs=4, batch_size=32, validation_split=0.1)
    model.save("sign_model.h5")
    print("Saved sign_model.h5 (dummy model). Replace with your trained model when ready.")
