import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pickle
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer

# Đọc dữ liệu đã xử lý
print("Đang đọc dữ liệu...")
data = pd.read_csv('emotion_dataset_cleaned_tfidf.csv')

# Đọc TF-IDF vectorizer đã lưu
print("Đang đọc TF-IDF vectorizer...")
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Đọc label encoder
print("Đang đọc label encoder...")
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Chuyển đổi văn bản thành vector TF-IDF
print("Đang chuyển đổi văn bản thành vector TF-IDF...")
X = tfidf_vectorizer.transform(data['Text']).toarray()
y = data['Emotion'].values

# Chia tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape dữ liệu cho LSTM (samples, time steps, features)
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Xây dựng mô hình LSTM
print("Đang xây dựng mô hình LSTM...")
model = Sequential([
    LSTM(128, input_shape=(1, X_train.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')  # Số lớp đầu ra bằng số cảm xúc
])

# Biên dịch mô hình
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# In thông tin mô hình
model.summary()

# Huấn luyện mô hình
print("\nBắt đầu huấn luyện mô hình...")
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# Đánh giá mô hình
print("\nĐánh giá mô hình trên tập test:")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Độ chính xác trên tập test: {test_accuracy:.4f}")
print(f"Loss trên tập test: {test_loss:.4f}")

# Lưu mô hình
print("\nĐang lưu mô hình...")
model.save('emotion_classifier_tfidf_lstm.h5')

# Lưu lịch sử huấn luyện
with open('training_history_tfidf_lstm.pkl', 'wb') as f:
    pickle.dump(history.history, f)

print("\nMô hình đã được lưu thành công!")

# Hàm dự đoán cảm xúc
def predict_emotion(text):
    # Chuyển đổi văn bản thành vector TF-IDF
    text_vector = tfidf_vectorizer.transform([text]).toarray()
    
    # Reshape input cho LSTM
    text_vector = text_vector.reshape(1, 1, -1)
    
    # Dự đoán
    prediction = model.predict(text_vector, verbose=0)
    predicted_class = np.argmax(prediction[0])
    
    # Chuyển đổi nhãn số thành tên cảm xúc
    emotion = label_encoder.inverse_transform([predicted_class])[0]
    
    # Lấy xác suất cho mỗi cảm xúc
    emotion_probs = dict(zip(label_encoder.classes_, prediction[0]))
    
    return emotion, emotion_probs

# Ví dụ dự đoán
print("\nThử dự đoán một số mẫu từ tập test:")
test_texts = [
    "I am so happy today!",
    "This makes me really angry",
    "I feel sad about what happened"
]

for text in test_texts:
    emotion, probs = predict_emotion(text)
    print(f"\nVăn bản: {text}")
    print(f"Cảm xúc dự đoán: {emotion}")
    print("Xác suất cho mỗi cảm xúc:")
    for emo, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        print(f"{emo}: {prob:.4f}") 