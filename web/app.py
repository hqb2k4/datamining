from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import pickle
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
import os
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

app = Flask(__name__)

# Đường dẫn tới thư mục model - xử lý cả Docker và chạy trực tiếp
def get_model_directory():
    # Kiểm tra biến môi trường MODEL_DIR (cho Docker)
    env_model_dir = os.environ.get('MODEL_DIR')
    if env_model_dir and os.path.exists(env_model_dir):
        return env_model_dir
    
    # Đường dẫn tương đối cho Windows/Linux (chạy trực tiếp)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    relative_model_dir = os.path.join(current_dir, '..', 'model')
    if os.path.exists(relative_model_dir):
        return os.path.abspath(relative_model_dir)
    
    # Thử các đường dẫn khác
    alternative_paths = [
        '/model',  # Docker
        '../model',  # Relative
        './model',   # Current directory
        os.path.join(os.path.dirname(current_dir), 'model')  # Parent directory
    ]
    
    for path in alternative_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            return abs_path
    
    return None

model_dir = get_model_directory()
logger.info(f"Model directory path: {model_dir}")

# Load model và các thành phần tiền xử lý
logger.info("Loading model and preprocessing components...")
try:
    if not model_dir:
        raise FileNotFoundError("Model directory not found in any expected location")
    
    model_path = os.path.join(model_dir, 'emotion_classifier_tfidf_lstm.h5')
    vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
    encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
    
    logger.info(f"Model path: {model_path}")
    logger.info(f"Vectorizer path: {vectorizer_path}")
    logger.info(f"Encoder path: {encoder_path}")
    
    # Kiểm tra sự tồn tại của thư mục và các file
    logger.info(f"Model directory exists: {os.path.exists(model_dir)}")
    if os.path.exists(model_dir):
        logger.info(f"Files in model directory: {os.listdir(model_dir)}")
    
    logger.info(f"Model file exists: {os.path.exists(model_path)}")
    logger.info(f"Vectorizer file exists: {os.path.exists(vectorizer_path)}")
    logger.info(f"Encoder file exists: {os.path.exists(encoder_path)}")
    
    if not all([os.path.exists(model_path), os.path.exists(vectorizer_path), os.path.exists(encoder_path)]):
        raise FileNotFoundError("One or more model files are missing")
    
    model = tf.keras.models.load_model(model_path)
    logger.info("Model loaded successfully")

    with open(vectorizer_path, 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    logger.info("Vectorizer loaded successfully")

    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    logger.info("Label encoder loaded successfully")
    
    logger.info("All components loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model or components: {e}", exc_info=True)
    raise

# Định nghĩa các hàm tiền xử lý
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    # Chuyển thành chữ thường
    text = text.lower()
    
    # Xóa số
    text = re.sub(r'\d+', '', text)
    
    # Xóa các tag người dùng
    text = re.sub(r'@\w+', '', text)
    
    # Xóa dấu câu
    text = re.sub(r'[^\w\s]', '', text)
    
    # Xóa khoảng trắng thừa
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    
    # Tokenize, loại bỏ stopwords và stem
    tokens = text.split()
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    tokens = [token for token in tokens if token.strip()]
    
    return ' '.join(tokens)

def predict_emotion(text):
    # Tiền xử lý văn bản
    processed_text = preprocess_text(text)
    logger.info(f"Preprocessed text: {processed_text}")
    
    # Chuyển đổi thành vector TF-IDF
    text_vector = tfidf_vectorizer.transform([processed_text]).toarray()
    
    # Reshape cho LSTM
    text_vector = text_vector.reshape(1, 1, -1)
    
    # Dự đoán
    prediction = model.predict(text_vector, verbose=0)
    predicted_class = np.argmax(prediction[0])
    
    # Lấy tên cảm xúc
    emotion = label_encoder.inverse_transform([predicted_class])[0]
    logger.info(f"Predicted emotion: {emotion}")
    
    # Lấy xác suất cho mỗi cảm xúc
    emotion_probs = dict(zip(label_encoder.classes_, prediction[0].tolist()))
    
    return emotion, emotion_probs

@app.route('/')
def index():
    logger.info("Index page requested")
    return render_template('index.html')

@app.route('/lstm')
def lstm():
    logger.info("LSTM information page requested")
    return render_template('lstm.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    logger.info("Analysis request received")
    data = request.json
    text = data.get('text', '')
    
    if not text:
        logger.warning("Empty text received")
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        logger.info(f"Processing text: {text[:50]}{'...' if len(text) > 50 else ''}")
        emotion, probabilities = predict_emotion(text)
        
        response = {
            'emotion': emotion,
            'probabilities': probabilities
        }
        logger.info(f"Analysis completed: {emotion}")
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        return jsonify({'error': 'An error occurred during analysis'}), 500

if __name__ == '__main__':
    logger.info("Starting application...")
    # Sử dụng host 0.0.0.0 để đảm bảo có thể truy cập từ bên ngoài container
    app.run(debug=True, host='0.0.0.0', port=5000)