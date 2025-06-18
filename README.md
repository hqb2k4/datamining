# Emotion Analysis Web Application

Ứng dụng phân tích cảm xúc từ văn bản sử dụng mô hình LSTM với TF-IDF.

## Cấu trúc thư mục

./web/
    static/                # Chứa các tệp tĩnh
        css/
            style.css
        js/
            script.js
    templates/             # Chứa các tệp HTML
        index.html
    app.py                 # Ứng dụng Flask
    requirements.txt       # Thư viện cần thiết
    Dockerfile             # Cấu hình Docker
    docker-compose.yml     # Cấu hình Docker Compose
    README.md              # Hướng dẫn sử dụng

Tính năng
- Phân tích cảm xúc văn bản: Nhận diện 8 loại cảm xúc (anger, disgust, fear, joy, neutral, sadness, shame, surprise)
- Giao diện web thân thiện: Interface đơn giản, dễ sử dụng
- Biểu đồ trực quan: Hiển thị xác suất của từng cảm xúc dưới dạng biểu đồ cột
- Containerized: Hỗ trợ Docker để triển khai dễ dàng
- Real-time processing: Xử lý và trả kết quả ngay lập tức

Cài đặt và chạy
# Phương pháp 1: Chạy trực tiếp (Development):
    cd web
    pip install -r requirements.txt

- Chạy ứng dụng:
    python app.py

- Truy cập ứng dụng:
    http://localhost:5000

# Phương pháp 2: Sử dụng Docker (Production)

- Build và chạy với Docker Compose:
    cd web
    docker-compose up --build

- Truy cập ứng dụng:
    http://localhost:5000

- Dừng ứng dụng:
    docker-compose down