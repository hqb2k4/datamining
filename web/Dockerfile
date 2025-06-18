FROM python:3.9-slim

WORKDIR /app

# Copy requirements và cài đặt
COPY requirements.txt .
RUN pip install -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Copy ứng dụng
COPY . .

# Tạo thư mục model
RUN mkdir -p /model

EXPOSE 5000
CMD ["python", "app.py"]