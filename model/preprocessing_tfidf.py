import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle

# Định nghĩa stop words và stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

data = pd.read_csv('emotion_dataset_raw.csv')
print("Data ban đầu:\n", data)

# Xử lý khoảng trắng liên tiếp
data['Whitespace_Count'] = data['Text'].str.count(r'\s{2,}')
data_with_whitespace = data[data['Whitespace_Count'] > 1]

print("Data với nhiều khoảng trắng:\n", data_with_whitespace)

# Hàm để chuyển đổi chữ hoa thành chữ thường và thay thế các ký tự không mong muốn
def lower_case(x):
    return x.lower()

data['Text'] = data['Text'].apply(lower_case)
data['Text'] = data['Text'].str.replace(r'\d+', '', regex=True) 

# Xóa các tag người dùng (@username)
data['Text'] = data['Text'].str.replace(r'@\w+', '', regex=True)

data['Text'] = data['Text'].str.replace(r'[^\w\s]', '', regex=True)  

data['Text'] = data['Text'].str.strip()
data['Text'] = data['Text'].str.replace(r'\s+', ' ', regex=True)

print("10 dòng đầu tiên sau khi xử lý chữ hoa, thường và các ký tự không mong muốn:\n",data.head(10))

# Hàm xử lý tokenization, loại bỏ stop words và stem từ
def tokenize_remove_stops(x):
    # Sử dụng phương pháp split đơn giản thay vì word_tokenize để tránh lỗi
    tokens = x.split()
    # Hoặc sử dụng word_tokenize từ nltk.tokenize thay vì nltk.word_tokenize
    # tokens = word_tokenize(x)
    
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    tokens = [token for token in tokens if token.strip()]
    return ' '.join(tokens)

data['Text'] = data['Text'].apply(tokenize_remove_stops)
print("10 dòng đầu tiên sau khi xử lý tokenization, loại bỏ stop words và stem từ:\n", data.head(10))

# Tiếp tục với phần còn lại của mã...
# Xử lý giá trị rỗng
empty_rows = data[(data['Text'].str.strip() == '') | (data['Emotion'].str.strip() == '')]
print(empty_rows)
print(f"Số dòng có giá trị là khoảng trắng cả cột 'Text' hoặc 'Emotion': {empty_rows.shape[0]}")

empty_rows_indices = empty_rows.index.tolist()
print(f"Vị trí của các dòng có giá trị là khoảng trắng cả cột 'Text' hoặc 'Emotion': {empty_rows_indices}")
data = data.drop(empty_rows_indices)

print(data.iloc[74:80])

data.drop(columns=['Whitespace_Count'], inplace=True)

# Xử lý giá trị trùng lặp
duplicate_rows = data[data.duplicated()]
print(f"Số dòng trùng lặp: {duplicate_rows.shape[0]}")

data = data.drop_duplicates()

duplicate_rows_after = data[data.duplicated()]
print(f"Số dòng trùng lặp sau khi xóa: {duplicate_rows_after.shape[0]}")

print("\nBắt đầu xử lý TF-IDF...")

# Khởi tạo và huấn luyện TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,  # Số lượng từ tối đa
    min_df=2,          # Tần suất xuất hiện tối thiểu trong tài liệu
    max_df=0.95,       # Tần suất xuất hiện tối đa trong tài liệu
    ngram_range=(1, 2) # Sử dụng cả unigram và bigram
)

# Chuyển đổi văn bản thành vector TF-IDF
print("Đang chuyển đổi văn bản thành vector TF-IDF...")
tfidf_matrix = tfidf_vectorizer.fit_transform(data['Text'])

# Lưu vectorizer để sử dụng sau này
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

print("\nBắt đầu xử lý nhãn...")

# Chuyển đổi nhãn thành số (0-7)
label_encoder = LabelEncoder()
data['Emotion'] = label_encoder.fit_transform(data['Emotion'])

# Lưu encoder để sử dụng sau này
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# In thông tin về các nhãn
print("\nThông tin về các nhãn:")
print("Số lượng nhãn duy nhất:", len(label_encoder.classes_))
print("Các nhãn và mã số tương ứng:")
for i, label in enumerate(label_encoder.classes_):
    print(f"{label}: {i}")

print("\nVí dụ về 5 dòng đầu tiên sau khi mã hóa nhãn:")
print(data[['Text', 'Emotion']].head())

# Lưu dữ liệu đã xử lý vào file emotion_dataset_cleaned_tfidf.csv
data.to_csv('emotion_dataset_cleaned_tfidf.csv', index=False)

# In ra thông tin về dữ liệu đã xử lý
print("\nDữ liệu đã được xử lý và lưu vào file emotion_dataset_cleaned_tfidf.csv")
print(data.info())
print("\nKích thước vector TF-IDF:", tfidf_matrix.shape)
print("\nVí dụ về vector TF-IDF của văn bản đầu tiên:", tfidf_matrix[0].toarray()[0][:5], "...")