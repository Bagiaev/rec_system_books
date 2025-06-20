# database/prepare_data.py
import pandas as pd

# Загрузка данных
books = pd.read_csv('data/Books.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
ratings = pd.read_csv('data/Ratings.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
users = pd.read_csv('data/Users.csv', sep=';', encoding='latin-1', on_bad_lines='skip')

# Переименуем столбцы для удобства
books.columns = ['isbn', 'title', 'author', 'year', 'publisher', 'image_s', 'image_m', 'image_l']
ratings.columns = ['user_id', 'isbn', 'rating']
users.columns = ['user_id', 'location', 'age']

# Очистка данных
books = books.dropna(subset=['isbn', 'title'])
ratings = ratings[ratings['rating'] > 0]  # Уберем рейтинги = 0
users = users.dropna(subset=['user_id'])

# Сохраняем очищенные версии
books.to_csv('data/clean_books.csv', index=False)
ratings.to_csv('data/clean_ratings.csv', index=False)
users.to_csv('data/clean_users.csv', index=False)
