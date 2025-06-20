# database/populate_data.py

import sys
import io
import pandas as pd
from sqlalchemy import create_engine

# Устанавливаем UTF-8 как кодировку вывода для корректного отображения символов
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Загрузка данных
books = pd.read_csv('data/clean_books.csv')
ratings = pd.read_csv('data/clean_ratings.csv')
users = pd.read_csv('data/clean_users.csv')

# Оставляем только те оценки, где книги существуют
# в процессе записи в бд данных мы столкнулись с проблемой
# где данные с isbn в ratings.csv отсутствуют в файле books.csv
# поэтому мы удаляем некоторые данные. Это около 8 процентов от всех данных
valid_ratings = ratings[ratings['isbn'].isin(books['isbn'])]


# Подключение к PostgreSQL
# ЗАМЕНИ на свои значения: user, password, host и порт
engine = create_engine('postgresql://postgres:Ars8095835@localhost:5433/bookdb')

# Загрузка в базу данных
users.to_sql('users', engine, if_exists='append', index=False)
books.to_sql('books', engine, if_exists='append', index=False)
#ratings.to_sql('ratings', engine, if_exists='append', index=False)
valid_ratings.to_sql('ratings', engine, if_exists='append', index=False)

print("✅ Данные успешно загружены в PostgreSQL.")
