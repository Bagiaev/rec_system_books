import pandas as pd
import pickle
from pathlib import Path
from typing import Optional
import streamlit as st
# Пример класса HybridRecommender

class HybridRecommender:
    def __init__(self, books_df: pd.DataFrame):
        self.books_df = books_df
        # Возможно, вам нужно добавить другие модели или параметры для гибридного подхода

    def recommend(self, user_id: int, book_title: str, top_n: int = 5) -> pd.DataFrame:
        """
        Возвращает гибридные рекомендации на основе user_id и названия книги.

        Args:
            user_id (int): ID пользователя.
            book_title (str): Название книги для контентной фильтрации.
            top_n (int): Количество рекомендаций (по умолчанию 5).

        Returns:
            pd.DataFrame: Датафрейм с рекомендациями.
        """
        # Для примера просто будем объединять контентную фильтрацию и коллаборативную фильтрацию

        # 1. Рекомендации на основе контентной фильтрации (например, по названию книги)
        content_based_recs = self.get_content_based_recommendations(book_title, top_n)
        
        # 2. Рекомендации на основе коллаборативной фильтрации (например, по user_id)
        user_based_recs = self.get_user_based_recommendations(user_id, top_n)

        # Объединяем рекомендации
        hybrid_recs = pd.concat([content_based_recs, user_based_recs]).drop_duplicates(subset=['title']).head(top_n)
        
        return hybrid_recs

    def get_user_based_recommendations(self, user_id: int, top_n: int) -> pd.DataFrame:
        """
        Возвращает рекомендации для пользователя на основе коллаборативной фильтрации.

        Args:
            user_id (int): ID пользователя.
            top_n (int): Количество рекомендаций.

        Returns:
            pd.DataFrame: Рекомендации на основе пользователей.
        """
        # Здесь можно использовать вашу модель коллаборативной фильтрации (например, SVD)
        # Для примера просто выберем случайные книги
        user_recs = self.books_df.sample(n=top_n)
        return user_recs[['title', 'author', 'year', 'image_m']]

    def get_content_based_recommendations(self, title: str, top_n: int) -> pd.DataFrame:
        """
        Возвращает рекомендации на основе контентной фильтрации (по названию книги).

        Args:
            title (str): Название книги для контентной фильтрации.
            top_n (int): Количество рекомендаций.

        Returns:
            pd.DataFrame: Рекомендации на основе контентной фильтрации.
        """
        # Для контентной фильтрации просто выберем книги с похожими названиями
        similar_books = self.books_df[self.books_df['title'].str.contains(title, case=False, na=False)]
        return similar_books.head(top_n)
    

test_books = pd.DataFrame({
    'title': [
        '1984', 'Animal Farm', 'Harry Potter and the Philosopher\'s Stone',
        'The Great Gatsby', 'To Kill a Mockingbird', '1984: Special Edition',
        'Harry Potter and the Chamber of Secrets', 'George Orwell Collection'
    ],
    'author': [
        'George Orwell', 'George Orwell', 'J.K. Rowling',
        'F. Scott Fitzgerald', 'Harper Lee', 'George Orwell',
        'J.K. Rowling', 'George Orwell'
    ],
    'year': [1949, 1945, 1997, 1925, 1960, 2017, 1998, 2020],
    'image_m': ['url1', 'url2', 'url3', 'url4', 'url5', 'url6', 'url7', 'url8']
})
hybrid_model = HybridRecommender(test_books)
print("\n=== Тест гибридных рекомендаций ===")
hybrid_recs = hybrid_model.recommend(345, "1984", 4)
print(hybrid_recs[['title', 'author']])