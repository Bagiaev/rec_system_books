import pandas as pd
import numpy as np
from scipy.sparse import load_npz
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentRecommender:
    def __init__(self, 
                 books_path='data/clean_books.csv',
                 tfidf_vectorizer_path='models/tfidf_vectorizer.pkl',
                 tfidf_matrix_path='models/tfidf_matrix.npz',
                 top_indices_path='models/top_5_indices.npy',
                 top_scores_path='models/top_5_scores.npy',
                 book_indices_path='models/book_indices.pkl'):
        
        # Загрузка данных с обработкой дубликатов
        self.books = pd.read_csv(books_path).drop_duplicates(subset=['title'])
        
        # Загрузка моделей с обработкой версий
        try:
            with open(tfidf_vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
        except Exception as e:
            print(f"Ошибка загрузки векторайзера: {e}. Инициализируем новый.")
            self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            
        self.tfidf_matrix = load_npz(tfidf_matrix_path)
        self.top_indices = np.load(top_indices_path)
        self.top_scores = np.load(top_scores_path)
        
        with open(book_indices_path, 'rb') as f:
            self.book_indices = pickle.load(f)

    def get_recommendations(self, book_title, top_n=5):
        book_title_lower = book_title.lower()
        
        if book_title_lower not in self.book_indices:
            return f"Книга '{book_title}' не найдена."
        
        idx = self.book_indices[book_title_lower]
        recommended_indices = self.top_indices[idx][:top_n]
        recommended_scores = self.top_scores[idx][:top_n]
        
        recommendations = self.books.iloc[recommended_indices][['title', 'author', 'year']]
        recommendations['similarity'] = recommended_scores
        
        # Удаление дубликатов и самой книги из рекомендаций
        recommendations = recommendations[
            (recommendations['title'].str.lower() != book_title_lower)
        ].drop_duplicates(subset=['title'])
        
        return recommendations.reset_index(drop=True)

if __name__ == '__main__':
    recommender = ContentRecommender()
    test_books = [
        "Harry Potter and the Sorcerer's Stone",
        "The Great Gatsby",
        "Nonexistent Book"
    ]
    
    for book in test_books:
        print(f"\nРекомендации для '{book}':")
        recs = recommender.get_recommendations(book)
        print(recs if isinstance(recs, pd.DataFrame) else recs)