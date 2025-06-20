# recommenderSVD.py

import pandas as pd
import numpy as np
import pickle

class SimpleSVD:
    def __init__(self, weights):
        self.pu = weights['pu']
        self.qi = weights['qi']
        self.bu = weights['bu']
        self.bi = weights['bi']
        # Create mappings from user/item IDs to indices
        self.user_mapping = weights.get('user_mapping', {})
        self.item_mapping = weights.get('item_mapping', {})
    
    def predict(self, uid, iid):
        # Convert user and item IDs to indices
        try:
            user_idx = self.user_mapping[uid]
            item_idx = self.item_mapping[iid]
            return self.bu[user_idx] + self.bi[item_idx] + np.dot(self.pu[user_idx], self.qi[item_idx])
        except KeyError:
            # Return a default prediction if user/item not found
            return np.mean(self.bu) + np.mean(self.bi)

class SVDRecommender:
    def __init__(self, ratings_path='data/clean_ratings.csv', 
                 books_path='data/clean_books.csv', 
                 data_path='models/svd_weights.npz'):
        self.ratings_df = pd.read_csv(ratings_path)
        self.books_df = pd.read_csv(books_path)
        self.model = None
        self.load_model(data_path)
        
    def load_model(self, data_path):
        data = np.load(data_path, allow_pickle=True)
        self.model = SimpleSVD(data)
        
    def recommend_books(self, user_id, top_n=10):
        if self.model is None:
            raise Exception("Model not loaded. Please load the model first.")

        # Get all unique books
        all_books = self.books_df['isbn'].unique()
        
        # Get books the user has already rated
        rated_books = self.ratings_df[self.ratings_df['user_id'] == user_id]['isbn'].tolist()
        
        # Get books to predict (those not rated by the user)
        books_to_predict = [isbn for isbn in all_books if isbn not in rated_books]
        
        # Make predictions
        predictions = []
        for isbn in books_to_predict:
            try:
                pred = self.model.predict(user_id, isbn)
                predictions.append((isbn, pred))
            except:
                continue
        
        # Sort predictions by estimated rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N recommendations
        top_books = predictions[:top_n]
        
        # Prepare results dataframe
        result_df = self.books_df[self.books_df['isbn'].isin([isbn for isbn, _ in top_books])]
        result_df = result_df[['title', 'author', 'year']]
        
        return result_df.reset_index(drop=True)

# Example usage
if __name__ == '__main__':
    recommender = SVDRecommender()
    
    # Test with a sample user
    try:
        user_id = 276729  # Example user ID
        recommendations = recommender.recommend_books(user_id=user_id, top_n=5)
        print(f"Recommendations for user {user_id}:")
        print(recommendations)
    except Exception as e:
        print(f"Error: {e}")