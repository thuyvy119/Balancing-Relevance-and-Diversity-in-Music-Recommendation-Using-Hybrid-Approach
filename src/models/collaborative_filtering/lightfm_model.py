import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset

class LightFMRecommender:
    def __init__(self, loss='warp', learning_rate= 0.05, item_alpha=1e-6, user_alpha=1e-6):
        self.model = LightFM(
            loss=loss,
            learning_rate=learning_rate,
            item_alpha=item_alpha,
            user_alpha=user_alpha
        )
        self.dataset = None
        self.interactions = None
        self.user_feature_matrix = None

    def fit(self, data, epochs, num_threads, user_col='uid', item_col='iid', rating_col='label'):
        """Train the model with historical data"""
        # Initialize dataset
        self.dataset = Dataset()
        self.dataset.fit(data[user_col], data[item_col])
        
        # Build interactions
        self.interactions, weights = self.dataset.build_interactions(
            (row[user_col], row[item_col], row[rating_col])
            for _, row in data.iterrows()
        )
        
        # Add user features (historical items)
        # user_features = []
        # for _, row in df.iterrows():
        #     features = [f"prev_{iid}" for iid in row['his']]
        #     user_features.append((row[user_col], features))
        # self.user_feature_matrix = self.dataset.build_user_features(user_features)
        
        # train
        self.model.fit(
            self.interactions,
            user_features=self.user_feature_matrix,
            epochs=epochs,
            num_threads=num_threads,
            sample_weight=weights
        )
        
        return self.model, self.dataset, self.interactions

    def recommend(self, data, user_id, k, filter_rated=True):
        """Generate top-k recommendations"""
        if self.dataset is None:
            raise ValueError("Model is not trained yet - call fit() first")
        
        n_items = self.dataset.interactions_shape()[1]
        scores = self.model.predict(
            user_ids = np.full(n_items, user_id),
            item_ids = p.arange(n_items),
        )
        #  user_features=self.user_feature_matrix
        
        # filter rated items
        rated_items = set(data[data['uid'] == user_id]['iid'])
        scores[list(rated_items)] = -np.inf

        top_items = np.argsort(-scores)[:k]
        
        items_lookup = {
        iid: {
            'iid': iid,
            'title': title,
            'categories': cat,
        }
        for iid, title, cat in zip(data['iid'], data['title'], data['categories'])
    }
        
        recs = [
        {
            **items_lookup[i],
            'score': float(scores[i])
        }
        for i in top_items if i in items_lookup
    ]
        
        return recs
    
    
if __name__ == "__main__":
    data = pd.read_csv('data/train.csv')
    model = LightFMRecommender()
    model.fit(data, epochs=50, num_threads=4)
    recs = model.recommend(data, user_id=1, k=10)
    print(recs)