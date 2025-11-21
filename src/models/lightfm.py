from lightfm import LightFM
from lightfm.data import Dataset
from tqdm import tqdm
import numpy as np

class LightFMTrainer:
    def __init__(self):
        self.model = None
        self.dataset = None
        self.fitted_data = None
        
    def fit_and_train(self, data, learning_rate=0.01, epochs=50):
        dataset = Dataset()
        dataset.fit(users=data['uid'].unique(), items=data['iid'].unique())
        
        interactions, weights = dataset.build_interactions(
            ((row['uid'], row['iid'], row['label']) for _, row in data.iterrows()))
        
        model = LightFM(loss='warp', learning_rate=learning_rate, 
                        item_alpha=1e-6, user_alpha=1e-6)
        model.fit(interactions, epochs=epochs, num_threads=4)
        
        self.model = model
        self.dataset = dataset
        self.fitted_data = data
        self.interactions = interactions
        self.weights = weights
        
        return model, dataset, interactions, data, weights
    
    def get_top_k(self, k=50):
        user_id_map, _, item_id_map, _ = self.dataset.mapping()
        index_to_item_id = {v: k for k, v in item_id_map.items()}
        
        all_user_ids = self.fitted_data['uid'].unique()
        res = {}
        
        for user in tqdm(all_user_ids):
            if user not in user_id_map:
                continue
                
            user_index = user_id_map[user]
            n_items = len(item_id_map)
            scores = self.model.predict(user_index, np.arange(n_items))
            
            top_indices = np.argsort(-scores)[:k]
            
            recs = []
            for idx in top_indices:
                raw_item_id = index_to_item_id[idx]
                item_rows = self.fitted_data[self.fitted_data['iid'] == raw_item_id]
                if not item_rows.empty:
                    item_row = item_rows.iloc[0]
                    recs.append({
                        'iid': raw_item_id,
                        'title': item_row.get('title', ''),
                        'categories': item_row.get('categories', []),
                        'description': item_row.get('description', ''),
                        'score': float(scores[idx])
                    })
            
            res[user] = recs
        
        return res