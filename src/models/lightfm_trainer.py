import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from tqdm import tqdm


class LightFMTrainer:

    def __init__(self, learning_rate=0.005, epochs=20, no_components=10,
                 alpha=1e-4, random_state=42, num_threads=1):
        self.lr, self.epochs = learning_rate, epochs
        self.no_components, self.alpha = no_components, alpha
        self.random_state, self.num_threads = random_state, num_threads
        self.model = self.dataset = self.fitted_data = None

    def fit_and_train(self, data):
        # label = 1 iff rating >= 4 (see preprocessing); keep positives, one row per (user, item)
        data = (data[data['label'] > 0]
                .drop_duplicates(['uid', 'iid']).reset_index(drop=True))

        dataset = Dataset()
        dataset.fit(users=data['uid'].unique(), items=data['iid'].unique())
        interactions, weights = dataset.build_interactions(zip(data['uid'], data['iid']))

        model = LightFM(loss='warp', learning_rate=self.lr,
                        no_components=self.no_components,
                        item_alpha=self.alpha, user_alpha=self.alpha,
                        random_state=self.random_state)
        model.fit(interactions, epochs=self.epochs, num_threads=self.num_threads)

        self.model, self.dataset = model, dataset
        self.fitted_data, self.interactions, self.weights = data, interactions, weights
        return model, dataset, interactions, data, weights

    def get_top_k(self, k=50, exclude_seen=True):
        """Top-k candidate pool per user. Items the user already interacted with in
        training are excluded (they cannot be useful test-time recommendations)."""
        user_id_map, _, item_id_map, _ = self.dataset.mapping()
        index_to_item = {v: key for key, v in item_id_map.items()}
        meta = (self.fitted_data.drop_duplicates('iid').set_index('iid')
                [['title', 'categories', 'description']].to_dict('index'))
        csr = self.interactions.tocsr()
        n_items = len(item_id_map)
        all_items = np.arange(n_items)
        kk = min(k, n_items - 1)

        res = {}
        for user in tqdm(self.fitted_data['uid'].unique()):
            u = user_id_map[user]
            scores = self.model.predict(u, all_items).astype(float)
            if exclude_seen:
                scores[csr[u].indices] = -np.inf
            top = np.argpartition(-scores, kk)[:kk]
            top = top[np.argsort(-scores[top])]
            recs = []
            for idx in top:
                if not np.isfinite(scores[idx]):
                    continue
                iid = index_to_item[idx]
                m = meta.get(iid, {})
                recs.append({'iid': iid,
                             'title': m.get('title', ''),
                             'categories': m.get('categories', []),
                             'description': m.get('description', ''),
                             'score': float(scores[idx])})
            res[user] = recs
        return res