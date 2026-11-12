import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict

class RetrievalEngine:  
    def __init__(self, chroma_db, embedding_model, embedding_lookup=None):
        self.chroma_db = chroma_db
        self.embedding_model = embedding_model
        # self.embedding_lookup = embedding_lookup or {}
        # print(f"Embedding lookup size: {len(self.embedding_lookup)}")

    def _get_cf_score(self, iid: str) -> float:
        results = self.chroma_db._collection.get(where={"iid": iid})
        return results['metadatas'][0].get('cf_score', 0) if results else 0

    def hybrid_retrieve(self, user_recs: List[Dict], query: str,
                cf_weight: float = 0.5) -> List[Dict]:
        """
        Hybrid Retrieval Process:
        1. Get semantic similarity between query and each item
        2. Combine with CF scores
        3. Return weighted hybrid scores
        
        Args:
            user_recs: List of user's recommendations with CF scores
            query: User's search query
            cf_weight: 0.5 = equal weight to CF and semantic scores
        """
        query_embedding = self.embedding_model.embed_query(query)
        hybrid_results = []
        
        for rec in user_recs:
            # item_embedding = self.embedding_lookup.get(rec['iid'])
            # if item_embedding is None:
            #     continue
            item_embedding = self.embedding_model.embed_query(
                rec["augmented_text"]
            )
            
            semantic_score = cosine_similarity([query_embedding], [item_embedding])[0][0]
            
            cf_score = rec.get('score', self._get_cf_score(rec['iid']))
            
            hybrid_score = (cf_weight * cf_score) + ((1 - cf_weight) * semantic_score)
            
            hybrid_results.append({
                **rec,
                "semantic_score": float(semantic_score),
                "cf_score": float(cf_score),
                "hybrid_score": float(hybrid_score)
            })
        
        return sorted(hybrid_results, key=lambda x: x['hybrid_score'], reverse=True)

    def mmr_diversify(self, items: List[Dict], query: str,
                    lambda_param: float = 0.7, top_k: int = 50) -> List[Dict]:
        """"
        MMR Diversification Process:
        1. Compute query-doc similarity for relevance
        2. Compute doc-doc similarity for diversity
        3. Select items that maximize: 
           lambda * (query relevance) - (1-lambda) * (max similarity to selected docs)
        
        Args:
            items: List of scored items from hybrid_retrieve
            query: Original user query
            lambda_param: 1.0 = pure relevance, 0.0 = pure diversity
            top_k: Number of items to return
        """
        if not items:
            return []
        
        # prepare embeddings
        query_embedding = self.embedding_model.embed_query(query)
        
        # valid_items = []
        # item_embeddings = []
        # for item in items:
        #     emb = self.embedding_lookup.get(item['iid'])

        #     if emb is None:
        #         continue

        #     valid_items.append(item)
        #     item_embeddings.append(emb)
            
        # items = valid_items
        # if len(items) == 0:
        #     return []
        # item_embeddings = np.array(item_embeddings)
        # item_embeddings = np.array([embedding_lookup[item['iid']] for item in items])
        
        item_embeddings = np.array([
            self.embedding_model.embed_query(item["augmented_text"]) for item in items])
        
        # compute similarity matrices
        query_similarities = cosine_similarity([query_embedding], item_embeddings)[0]
        doc_similarities = cosine_similarity(item_embeddings)
        
        # MMR selection
        selected_items = []
        selected_indices = []
        remaining_indices = set(range(len(items)))
        mmr_scores_dict = {}
        
        while len(selected_indices) < min(top_k, len(items)) and remaining_indices:
            cur_score = []
            for idx in remaining_indices:
                if not selected_indices:
                    # 1st item uses query similarity only
                    score = lambda_param * query_similarities[idx]
                else:
                    # subsequent items balance query and doc similarities
                    max_sim_to_selected = max(doc_similarities[idx][s_idx] 
                                            for s_idx in selected_indices)
                    score = (lambda_param * query_similarities[idx] - 
                        (1 - lambda_param) * max_sim_to_selected)
                cur_score.append((idx, score))
            
            if not cur_score:
                break
            
            # select item with highest MMR score
            best_idx, best_score = max(cur_score, key=lambda x: x[1])
            selected_items.append(items[best_idx])
            selected_indices.append(best_idx)
            mmr_scores_dict[best_idx]= best_score
            remaining_indices.remove(best_idx)
        
        results = []
        for item in selected_items:
            original_idx = items.index(item)
            results.append({
                **item,
                'mmr_score': mmr_scores_dict.get(original_idx, 0),
                'diversity_score': mmr_scores_dict.get(original_idx, 0)
            })
        return results