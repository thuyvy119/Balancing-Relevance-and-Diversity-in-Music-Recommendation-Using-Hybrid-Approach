import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.services.pipeline_artifacts import PipelineArtifacts

def evaluate_query_relevance(recommendations, query, pipeline_objects: PipelineArtifacts, similarity_threshold=0.25):
    try:
        embedder = pipeline_objects.retriever.embedding_model
        query_embedding = embedder.embed_query(query)
        
        relevance_scores = []
        similarity_scores = []
        
        for rec in recommendations:
            # Use the item's semantic representation
            item_text = f"{rec.get('title', '')} {rec.get('categories', '')} {rec.get('description', '')}"
            item_embedding = embedder.embed_query(item_text)
            
            similarity = cosine_similarity([query_embedding], [item_embedding])[0][0]
            is_relevant = similarity >= similarity_threshold
            
            relevance_scores.append(1 if is_relevant else 0)
            similarity_scores.append(similarity)
        
        num_relevant = sum(relevance_scores)
        precision = num_relevant / len(relevance_scores) if recommendations else 0
        hit_rate = 1 if num_relevant > 0 else 0
        avg_similarity = np.mean(similarity_scores) if similarity_scores else 0
        
        # Advanced rank-aware metrics
        # MAP (Mean Average Precision)
        ap = 0.0
        relevant_count = 0
        for i, rel in enumerate(relevance_scores):
            if rel == 1:
                relevant_count += 1
                ap += relevant_count / (i + 1)
        map_score = ap / num_relevant if num_relevant > 0 else 0
        
        # MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for i, rel in enumerate(relevance_scores):
            if rel == 1:
                mrr = 1.0 / (i + 1)
                break
        
        # NDCG (Normalized Discounted Cumulative Gain)
        dcg = 0.0
        for i, (rel, sim) in enumerate(zip(relevance_scores, similarity_scores)):
            # Use similarity as gain, discounted by position
            gain = sim if rel == 1 else 0  # Could also use binary relevance
            dcg += gain / np.log2(i + 2)  # i+2 because log base 2 of 1 is 0
            
        # Ideal DCG (sorted by relevance)
        ideal_relevance = sorted([sim if rel == 1 else 0 for rel, sim in zip(relevance_scores, similarity_scores)], reverse=True)
        idcg = sum(gain / np.log2(i + 2) for i, gain in enumerate(ideal_relevance))
        ndcg = dcg / idcg if idcg > 0 else 0
        
        return {
            # Basic metrics
            f'Precision@{len(recommendations)}': precision,
            f'HitRate@{len(recommendations)}': hit_rate,
            'avg_similarity': avg_similarity,
            'num_relevant': num_relevant,
            
            # Advanced metrics
            f'MAP@{len(recommendations)}': map_score,
            f'MRR@{len(recommendations)}': mrr,
            f'NDCG@{len(recommendations)}': ndcg,
            
            'relevance_scores': relevance_scores,
            'similarity_scores': similarity_scores
        }
        
    except Exception as e:
        print(f"Error in evaluate_advanced_relevance: {e}")
        return {
            f'Precision@{len(recommendations)}': 0,
            f'HitRate@{len(recommendations)}': 0,
            'avg_similarity': 0,
            'num_relevant': 0,
            f'MAP@{len(recommendations)}': 0,
            f'MRR@{len(recommendations)}': 0,
            f'NDCG@{len(recommendations)}': 0,
            'relevance_scores': [0] * len(recommendations),
            'similarity_scores': [0] * len(recommendations)
        }
        
    #     precision = sum(relevance_scores) / len(relevance_scores) if recommendations else 0
    #     hit_rate = 1 if any(relevance_scores) else 0
    #     avg_similarity = np.mean(similarity_scores) if similarity_scores else 0
        
    #     return {
    #         f'Precision@{len(recommendations)}': precision,
    #         f'HitRate@{len(recommendations)}': hit_rate,
    #         'avg_similarity': avg_similarity,
    #         'relevance_scores': relevance_scores,
    #         'similarity_scores': similarity_scores
    #     }
    # except Exception as e:
    #     print(f"Error in evaluate_query_relevance_auto: {e}")
        
    #     return {
    #         f'Precision@{len(recommendations)}': 0,
    #         f'HitRate@{len(recommendations)}': 0,
    #         'avg_similarity': 0,
    #         'relevance_scores': [0] * len(recommendations),
    #         'similarity_scores': [0] * len(recommendations)
    #     }

def calculate_aggregate_query_relevance_metrics(user_metrics_list, k=10):
    if not user_metrics_list:
        return {}

    df = pd.DataFrame(user_metrics_list)
    
    precision_col = f'Precision@{k}'
    hitrate_col = f'HitRate@{k}'
    map_col = f'MAP@{k}'
    mrr_col = f'MRR@{k}'
    ndcg_col = f'NDCG@{k}'
    
    aggregate_metrics = {
        f'Mean_Precision@{k}': df[precision_col].mean() if precision_col in df.columns else 0,
        f'Mean_HitRate@{k}': df[hitrate_col].mean() if hitrate_col in df.columns else 0,
        'Mean_Similarity': df['avg_similarity'].mean() if 'avg_similarity' in df.columns else 0,
        f'Mean_MAP@{k}': df[map_col].mean() if map_col in df.columns else 0,
        f'Mean_MRR@{k}': df[mrr_col].mean() if mrr_col in df.columns else 0,
        f'Mean_NDCG@{k}': df[ndcg_col].mean() if ndcg_col in df.columns else 0,
        
        # Additional statistics
        'Total_Users': len(df),
        'Std_Precision': df[precision_col].std() if precision_col in df.columns else 0,
        'Std_MAP': df[map_col].std() if map_col in df.columns else 0,
        'Std_MRR': df[mrr_col].std() if mrr_col in df.columns else 0
    }
    
    return aggregate_metrics
