import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from src.services.pipeline_loader import PipelineArtifacts


def clean_categories(categories_list):
    # print(f"  clean_categories input: {categories_list} (type: {type(categories_list)})")
    
    if not categories_list:
        print("  Empty categories list")
        return []
    
    if isinstance(categories_list, str):
        # print(f"Input is string, splitting by comma")
        # split by comma if it's a string
        categories_list = [cat.strip() for cat in categories_list.split(',')]
    
    if not isinstance(categories_list, list):
        print("Not a list, returning empty")
        return []
    
    # Generic categories that don't provide meaningful diversity information
    generic_categories = {
        'CDs & Vinyl', 'Digital Music', 'Music', 'Albums', 
        'CDs', 'Vinyl'
    }
    
    cleaned = []
    for i, category in enumerate(categories_list):
        # print(f"Processing category {i}: '{category}'")
        
        if isinstance(category, str):
            category = category.strip()
            if category not in generic_categories:
                cleaned.append(category)
            #     print(f"Added: '{category}'")
            # else:
            #     print(f"Skipped (generic): '{category}'")
        else:
            print(f"Skipped (not string): '{category}'")
    
    # print(f"Final cleaned categories: {cleaned}")
    return cleaned

def calculate_intra_list_similarity(recommendations, pipeline_objects: PipelineArtifacts):
    if len(recommendations) <= 1:
        return 0
    
    try:
        if pipeline_objects and hasattr(pipeline_objects.retriever, 'embedding_model'):
            embedder = pipeline_objects.retriever.embedding_model
            
            item_embeddings = []
            for rec in recommendations:
                item_text = f"{rec.get('title', '')} {rec.get('categories', '')}"
                embedding = embedder.embed_query(item_text)
                item_embeddings.append(embedding)
            
            # calculate pairwise similarities
            similarity_matrix = cosine_similarity(item_embeddings)
            
            # intra-list similarity = average similarity between different items
            total_similarity = 0
            pair_count = 0
            
            for i in range(len(recommendations)):
                for j in range(i + 1, len(recommendations)):
                    total_similarity += similarity_matrix[i][j]
                    pair_count += 1
            
            return total_similarity / pair_count if pair_count > 0 else 0
            
    except Exception as e:
        print(f"Error calculating intra-list similarity: {e}")
    
    # Fallback: use category-based similarity
    category_sets = []
    for rec in recommendations:
        raw_categories = rec.get('categories', [])
        cleaned_cats = clean_categories(raw_categories)
        category_sets.append(set(cleaned_cats))
    
    total_similarity = 0
    pair_count = 0
    
    for i in range(len(category_sets)):
        for j in range(i + 1, len(category_sets)):
            # jaccard similarity between category sets
            intersection = len(category_sets[i] & category_sets[j])
            union = len(category_sets[i] | category_sets[j])
            similarity = intersection / union if union > 0 else 0
            total_similarity += similarity
            pair_count += 1
    
    return total_similarity / pair_count if pair_count > 0 else 0

def calculate_novelty(recommendations, pipeline_objects):
    """
    Calculate how novel/unexpected the recommendations are
    Higher values = more novel/surprising recommendations
    """
    if not recommendations:
        return 0
    
    try:
        # Simple novelty: inverse of average category frequency
        all_categories = []
        for rec in recommendations:
            raw_categories = rec.get('categories', [])
            cleaned_cats = clean_categories(raw_categories)
            all_categories.extend(cleaned_cats)
        
        if not all_categories:
            return 0
        
        category_counts = Counter(all_categories)
        total_items = len(recommendations)
        
        # Novelty = 1 - (average category frequency)
        avg_category_freq = sum(category_counts.values()) / len(category_counts) / total_items
        novelty = 1 - avg_category_freq
        
        return max(0, novelty)
        
    except Exception as e:
        print(f"Error calculating novelty: {e}")
        return 0

def calculate_serendipity(recommendations, pipeline_objects):
    """
    Calculate serendipity - balance between relevance and unexpectedness
    Higher values = good mix of relevant but surprising items
    """
    if len(recommendations) <= 1:
        return 0
    
    try:
        # Serendipity = Novelty * (1 - IntraListSimilarity)
        # This rewards lists that are novel but not too similar to each other
        novelty = calculate_novelty(recommendations, pipeline_objects)
        intra_list_sim = calculate_intra_list_similarity(recommendations, pipeline_objects)
        
        serendipity = novelty * (1 - intra_list_sim)
        return serendipity
        
    except Exception as e:
        print(f"Error calculating serendipity: {e}")
        return 0

def calculate_diversity_metrics(recommendations,  pipeline_objects=None):
    # print(f"\n=== DEBUGGING DIVERSITY METRICS ===")
    
    if not recommendations:
        return {
            'CategoryCoverage': 0,
            'UniqueCategories': 0,
            'Entropy': 0,
            'GiniIndex': 0,
            'IntraListSimilarity': 0,
            'Novelty': 0,
            'Serendipity': 0
        }
    
    all_categories = []
    for rec in recommendations:
        raw_categories = rec.get('categories', [])
        cleaned_cats = clean_categories(raw_categories)
        all_categories.extend(cleaned_cats)
    
    if not all_categories:
        return {
            'CategoryCoverage': 0,
            'UniqueCategories': 0,
            'Entropy': 0,
            'GiniIndex': 0,
            'IntraListSimilarity': 0,
            'Novelty': 0,
            'Serendipity': 0
        }
    
    category_counts = Counter(all_categories)
    n_recommendations = len(recommendations)
    unique_categories = len(category_counts)
    
    # Existing metrics
    category_coverage = unique_categories / n_recommendations
    total_categories = len(all_categories)
    
    entropy = 0
    for category, count in category_counts.items():
        p = count / total_categories
        entropy -= p * np.log2(p) if p > 0 else 0
    
    gini = 1 - sum((count / total_categories) ** 2 for count in category_counts.values())
    
    # 1. Intra-List Similarity (ILS)
    intra_list_similarity = calculate_intra_list_similarity(recommendations, pipeline_objects)
    
    # 2. Novelty (how unexpected items are)
    novelty = calculate_novelty(recommendations, pipeline_objects)
    
    # 3. Serendipity (balance of relevance and unexpectedness)
    serendipity = calculate_serendipity(recommendations, pipeline_objects)
    
    return {
        # Existing metrics
        'CategoryCoverage': category_coverage,
        'UniqueCategories': unique_categories,
        'Entropy': entropy,
        'GiniIndex': gini,
        
        # New intra-list metrics
        'IntraListSimilarity': intra_list_similarity,
        'Novelty': novelty,
        'Serendipity': serendipity
    }
    
    # all_categories = []
    
    # for i, rec in enumerate(recommendations):
    #     raw_categories = rec.get('categories', [])
        
    #     cleaned_cats = clean_categories(raw_categories)
        
    #     all_categories.extend(cleaned_cats)
    
    # if not all_categories:
    #     return {
    #         'CategoryCoverage': 0,
    #         'UniqueCategories': 0,
    #         'Entropy': 0,
    #         'GiniIndex': 0,
    #         'GenreDiversity': 0
    #     }
    
    # category_counts = Counter(all_categories)
    # # calculate metrics
    # n_recommendations = len(recommendations)
    # unique_categories = len(category_counts)
    
    # category_coverage = unique_categories / n_recommendations
    # entropy = 0
    # gini = 0
    
    # total_categories = len(all_categories)
    
    # # entropy
    # for category, count in category_counts.items():
    #     p = count / total_categories
    #     entropy -= p * np.log2(p) if p > 0 else 0
    #     # print(f"  Category '{category}': count={count}, probability={p:.3f}")
    
    # # Gini index
    # gini = 1 - sum((count / total_categories) ** 2 for count in category_counts.values())
    
    
    # return {
    #     'CategoryCoverage': category_coverage,
    #     'UniqueCategories': unique_categories,
    #     'Entropy': entropy,
    #     'GiniIndex': gini
    # }

def calculate_aggregate_diversity_metrics(all_diversity_metrics):
    """
    Calculate aggregate diversity metrics across all users
    """
    if not all_diversity_metrics:
        return {}
    
    df = pd.DataFrame(all_diversity_metrics)
    return {
        'Mean_CategoryCoverage': df['CategoryCoverage'].mean(),
        'Mean_UniqueCategories': df['UniqueCategories'].mean(),
        'Mean_Entropy': df['Entropy'].mean(),
        'Mean_GiniIndex': df['GiniIndex'].mean(),
        'Mean_IntraListSimilarity': df['IntraListSimilarity'].mean(),
        'Mean_Novelty': df['Novelty'].mean(),
        'Mean_Serendipity': df['Serendipity'].mean(),
        
        # Std
        'Std_CategoryCoverage': df['CategoryCoverage'].std(),
        'Std_UniqueCategories': df['UniqueCategories'].std(),
        'Std_IntraListSimilarity': df['IntraListSimilarity'].std()
    }