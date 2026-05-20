import os
import json
import pickle
import pandas as pd
from datetime import datetime

def save_pipeline(df, chroma_db, save_dir="saved_pipeline"):
    os.makedirs(save_dir, exist_ok=True)

    df.to_pickle(f"{save_dir}/recommendations.pkl")
    chroma_db.persist()

def load_recommendations(save_dir="saved_pipeline"):
    return pd.read_pickle(f"{save_dir}/recommendations.pkl")

def save_recommendations(all_recommendations, test_query, save_prefix=None, save_dir="saved_recommendations"):
    os.makedirs(save_dir, exist_ok=True)
    
    if not all_recommendations:
        print("No recommendations to save")
        return
    
    if save_prefix is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_slug = test_query.replace(' ', '_').replace('-', '_')[:40]
        save_prefix = f"{timestamp}_{query_slug}"
    
    debug_data = []
    
    for rec_data in all_recommendations:
        user_row = {
            'user_id': rec_data['user_id'],
            'query': test_query,
            'recommendation_count': len(rec_data['recommendations'])
        }
        
        for i, rec in enumerate(rec_data['recommendations'], 1):
            description = rec.get('description', '')
            if not isinstance(description, str):
                description = str(description) if description is not None else ''
            description = description[:100] if description else ''
            
            user_row.update({
                f'rec_{i}_title': rec.get('title', ''),
                f'rec_{i}_id': rec.get('iid', ''),
                f'rec_{i}_categories': str(rec.get('categories', [])),
                f'rec_{i}_description': description,
            })
        
        # fill remaining slots if user has fewer than 10 recommendations
        num_recs = len(rec_data['recommendations'])
        if num_recs < 10:
            for i in range(num_recs + 1, 11):
                user_row.update({
                    f'rec_{i}_title': '',
                    f'rec_{i}_id': '',
                    f'rec_{i}_categories': '',
                    f'rec_{i}_description': '',
                    f'rec_{i}_similarity': 0,
                    f'rec_{i}_relevance': 0
                })
        
        debug_data.append(user_row)
    
    debug_df = pd.DataFrame(debug_data)
    recommendations_filename = f"{save_prefix}_recommendations.csv"
    debug_df.to_csv(f"{save_dir}/{recommendations_filename}", index=False)
    print(f"Recommendations saved to: {save_dir}/{recommendations_filename}")
    
    return recommendations_filename

def save_comprehensive_results(relevance_metrics, diversity_metrics, test_query="", save_prefix=None, save_dir="saved_results"):
    os.makedirs(save_dir, exist_ok=True)
    if not relevance_metrics or not diversity_metrics:
        print("No data to save for comprehensive results")
        return
    
    # generate consistent prefix if not provided
    if save_prefix is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_slug = test_query.replace(' ', '_').replace('-', '_')[:40]
        save_prefix = f"{timestamp}_{query_slug}"
    
    rel_df = pd.DataFrame(relevance_metrics)
    div_df = pd.DataFrame(diversity_metrics)
    
    results_df = rel_df.merge(div_df, on='user_id', suffixes=('_rel', '_div'))
    
    detailed_filename = f"{save_prefix}_evaluation_results.csv"
    results_df.to_csv(detailed_filename, index=False)
    print(f"Detailed evaluation results saved to: {detailed_filename}")
    
    # save summary
    summary_data = {
        'Metric': [
            'Precision@10', 'HitRate@10', 'AvgSemanticSimilarity', 'MAP@10', 'MRR@10', 'NDCG@10',
            'CategoryCoverage', 'UniqueCategories', 'Entropy', 'GiniIndex', 
            'IntraListSimilarity', 'Novelty', 'Serendipity'
        ],
        'Value': [
            rel_df.get('Precision@10', pd.Series([0])).mean(),
            rel_df.get('HitRate@10', pd.Series([0])).mean(),
            rel_df.get('avg_similarity', pd.Series([0])).mean(),
            rel_df.get('MAP@10', pd.Series([0])).mean(),
            rel_df.get('MRR@10', pd.Series([0])).mean(),
            rel_df.get('NDCG@10', pd.Series([0])).mean(),
            div_df.get('CategoryCoverage', pd.Series([0])).mean(),
            div_df.get('UniqueCategories', pd.Series([0])).mean(),
            div_df.get('Entropy', pd.Series([0])).mean(),
            div_df.get('GiniIndex', pd.Series([0])).mean(),
            div_df.get('IntraListSimilarity', pd.Series([0])).mean(),
            div_df.get('Novelty', pd.Series([0])).mean(),
            div_df.get('Serendipity', pd.Series([0])).mean()
        ],
        'Query': test_query,
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Users_Evaluated': len(results_df)
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_filename = f"{save_prefix}_evaluation_summary.csv"
    summary_df.to_csv(f"{save_dir}/{summary_filename}", index=False)
    print(f"Evaluation summary saved to: {save_dir}/{summary_filename}")
    
    return detailed_filename, summary_filename
