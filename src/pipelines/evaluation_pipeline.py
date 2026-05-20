from src.services.pipeline_loader import load_pipeline
from src.evaluation.evaluator import evaluate_users
from src.utils.io import save_recommendations, save_comprehensive_results
from src.evaluation.visualization import create_comprehensive_visualizations

def evaluation_pipeline(
    train_data,
    test_data,
    query,
    sample_size=None,
    use_llm=True
):
    print("\nLoading pipeline...")
    pipeline_objects = load_pipeline()

    print("\nStarting evaluation... \n")
    results = evaluate_users(
        pipeline_objects,
        train_data,
        test_data,
        query=query,
        sample_size=sample_size,
        use_llm=use_llm
    )
    agg_rel = results["aggregate_relevance"]
    agg_div = results["aggregate_diversity"]
    print("\nFINAL RESULTS")
    print("=" * 50)

    print(f"\nQUERY RELEVANCE METRICS ({agg_rel.get('Total_Users',0)} users):")
    print(f"   Average Precision@10:    {agg_rel.get('Mean_Precision@10',0):.4f}")
    print(f"   Average Hit Rate@10:     {agg_rel.get('Mean_HitRate@10',0):.4f}")
    print(f"   Average Semantic Similarity: {agg_rel.get('Mean_Similarity',0):.4f}")
    print(f"   Average MAP@10:          {agg_rel.get('Mean_MAP@10',0):.4f}")
    print(f"   Average MRR@10:          {agg_rel.get('Mean_MRR@10',0):.4f}")
    print(f"   Average NDCG@10:         {agg_rel.get('Mean_NDCG@10',0):.4f}")

    print("\nDIVERSITY METRICS:")
    print(f"   Mean Category Coverage: {agg_div.get('Mean_CategoryCoverage',0):.4f}")
    print(f"   Mean Unique Categories: {agg_div.get('Mean_UniqueCategories',0):.2f}")
    print(f"   Mean Entropy:          {agg_div.get('Mean_Entropy',0):.4f}")
    print(f"   Mean Gini Index:       {agg_div.get('Mean_GiniIndex',0):.4f}")
    print(f"   Mean Intra-List Similarity: {agg_div.get('Mean_IntraListSimilarity',0):.4f}")
    print(f"   Mean Novelty:              {agg_div.get('Mean_Novelty',0):.4f}")
    print(f"   Mean Serendipity:          {agg_div.get('Mean_Serendipity',0):.4f}")

    save_recommendations(results["recommendations"], query)

    save_comprehensive_results(
        results["relevance_metrics"],
        results["diversity_metrics"],
        query
    )
    
    print("\nGenerating visualizations...")
    create_comprehensive_visualizations(results["relevance_metrics"], results["diversity_metrics"], query=query, save_dir="evaluation_plots")
    print("\nVisualizations generated successfully!")

    print("\nEvaluation completed successfully!")

    return results