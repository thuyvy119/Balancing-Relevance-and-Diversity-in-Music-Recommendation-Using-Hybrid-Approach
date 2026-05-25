import random
from src.services.recommender_service import recommend_for_user
from src.evaluation.relevance_metrics import evaluate_query_relevance, calculate_aggregate_query_relevance_metrics
from src.evaluation.diversity_metrics import calculate_diversity_metrics, calculate_aggregate_diversity_metrics
from src.utils.user_utils import get_qualified_users

def evaluate_users(pipeline_objects, train_data, test_data, query, sample_size=None, use_llm=True):
    qualified_users = get_qualified_users(train_data, test_data)
    print(f"Qualified users count: {len(qualified_users)}")
    
    if sample_size:
        qualified_users = random.sample(qualified_users, sample_size)
        
    all_relevance_metrics = []
    all_diversity_metrics = []
    all_recommendations = []

    for user_id in qualified_users:
        print (f"Evaluating user {user_id}...")
        recommendations = recommend_for_user(
            pipeline_objects,
            user_id=user_id,
            query=query,
            use_llm=use_llm)

        if not recommendations:
            continue
        
        print("Evaluating relevance...")
        relevance = evaluate_query_relevance(
            recommendations,
            query,
            pipeline_objects
        )

        print("Evaluating diversity...")
        diversity = calculate_diversity_metrics(
            recommendations,
            pipeline_objects
        )

        relevance["user_id"] = user_id
        diversity["user_id"] = user_id

        all_relevance_metrics.append(relevance)
        all_diversity_metrics.append(diversity)

        all_recommendations.append({
            "user_id": user_id,
            "query": query,
            "recommendations": recommendations
        })

    aggregate_relevance = calculate_aggregate_query_relevance_metrics(
        all_relevance_metrics
    )

    aggregate_diversity = calculate_aggregate_diversity_metrics(
        all_diversity_metrics
    )

    return {
        "relevance_metrics": all_relevance_metrics,
        "diversity_metrics": all_diversity_metrics,
        "aggregate_relevance": aggregate_relevance,
        "aggregate_diversity": aggregate_diversity,
        "recommendations": all_recommendations
    }