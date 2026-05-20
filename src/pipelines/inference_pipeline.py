from src.services.pipeline_loader import load_pipeline
from src.services.recommender_service import recommend_for_user


def inference_pipeline(user_id, query, debug=False):

    pipeline_objects = load_pipeline()

    results = recommend_for_user(
        pipeline_objects,
        user_id=user_id,
        query=query,
        debug=debug
    )

    return results


# from src.utils.user_utils import check_user_type

# def recommend_for_user(
#     pipeline_objects,
#     user_id,
#     query,
#     debug=False
# ):
#     df = pipeline_objects["df"]
#     chroma_db = pipeline_objects["chroma_db"]
#     retriever = pipeline_objects["retriever"]

#     if not check_user_type(user_id, df):
#         raise ValueError("New-user flow not implemented yet")

#     user_recs = df[df.user_id == user_id].iloc[0]["recommendations"]

#     augmented = []
#     for rec in user_recs:
#         chunks = chroma_db.similarity_search(
#             query=rec.get("combined_text", rec.get("title", "")),
#             k=3,
#             filter={"iid": rec["iid"]}
#         )

#         augmented.append({
#             **rec,
#             "augmented_text": rec.get("combined_text", "") +
#                             " ".join(c.page_content for c in chunks)
#         })

#     scored = retriever.hybrid_retrieve(augmented, query)
#     diversified = retriever.mmr_diversify(scored, query)

#     return diversified[:10]
