def recommend_for_user(pipeline_objects, user_id, query, debug=False, use_llm=True):

    df = pipeline_objects.df
    chroma_db = pipeline_objects.chroma_db
    retriever = pipeline_objects.retriever
    # embedding_lookup = pipeline_objects.embedding_lookup
    llm_reranker = pipeline_objects.llm_reranker

    user_recs = df[df["user_id"] == user_id]["recommendations"].iloc[0]
    print(f"Retrieved {len(user_recs)} recommendations for user {user_id}")
    
    augmented = []

    for rec in user_recs:
        try:
            chunks = chroma_db.similarity_search(
                query=rec.get("combined_text", ""),
                k=3,
                filter={"iid": rec["iid"]},
            )

            augmented_text = (
                rec.get("combined_text", "")
                + "\n"
                + " ".join([c.page_content for c in chunks])
            )

        except Exception as e:
            print(f"Augmentation error while processing recommendation for user {user_id}: {e}")
            augmented_text = rec.get("combined_text", "")

        augmented.append({
            **rec,
            "augmented_text": augmented_text
        })
        
    print("Starting hybrid retrieval...")
    hybrid_results = retriever.hybrid_retrieve(augmented, query)
    print(f"Hybrid retrieval results: {len(hybrid_results)}")
    diversified = retriever.mmr_diversify(hybrid_results, query, top_k=20)
    print(f"Length of MMR results: {len(diversified)}")
    if use_llm:
        final_results = llm_reranker.rerank(query=query, candidates=diversified[:20])
    else:
        final_results = diversified
        
    if debug:
        print("\nFinal Reranked Results:")
        
        for idx, item in enumerate(final_results):
            print(
                f"{idx + 1}. {item.get('title', 'No Title')} |"
                f"(IID: {item.get('iid', 'N/A')}) |"
                f"Hybrid Score: {item.get('hybrid_score', 0):.4f} |"
                f"Semantic Score: {item.get('semantic_score', 0):.4f} |"
            )
    return diversified[:10]