import pandas as pd
from src.models.lightfm import LightFMTrainer
from src.models.vectorDBbuild import VectorDBBuilder
from src.models.hybrid_retrieval import RetrievalEngine
from src.models.llm import LLMReranker
from langchain_huggingface import HuggingFaceEmbeddings
from src.utils.io import save_pipeline

def train_pipeline(train_data_path, k=50):
    train_data = pd.read_csv(train_data_path)

    lightfm_model = LightFMTrainer()
    lightfm_model.fit_and_train(train_data, learning_rate=0.01, epochs=50)

    all_recommendations = lightfm_model.get_top_k(k=k)

    df = pd.DataFrame([
        {"user_id": uid, "recommendations": recs}
        for uid, recs in all_recommendations.items()
    ])

    vector_builder = VectorDBBuilder()
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    df = vector_builder.preprocess_for_augmentation(df)
    # print("TRAIN PIPELINE CHECK:")
    # print(df.iloc[0]["recommendations"][0].keys())
    
    items_df = vector_builder.extract_items_metadata(df)
    # for _, row in df.iterrows():
    #     for rec in row['recommendations']:
    #         rec['embedding'] = embedding_lookup.get(rec['iid'])
    
    chroma_db = vector_builder.build_from_dataframe(items_df)

    retriever = RetrievalEngine(None, embedder)
    retriever.chroma_db = chroma_db
    llm_reranker = LLMReranker()

    save_pipeline(df, chroma_db)

    return {
        "df": df,
        "chroma_db": chroma_db,
        "retriever": retriever,
        "llm_reranker": llm_reranker
    }
