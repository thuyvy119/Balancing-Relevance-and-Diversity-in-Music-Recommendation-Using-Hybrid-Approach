from src.utils.io import load_recommendations
from src.services.pipeline_artifacts import PipelineArtifacts
import pandas as pd
from src.models.llm import LLMReranker
from langchain_community.vectorstores import Chroma
# from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.models.hybrid_retrieval import RetrievalEngine

def load_pipeline(
    save_dir="saved_pipeline",
    chroma_db_path="./chroma_db"
):

    df = load_recommendations(save_dir)

    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    chroma_db = Chroma(
        persist_directory=chroma_db_path,
        embedding_function=embedder,
    )

    retriever = RetrievalEngine(
    chroma_db=chroma_db,
    embedding_model=embedder,
    # embedding_lookup=embedding_lookup
)
    # embedding_lookup = {}
    # for _, row in df.iterrows():
    #     for rec in row['recommendations']:
    #         if 'embedding' in rec:
                # embedding_lookup[rec['iid']] = rec['embedding']
    
    llm_reranker = LLMReranker()

    return PipelineArtifacts(
        df= df,
        chroma_db = chroma_db,
        retriever = retriever,
        # embedding_lookup = embedding_lookup,
        llm_reranker = llm_reranker
    )  