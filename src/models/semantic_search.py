from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity

class NewUserRetriever:
    def __init__(self, chroma_db, embedder):
        self.db = chroma_db
        self.embedder = embedder
    
    def retrieve(self, query, k=50):
        """Pure semantic search for new users"""
        results = self.db.similarity_search(query, k=k)
        return [{
            'iid': r.metadata['item_id'],
            'title': r.metadata['title'],
            'semantic_score': cosine_similarity(
                [self.embedder.embed_query(query)],
                [self.embedder.embed_query(r.page_content)]
            )[0][0],
            'source': 'semantic'
        } for r in results]