from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import pandas as pd
from typing import List

class VectorDBBuilder:
    def __init__(self, embedding_model: str = "sentence-transformers/all-mpnet-base-v2"):
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " "]
        )

    def preprocess_for_augmentation(self, df):
        for _, row in df.iterrows():
            for rec in row['recommendations']:
                rec['combined_text'] = (
                    f"Title: {rec['title']}\n"
                    f"Categories: {rec.get('categories', '')}\n"
                    f"Description: {rec.get('description', '')}"
                )
                rec['cf_score'] = float(rec.get('score', 0)) 
        return df
    
    # def preprocess_for_augmentation(self, df):
    #     df_processed = df.copy()
    #     df_processed['combined_text'] = ("Title: " + df_processed['title'].fillna('') + "\n" +
    #         "Categories: " + df_processed['categories'].fillna('').astype(str) + "\n" +
    #         "Description: " + df_processed['description'].fillna('')
    #     )
    #     df_processed['cf_score'] = df_processed.get('score', 0.0)
    #     return df_processed
    
    def extract_items_metadata(self, df):
        items_metadata = []
        for _, row in df.iterrows():
            for rec in row['recommendations']:
                items_metadata.append({
                    'iid': rec['iid'],
                    'title': rec['title'],
                    'categories': rec.get('categories', []),
                    'description': rec.get('description', ''),
                    'cf_score': rec['cf_score'],
                    'combined_text': rec.get('combined_text', '')
                })
        return pd.DataFrame(items_metadata).drop_duplicates('iid')

    def create_documents(self, items_df: pd.DataFrame) -> List[Document]:
        documents = []
        for _, row in items_df.iterrows():
            full_text = (
                f"Title: {row['title']}\n"
                f"Categories: {', '.join(row.get('categories', []))}\n"
                f"Description: {row['description']}\n"
            )
            chunks = self.text_splitter.split_text(full_text)
            for i, chunk in enumerate(chunks):
                documents.append(Document(
                    page_content=chunk,
                    metadata={
                        "item_id": row['iid'],
                        "title": row['title'],
                        "chunk_num": i,
                        "categories": row.get('categories', []),
                        "cf_score": row['cf_score']
                    }
                ))
        return documents

    def build_from_dataframe(self, items_df: pd.DataFrame, persist_dir: str = "./chroma_db"):
        documents = self.create_documents(items_df)
        return Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=persist_dir
        )

    @staticmethod
    def load_existing(persist_dir: str = "./chroma_db"):
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=HuggingFaceEmbeddings()
        )