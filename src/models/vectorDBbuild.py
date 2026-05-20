from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
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
    
    # def preprocess_for_augmentation(self, df):

    #     print("PREPROCESS CALLED")

    #     for _, row in df.iterrows():
    #         for rec in row['recommendations']:

    #             combined_text = (
    #                 f"Title: {rec['title']}\n"
    #                 f"Categories: {rec.get('categories', '')}\n"
    #                 f"Description: {rec.get('description', '')}"
    #             )

    #             rec['combined_text'] = combined_text

    #             rec['cf_score'] = float(rec.get('score', 0))

    #     sample = df.iloc[0]["recommendations"][0]

    #     print("SAMPLE AFTER PREPROCESS:")
    #     print(sample.keys())

    #     return df
    
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

        # items_df = pd.DataFrame(items_metadata).drop_duplicates('iid')
        # print(f"Computing embeddings for {len(items_df)} unique items...")
        # items_df['embedding'] = items_df['combined_text'].apply(
        #     lambda x: self.embedding_model.embed_query(x)
        # )

        # embedding_lookup = dict(
        #     zip(items_df['iid'], items_df['embedding'])
        # )

        # return items_df, embedding_lookup
        return pd.DataFrame(items_metadata).drop_duplicates('iid')
    
    # def extract_items_metadata(self, df):
    #     items_metadata = []
    #     seen_items = set()
    #     for _, row in df.iterrows():
    #         for rec in row['recommendations']:
    #             if rec['iid'] in seen_items:
    #                 continue
    #             seen_items.add(rec['iid'])
    #             combined_text = rec.get('combined_text', '')
    #             embedding = self.embedding_model.embed_query(combined_text)
    #             items_metadata.append({
    #                 'iid': rec['iid'],
    #                 'title': rec['title'],
    #                 'categories': rec.get('categories', []),
    #                 'description': rec.get('description', ''),
    #                 'cf_score': rec['cf_score'],
    #                 'combined_text': rec.get('combined_text', ''),
    #                 'embedding': embedding
    #             })
        
    #     items_df = pd.DataFrame(items_metadata)
    #     embedding_lookup = {row['iid']: row['embedding'] for _, row in items_df.iterrows()}
    #     items_df = items_df.drop(columns=['embedding'])

    #     return items_df, embedding_lookup

    def create_documents(self, items_df: pd.DataFrame) -> List[Document]:
        documents = []
        for _, row in items_df.iterrows():
            categories = row.get('categories', '')
            if isinstance(categories, list):
                categories = ", ".join(categories)
            full_text = (
                f"Title: {row['title']}\n"
                f"Categories: {categories}\n"
                f"Description: {row['description']}\n"
            )
            chunks = self.text_splitter.split_text(full_text)
            for i, chunk in enumerate(chunks):
                documents.append(Document(
                    page_content=chunk,
                    metadata={
                        "iid": row['iid'],
                        "title": row['title'],
                        "chunk_num": i,
                        "categories": row.get('categories', []),
                        "cf_score": row['cf_score']
                    }
                ))
        return documents

    def build_from_dataframe(self, items_df: pd.DataFrame, persist_dir: str = "./chroma_db"):
        documents = self.create_documents(items_df)
        print(f"Creating {len(documents)} documents...")
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