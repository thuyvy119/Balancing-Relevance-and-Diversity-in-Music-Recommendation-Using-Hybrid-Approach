from dataclasses import dataclass
import pandas as pd

@dataclass
class PipelineArtifacts:
    df: pd.DataFrame
    chroma_db: object
    retriever: object
    llm_reranker: object