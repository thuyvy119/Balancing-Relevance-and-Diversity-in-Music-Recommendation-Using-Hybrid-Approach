from src.pipelines.evaluation_pipeline import evaluation_pipeline
import pandas as pd

if __name__ == "__main__":
    train_data_path = "data/train_set.csv"
    test_data_path = "data/test_set.csv"
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    query = input("Enter query: ")
    results = evaluation_pipeline(train_data, test_data, query, sample_size=50, use_llm=True)
    print(results)