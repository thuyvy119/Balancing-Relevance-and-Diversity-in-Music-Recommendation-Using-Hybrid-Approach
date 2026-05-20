from src.pipelines.inference_pipeline import inference_pipeline

if __name__ == "__main__":
    user_id = input("Enter user ID: ")
    query = input("Enter query: ")
    recommendations = inference_pipeline(user_id, query)
    print(recommendations)