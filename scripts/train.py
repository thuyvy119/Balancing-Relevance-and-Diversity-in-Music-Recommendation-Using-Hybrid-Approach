import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from src.pipelines.train_pipeline import train_pipeline

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    pipeline_objects = train_pipeline(train_data_path="data/train_set.csv")
    print("Training completed successfully!")