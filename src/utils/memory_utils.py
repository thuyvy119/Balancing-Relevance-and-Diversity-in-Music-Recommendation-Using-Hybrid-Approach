import pandas as pd
import numpy as np
import gc
import psutil
import os

def get_memory_usage():
    """Get current memory usage of the process"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # convert to MB

def reduce_mem_usage(df, verbose=True):
    """Reduce memory usage of a dataframe by downcasting numeric columns"""
    start_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

def process_in_chunks(df, chunk_size=1000, operation=None):
    """Process a dataframe in chunks to avoid memory issues"""
    result_chunks = []
    
    for i in range(0, len(df), chunk_size):
        # get chunk
        chunk = df.iloc[i:i+chunk_size].copy()
        
        # apply operation if provided
        if operation:
            chunk = operation(chunk)
        
        # reduce memory usage
        chunk = reduce_mem_usage(chunk, verbose=False)
        
        result_chunks.append(chunk)
        
        # clean up memory
        gc.collect()
        
        if i % 10000 == 0:
            print(f'Processed {i} rows. Current memory usage: {get_memory_usage():.2f} MB')
    
    return pd.concat(result_chunks, ignore_index=True)

def clean_text_columns(df):
    """Clean text columns by removing unnecessary whitespace and converting to string"""
    text_columns = df.select_dtypes(include=['object']).columns
    for col in text_columns:
        df[col] = df[col].astype(str).str.strip()
    return df

def save_in_chunks(df, filename, chunk_size=1000):
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        mode = 'w' if i == 0 else 'a'
        chunk.to_pickle(filename + f'.part{i//chunk_size}')
        
        if i % 10000 == 0:
            print(f'Saved {i} rows. Current memory usage: {get_memory_usage():.2f} MB')

def load_in_chunks(filename_pattern, chunk_size=1000):
    """Load a dataframe from disk in chunks"""
    import glob
    
    chunks = []
    for filename in sorted(glob.glob(filename_pattern + '.part*')):
        chunk = pd.read_pickle(filename)
        chunk = reduce_mem_usage(chunk, verbose=False)
        chunks.append(chunk)
        
        if len(chunks) % 10 == 0:
            print(f'Loaded {len(chunks)} chunks. Current memory usage: {get_memory_usage():.2f} MB')
    
    return pd.concat(chunks, ignore_index=True) 
