import os
import pandas as pd

def save_embeddings(embeddings_df:pd.DataFrame, filepath:str, format:str='csv', set_column_names:bool=False):
    """Save the embeddings to a file."""
    if(set_column_names is True):
        # Apply the columns name [id, dim_1, ... dim_d]."""
        columns = [f'dim_{i}' for i in range(embeddings_df.shape[1])]
        columns[0] = 'id'
        embeddings_df.columns = columns
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if(format=='csv'):
        embeddings_df.to_csv(filepath, index=False)
    elif(format=='pkl'):
        embeddings_df.to_pickle(filepath)
    print(f"Embeddings saved to {filepath}")

