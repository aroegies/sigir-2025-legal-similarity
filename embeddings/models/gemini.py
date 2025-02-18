
import os
import pickle
import glob

import pandas as pd
import google.generativeai as genai

import utils

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def get_embedding(text: str):
        result = genai.embed_content(model="models/text-embedding-004",
                                     content=text)
        return result['embedding']

def embed_gemini(clause_dir: str, experiment_name: str, clause_type: str):
    print(f"Embedding {clause_type}...")
    # each csv file contains a clause type
    glob_path = f"{clause_dir}/Similarity Clauses - {clause_type}*.csv"
    filepath = glob.glob(glob_path)
    print(filepath)
    df = pd.read_csv(filepath[0])
    # take first 20 rows
    df = df.head(20)
    # embed all the text for each column and pickle
    for col_name in df.columns.tolist():
        embeddings = []
        for i, text in enumerate(df[col_name].tolist()):
            print(f"Embedding {col_name} {i}")
            emb = get_embedding(text)
            embeddings.append(emb)
        output_dir = f"{experiment_name}/{clause_type}"
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/{utils.strip_stuff(col_name)}.pkl", 'wb') as w:
            print(f"Pickling {col_name}")
            pickle.dump(embeddings, w)

if __name__ == "__main__":

    # experiment_name is used to distinguish between different embedding models or configurations,
    # used to name output directory
    experiment_name = "gemini/text-embedding-004" # SET THIS CAREFULLY
    clause_dir = "../clauses"

    clause_types = [
        "Assignment",
        "Transfer of Data",
        "Exclusivity",
        "Non-Solicit",
        "Permitted Use of Data",
        "Audit Right",
        "License Grant",
        "MFN",
        "Publicity",
        "Termination for Convenience"
    ]


    for clause_type in clause_types:
        embed_gemini(clause_dir, experiment_name, clause_type)

