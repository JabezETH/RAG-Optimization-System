
# Importing libraries
import sys
import json
from dotenv import load_dotenv
import pandas as pd
load_dotenv()
sys.path.insert(1, '/home/jabez/rizzbuzz with poetry/RAG-Optimization-System/scripts')
import file_loader 

# Load JSON from file
json_path = '../filepath.json'

with open(json_path, 'r') as json_file:
    file_paths = json.load(json_file)
data_file_path = file_paths['data_file_path']

# loading data
data = file_loader.load_csv(data_file_path)

# RecursiveCharacterTextSplitter with 500 chunking size vector store
chunk_size= 500
chunk_overlap= 150
persist_directory = '../smaller_vector_db'
vectorstore_character = file_loader.character_text_splitter_large_embedding(data, chunk_size, chunk_overlap, persist_directory)

# RecursiveCharacterTextSplitter with 1000 chunking size vector store
persist_directory=persist_directory = '/large_vector_db'
chunk_size= 1000
chunk_overlap= 250

persist_directory = '../semantic_vector_db'

# SemanticTextSplitter vector store
vectorstore_character = file_loader.semantic_text_splitter(data, chunk_size, chunk_overlap, persist_directory)

# text-embedding-3-large vectore store
chunk_size= 500
chunk_overlap= 150
persist_directory = '../text_embedding_large'
vectorstore_character = file_loader.character_text_splitter_large_embedding(data, chunk_size, chunk_overlap, persist_directory)
