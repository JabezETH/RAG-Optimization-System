# Importing libraries
import sys
import json
import unittest
from unittest.mock import patch, mock_open
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
load_dotenv()
sys.path.insert(1, '/home/jabez/rizzbuzz with poetry/RAG-Optimization-System/scripts')
import file_loader 
import pipelines 
import evaluation

class TestRAGOptimizationSystem(unittest.TestCase):

    @patch('builtins.open', new_callable=mock_open, read_data='{"data_file_path": "/home/jabez/rizzbuzz with poetry/RAG-Optimization-System/data/cnn_dailymail_3.0.0.csv"}')
    @patch('file_loader.load_csv')
    @patch('file_loader.character_text_splitter_large_embedding')
    @patch('file_loader.semantic_text_splitter')
    def test_main(self, mock_semantic_text_splitter, mock_character_text_splitter_large_embedding, mock_load_csv, mock_open):
        # Mock the environment loading
        with patch('dotenv.load_dotenv', return_value=True) as mock_load_dotenv:
            load_dotenv()
            mock_load_dotenv.assert_called_once()
        
        # Mock the file paths
        json_path = '/home/jabez/rizzbuzz with poetry/RAG-Optimization-System/filepath.json'
        
        # Define mock return values
        mock_load_csv.return_value = "mock_data"
        mock_character_text_splitter_large_embedding.return_value = "mock_vectorstore_character_large"
        mock_semantic_text_splitter.return_value = "mock_vectorstore_character_semantic"

        # Load JSON from file
        with open(json_path, 'r') as json_file:
            file_paths = json.load(json_file)
        
        # Verify file paths
        self.assertEqual(file_paths['data_file_path'], "/home/jabez/rizzbuzz with poetry/RAG-Optimization-System/data/cnn_dailymail_3.0.0.csv")

        # loading data
        data_file_path = file_paths['data_file_path']
        data = file_loader.load_csv(data_file_path)
        
        # Verify data loading
        mock_load_csv.assert_called_once_with("/home/jabez/rizzbuzz with poetry/RAG-Optimization-System/data/cnn_dailymail_3.0.0.csv")
        self.assertEqual(data, "mock_data")

        # RecursiveCharacterTextSplitter with 500 chunking size vector store
        chunk_size = 500
        chunk_overlap = 150
        persist_directory = '/home/jabez/rizzbuzz with poetry/RAG-Optimization-System/vector_store'
        vectorstore_character = file_loader.character_text_splitter_large_embedding(data, chunk_size, chunk_overlap, persist_directory)
        
        # Verify character_text_splitter_large_embedding call
        mock_character_text_splitter_large_embedding.assert_called_with("mock_data", chunk_size, chunk_overlap, persist_directory)
        self.assertEqual(vectorstore_character, "mock_vectorstore_character_large")

        # text-embedding-3-large vector store
        persist_directory = '../text_embedding_large'
        vectorstore_character = file_loader.character_text_splitter_large_embedding(data, chunk_size, chunk_overlap, persist_directory)
        
        # Verify character_text_splitter_large_embedding call
        mock_character_text_splitter_large_embedding.assert_called_with("mock_data", chunk_size, chunk_overlap, persist_directory)
        self.assertEqual(vectorstore_character, "mock_vectorstore_character_large")

        # RecursiveCharacterTextSplitter with 1000 chunking size vector store
        chunk_size = 1000
        chunk_overlap = 250
        persist_directory = '/home/jabez/rizzbuzz with poetry/RAG-Optimization-System/semantic_vector_db'

        # SemanticTextSplitter vector store
        vectorstore_character = file_loader.semantic_text_splitter(data, chunk_size, chunk_overlap, persist_directory)
        
        # Verify semantic_text_splitter call
        mock_semantic_text_splitter.assert_called_with("mock_data", chunk_size, chunk_overlap, persist_directory)
        self.assertEqual(vectorstore_character, "mock_vectorstore_character_semantic")

if __name__ == '__main__':
    unittest.main()