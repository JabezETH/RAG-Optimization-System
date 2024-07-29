import os
import shutil
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from typing import List


def load_csv(file_path):
    """
    Load a CSV file from the given file path.
    Args:
        file_path (str): The path to the CSV file.
    Returns:
        pandas.DataFrame: The loaded CSV data as a langchain DataFrame.
    """
    loader = CSVLoader(file_path, source_column="highlights")

    data = loader.load()
    return data


def character_text_splitter(docs: List[str], chunk_size: int, chunk_overlap: int) -> Chroma:
    """
    Split the given documents into chunks of text using the RecursiveCharacterTextSplitter and
    create a persistent Chroma vector store, replacing any existing data.
    
    Args:
        docs (List[str]): A list of documents to be split.
        chunk_size (int): The size of each text chunk.
        chunk_overlap (int): The overlap between consecutive text chunks.
        persist_directory (str): The directory to store the persistent vector store files.

    Returns:
        Chroma: A Chroma object containing the split documents and their embeddings.
    """
    persist_directory=persist_directory = '/home/jabez/rizzbuzz with poetry/RAG-Optimization-System/vector_store'
    # Remove the existing persistence directory if it exists
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    # Initialize the text splitter with the specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Split the documents into smaller chunks
    splits = text_splitter.split_documents(docs)
    
    # Create a Chroma vector store with persistence enabled
    vectorstore = Chroma.from_documents(splits, embedding=OpenAIEmbeddings(), persist_directory=persist_directory)
    
    return vectorstore



def semantic_text_splitter(docs):
    """
    Split the given documents into chunks of text using the SemanticChunker and OpenAIEmbeddings.
    
    Args:
        docs (List[str]): A list of documents to be split.
        
    Returns:
        Chroma: A Chroma object containing the split documents and their embeddings.
    """
    text_splitter = SemanticChunker(OpenAIEmbeddings(), breakpoint_threshold_type="percentile")
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    return vectorstore