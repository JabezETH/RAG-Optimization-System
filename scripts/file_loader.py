from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker


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
def character_text_splitter(docs,chunk_size, chunk_overlap):    
    """
    Split the given documents into chunks of text using the RecursiveCharacterTextSplitter.
    
    Args:
        docs (List[str]): A list of documents to be split.
        
    Returns:
        Chroma: A Chroma object containing the split documents and their embeddings.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
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