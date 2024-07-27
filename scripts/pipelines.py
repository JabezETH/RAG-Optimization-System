import sys
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.load import dumps, loads
import os
from uuid import uuid4
from langchain.prompts import ChatPromptTemplate
from langchain.load import dumps, loads
from operator import itemgetter

sys.path.insert(1, '/home/jabez/week_11/Contract-Advisor-RAG/scripts/data_processing.py')
load_dotenv()
import getpass
import os
from langsmith import Client

os.environ["LANGCHAIN_TRACING_V2"] = "true"



unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_PROJECT"] = f"Rag optimization system - {unique_id}"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
client = Client()

# First RAG pipeline
def simple_pipeline(vectorstore, question):
    """
    Generates an answer to a given question using a simple pipeline.
    Args:
        vectorstore (VectorStore): The vector store containing the contracts.
        question (str): The question to be answered.
    Returns:
        str: The answer to the question.
    Description:
    This function implements a simple pipeline for generating answers to questions based on contracts. It uses a retriever to retrieve the most similar contracts to the question, and then uses a prompt template to generate the answer. The prompt template includes the context of the retrieved contracts and the question itself. The answer is generated using a language model (ChatOpenAI with model "gpt-4o" and temperature 0.5).
    The function first creates a retriever using the given vectorstore. It then defines a prompt template with the input variables "context" and "question". The template includes the context of the retrieved contracts and the question itself. The function also defines a lambda function to format the retrieved documents into a string.
    The function creates a chain using the prompt template and the retriever. The chain is invoked with the question to generate the answer. The answer is returned as a string.
    Example:
        >>> vectorstore = ...
        >>> question = "What is the purpose of this contract?"
        >>> answer = simple_pipeline(vectorstore, question)
        >>> print(answer)
        The purpose of this contract is to ...
    """
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    
    # Define the prompt template
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are a contract assistant.
        Use the following contract as a context to answer the question. Do not use any outside knowledge. 
        If the answer is not in the context, simply state "The information is not available in the provided context."

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
    )
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.5)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Create the chain with the prompt template
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RunnableLambda(lambda inputs: prompt_template.format(**inputs))
        | llm
        | StrOutputParser()
    )

    # Invoke the chain with the question
    answer = rag_chain.invoke(question)
    return answer


def multi_query_pipeline(vectorstore, question):
    """
    Generates a multi-query pipeline to retrieve relevant documents from a vector database and answer a user question.
    
    Args:
        vectorstore (VectorStore): The vector store used for retrieval.
        question (str): The user question.
    
    Returns:
        str: The answer to the user question based on the retrieved documents.
    """
    retriever = vectorstore.as_retriever()
    template = """You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines. Original question: {question}"""
    prompt_perspectives = ChatPromptTemplate.from_template(template)


    generate_queries = (
        prompt_perspectives 
        | ChatOpenAI(temperature=0) 
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
         )
    def get_unique_union(documents: list[list]):
        """ Unique union of retrieved docs """
        # Flatten list of lists, and convert each Document to string
        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
        # Get unique documents
        unique_docs = list(set(flattened_docs))
        # Return
        return [loads(doc) for doc in unique_docs]

    # Retrieve
    retrieval_chain = generate_queries | retriever.map() | get_unique_union
    docs = retrieval_chain.invoke({"question":question})

    # RAG
    template = """Answer the following question based on this context:

    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(model="gpt-4",temperature=0.8)

    final_rag_chain = (
        {"context": retrieval_chain, 
        "question": itemgetter("question")} 
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = final_rag_chain.invoke({"question":question})
    return answer

def rag_fusion(vectorstore, question):
    """
    This function performs a fusion of retrieval augmented generation (RAG) queries and documents to answer a user question.
    
    Args:
        vectorstore (VectorStore): The vector store used for retrieval.
        question (str): The user question to be answered.
        
    Returns:
        str: The answer to the user question based on the retrieved documents.
    """
    llm = ChatOpenAI(model="gpt-4",temperature=0.8)
    retriever = vectorstore.as_retriever()
    # RAG-Fusion: Related
    template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
    Generate multiple search queries related to: {question} \n
    Output (4 queries):"""
    prompt_rag_fusion = ChatPromptTemplate.from_template(template)

    generate_queries = (
    prompt_rag_fusion 
    | ChatOpenAI(temperature=0)
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
    )
    def reciprocal_rank_fusion(results: list[list], k=60):
        """ Reciprocal_rank_fusion that takes multiple lists of ranked documents and an optional parameter k used in the RRF formula """
        
        # Initialize a dictionary to hold fused scores for each unique document
        fused_scores = {}

        # Iterate through each list of ranked documents
        for docs in results:
            # Iterate through each document in the list, with its rank (position in the list)
            for rank, doc in enumerate(docs):
                # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
                doc_str = dumps(doc)
                # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                # Retrieve the current score of the document, if any
                previous_score = fused_scores[doc_str]
                # Update the score of the document using the RRF formula: 1 / (rank + k)
                fused_scores[doc_str] += 1 / (rank + k)

        # Sort the documents based on their fused scores in descending order to get the final reranked results
        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        return reranked_results
    retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion

    # RAG
    template = """Answer the following question based on this context:

    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    final_rag_chain = (
        {"context": retrieval_chain_rag_fusion, 
        "question": itemgetter("question")} 
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = final_rag_chain.invoke({"question":question})
    return answer

