import sys
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.load import dumps, loads
from uuid import uuid4
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from langsmith import Client

# Insert the custom path to sys.path
sys.path.insert(1, '/home/jabez/week_11/Contract-Advisor-RAG/scripts/data_processing.py')
load_dotenv()

# Setting environment variables for Langchain
os.environ["LANGCHAIN_TRACING_V2"] = "true"
unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_PROJECT"] = f"Rag optimization system - {unique_id}"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
client = Client()

def simple_pipeline(vectorstore, question):
    """
    Execute a simple RAG pipeline with a given vector store and question.

    Args:
        vectorstore (Chroma): The vector store containing the document embeddings.
        question (str): The question to be answered.

    Returns:
        str: The generated answer.
    """
    try:
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
        
        # Define the prompt template
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            Answer the question based on only the given context. If there are no good answers, say "No good answers found".

            Context:
            {context}

            Question:
            {question}

            Answer:
            """
        )
        
        llm = ChatOpenAI(model="gpt-4", temperature=0.5)

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
    except Exception as e:
        print(f"An error occurred in simple_pipeline: {e}")

def multi_query_pipeline(vectorstore, question):
    """
    Execute a multi-query RAG pipeline with a given vector store and question.

    Args:
        vectorstore (Chroma): The vector store containing the document embeddings.
        question (str): The question to be answered.

    Returns:
        str: The generated answer.
    """
    try:
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
            """Unique union of retrieved documents."""
            flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
            unique_docs = list(set(flattened_docs))
            return [loads(doc) for doc in unique_docs]

        retrieval_chain = generate_queries | retriever.map() | get_unique_union
        docs = retrieval_chain.invoke({"question": question})

        template = """Answer the question based on only the given context. If there are no good answers, say "No good answers found".

        {context}

        Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)

        llm = ChatOpenAI(model="gpt-4", temperature=0.8)

        final_rag_chain = (
            {"context": retrieval_chain, 
            "question": itemgetter("question")} 
            | prompt
            | llm
            | StrOutputParser()
        )

        answer = final_rag_chain.invoke({"question": question})
        return answer
    except Exception as e:
        print(f"An error occurred in multi_query_pipeline: {e}")

def rag_fusion(vectorstore, question):
    """
    Execute a RAG-fusion pipeline with a given vector store and question.

    Args:
        vectorstore (Chroma): The vector store containing the document embeddings.
        question (str): The question to be answered.

    Returns:
        str: The generated answer.
    """
    try:
        llm = ChatOpenAI(model="gpt-4", temperature=0.8)
        retriever = vectorstore.as_retriever()
        
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
            """Reciprocal rank fusion that takes multiple lists of ranked documents and an optional parameter k used in the RRF formula."""
            fused_scores = {}

            for docs in results:
                for rank, doc in enumerate(docs):
                    doc_str = dumps(doc)
                    if doc_str not in fused_scores:
                        fused_scores[doc_str] = 0
                    previous_score = fused_scores[doc_str]
                    fused_scores[doc_str] += 1 / (rank + k)

            reranked_results = [
                (loads(doc), score)
                for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
            ]
            return reranked_results

        retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion

        template = """Answer the question using only the information provided in the context. If the context does not offer a suitable answer, reply with 'No good answers found:

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

        answer = final_rag_chain.invoke({"question": question})
        return answer
    except Exception as e:
        print(f"An error occurred in rag_fusion: {e}")
