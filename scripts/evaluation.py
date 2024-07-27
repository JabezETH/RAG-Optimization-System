# Importing libraries
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from datasets import Dataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from dotenv import load_dotenv
load_dotenv()

import os
from langsmith import Client
from uuid import uuid4

os.environ["LANGCHAIN_TRACING_V2"] = "true"

unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_PROJECT"] = f"Rag optimization system - {unique_id}"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
client = Client()

def generate_syntetic_testdata(documents, file_path):
    # generator with openai models
    generator_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
    critic_llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    embeddings = OpenAIEmbeddings()

    generator = TestsetGenerator.from_langchain(
        # rag_chain,
        generator_llm,
        critic_llm,
        embeddings
    )

    # generate testset
    testset = generator.generate_with_langchain_docs(documents, test_size=10, distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25})
    test_data = testset.to_pandas()
    test_data.to_csv(file_path, index=False)
    return test_data

def adding_answer_to_testdata(test_data, rag_pipeline, vector, retriever,file_path):
    questions = test_data['question'].to_list()
    ground_truth = test_data['ground_truth'].to_list()
    data = {'question': [], 'answer': [], 'contexts': [], 'ground_truth': ground_truth}
    
    for query in questions:
        data['question'].append(query)
        
        # Generate the chatbot response
        data['answer'].append(rag_pipeline(vector, query))
        
        # Retrieve relevant documents
        data['contexts'].append([doc.page_content for doc in retriever.get_relevant_documents(query)])
    
    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)
    dataset.to_csv(file_path, index=False)
    return dataset

def ragas_evaluator(dataset):
    from ragas.metrics import (
        answer_relevancy,
        faithfulness,
        context_recall,
        context_precision,
    )

    from ragas import evaluate

    result = evaluate(
        dataset = dataset,
        metrics=[
            context_precision,
            faithfulness,
            answer_relevancy,
            context_recall,
        ],
    )
    result
    evaluation_result = result.to_pandas()
    return evaluation_result