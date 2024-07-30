# Importing necessary libraries
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

# Setting environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"

unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_PROJECT"] = f"Rag optimization system - {unique_id}"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
client = Client()

def generate_syntetic_testdata(documents, file_path):
    """
    Generate synthetic test data using OpenAI models.

    Args:
        documents (list): A list of documents to generate test data from.
        file_path (str): The file path to save the generated test data CSV file.

    Returns:
        pandas.DataFrame: The generated synthetic test data as a pandas DataFrame.

    This function generates synthetic test data using OpenAI models. It takes a list of documents
    as input and generates a test set of 20 samples, with a distribution of 50% simple, 25% reasoning,
    and 25% multi_context. The generated test data is saved as a CSV file at the specified file path.
    The function returns the generated test data as a pandas DataFrame.
    """
    try:
        # Initialize generator with OpenAI models
        generator_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
        critic_llm = ChatOpenAI(model="gpt-4")
        embeddings = OpenAIEmbeddings()

        generator = TestsetGenerator.from_langchain(
            # rag_chain,
            generator_llm,
            critic_llm,
            embeddings
        )

        # Generate test set
        testset = generator.generate_with_langchain_docs(documents, test_size=20, distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25})
        test_data = testset.to_pandas()
        test_data.to_csv(file_path, index=False)
        return test_data
    except Exception as e:
        print(f"An error occurred while generating synthetic test data: {e}")

def adding_answer_to_testdata(test_data, rag_pipeline, vector, retriever):
    """
    Adds answers to the test data using a RAG pipeline and saves the result to a CSV file.

    Args:
        test_data (pandas.DataFrame): The test data containing questions and ground truth answers.
        rag_pipeline (callable): The RAG pipeline used to generate answers.
        vector (object): The vector used by the RAG pipeline.
        retriever (object): The retriever used to retrieve relevant documents.

    Returns:
        Dataset: The dataset containing the questions, answers, contexts, and ground truth answers.
    """
    try:
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
        return dataset
    except Exception as e:
        print(f"An error occurred while adding answers to the test data: {e}")

def ragas_evaluator(dataset):
    """
    Evaluates a given dataset using the RAGAS library's evaluation metrics.

    Args:
        dataset (Dataset): The dataset to be evaluated.

    Returns:
        pandas.DataFrame: A pandas DataFrame containing the evaluation results.
            The DataFrame has the following columns:
            - 'context_precision': The precision of the context retrieval.
            - 'faithfulness': The faithfulness of the generated answer.
            - 'answer_relevancy': The relevance of the generated answer.
            - 'context_recall': The recall of the context retrieval.
    """
    try:
        from ragas.metrics import (
            answer_relevancy,
            faithfulness,
            context_recall,
            context_precision,
        )

        from ragas import evaluate

        result = evaluate(
            dataset=dataset,
            metrics=[
                context_precision,
                faithfulness,
                answer_relevancy,
                context_recall,
            ],
        )
        evaluation_result = result.to_pandas()
        return evaluation_result
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")

def evaluation_mean(evaluation_result):
    """
    Calculate the mean values of the evaluation metrics from the given evaluation result.

    Args:
        evaluation_result (dict): A dictionary containing the evaluation metrics.

    Returns:
        str: A formatted string with the mean values of the evaluation metrics.

    Prints:
        The mean values of the evaluation metrics:
            - context_precision: The mean value of the context precision metric.
            - faithfulness: The mean value of the faithfulness metric.
            - answer_relevancy: The mean value of the answer relevancy metric.
            - context_recall: The mean value of the context recall metric.
    """
    try:
        context_precision = round(evaluation_result['context_precision'].mean() * 100, 2)
        faithfulness = round(evaluation_result['faithfulness'].mean() * 100, 2)
        answer_relevancy = round(evaluation_result['answer_relevancy'].mean() * 100, 2)
        context_recall = round(evaluation_result['context_recall'].mean() * 100, 2)
        
        result = (
            f'context_precision: {context_precision}%, '
            f'faithfulness: {faithfulness}%, '
            f'answer_relevancy: {answer_relevancy}%, '
            f'context_recall: {context_recall}%'
        )
        print(result)
        return result
    except KeyError as e:
        print(f"Missing key in evaluation result: {e}")
    except AttributeError as e:
        print(f"Ensure that the evaluation_result values support the 'mean' method: {e}")
    except Exception as e:
        print(f"An error occurred while calculating the mean evaluation metrics: {e}")
