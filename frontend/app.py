import streamlit as st
import pandas as pd
import sys
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Insert path for custom modules
sys.path.insert(1, '/home/jabez/rizzbuzz with poetry/RAG-Optimization-System/scripts')

import file_loader
import pipelines
import evaluation

# Loading data
file_path = '/home/jabez/rizzbuzz with poetry/RAG-Optimization-System/data/cnn_dailymail_3.0.0.csv'
data = file_loader.load_csv(file_path)

# Display title and initial greeting
st.write("""
# RAG Optimization System
""")

# Display a sample of the loaded data
st.write("### Loaded Data Sample")
df = pd.read_csv(file_path)
st.dataframe(df.head())

# Load synthetic test data
synthetic_test_data_path = '/home/jabez/rizzbuzz with poetry/RAG-Optimization-System/test_data/syntetic_test_data.csv'
synthetic_test_data = pd.read_csv(synthetic_test_data_path)
st.write("### Synthetic Test Data Sample")
st.dataframe(synthetic_test_data.head(19))

# Initialize Embeddings and Database
st.write("- Initializing Embeddings and Database")
embeddings = OpenAIEmbeddings()
db = Chroma(persist_directory="/home/jabez/rizzbuzz with poetry/RAG-Optimization-System/vector_store", embedding_function=embeddings)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# Initialize session state to store results
if 'results' not in st.session_state:
    st.session_state.results = {}

# Define a function to execute evaluation
def execute_evaluation(pipeline, db, retriever, test_data, key):
    progress_bar = st.progress(0)
    try:
        progress_bar.progress(10)

        st.write("- Adding Answers to Test Data")
        test_data_with_answer = evaluation.adding_answer_to_testdata(test_data, pipeline, db, retriever)
        progress_bar.progress(70)

        st.write("- Evaluating the Test Data")
        evaluation_result = evaluation.ragas_evaluator(test_data_with_answer)
        progress_bar.progress(90)

        # Calculate and display the result
        result = evaluation.evaluation_mean(evaluation_result)
        st.session_state.results[key] = result
        st.success(f"Evaluation Mean Result: {result}")
        progress_bar.progress(100)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Display results from session state
def display_results():
    for key, result in st.session_state.results.items():
        st.write(f"### {key} Evaluation Result")
        st.write(f"Evaluation Mean Result: {result}")

# Simple Rag Pipeline Evaluation with 500 chunk size
simple_rag_500_expander = st.expander("Simple Rag Pipeline Evaluation with 500 chunk size")
if simple_rag_500_expander.button("Execute Simple Rag Pipeline (500 chunk size)"):
    execute_evaluation(pipelines.simple_pipeline, db, retriever, synthetic_test_data, "Simple Rag Pipeline (500 chunk size)")

# Simple Rag Pipeline Evaluation with 1000 chunk size
simple_rag_1000_expander = st.expander("Simple Rag Pipeline Evaluation with 1000 chunk size")
if simple_rag_1000_expander.button("Execute Simple Rag Pipeline (1000 chunk size)"):
    large_db = Chroma(persist_directory="/home/jabez/rizzbuzz with poetry/RAG-Optimization-System/large_vector_db", embedding_function=embeddings)
    large_retriever = large_db.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    execute_evaluation(pipelines.simple_pipeline, large_db, large_retriever, synthetic_test_data, "Simple Rag Pipeline (1000 chunk size)")

# Multi Query Rag Pipeline Evaluation
multi_query_expander = st.expander("Multi Query Rag Pipeline Evaluation")
if multi_query_expander.button("Execute Multi Query Rag Pipeline"):
    execute_evaluation(pipelines.multi_query_pipeline, db, retriever, synthetic_test_data, "Multi Query Rag Pipeline")

# Rag Fusion Rag Pipeline Evaluation
rag_fusion_expander = st.expander("Rag Fusion Rag Pipeline Evaluation")
if rag_fusion_expander.button("Execute Rag Fusion Rag Pipeline"):
    execute_evaluation(pipelines.rag_fusion, db, retriever, synthetic_test_data, "Rag Fusion Rag Pipeline")

# Display all results
st.write("### Evaluation Results")
display_results()
