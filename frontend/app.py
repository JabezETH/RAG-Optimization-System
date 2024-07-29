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
synthetic_test_data = pd.read_csv('/home/jabez/rizzbuzz with poetry/RAG-Optimization-System/test_data/syntetic_test_data.csv')
st.write("### Synthetic Test Data Sample")
st.dataframe(synthetic_test_data.head(19))

st.write("### Simple Rag Pipeline Evaluation")
# Create a button to execute the function
if st.button("Execute Test"):
    try:
        # Add a progress bar
        progress_bar = st.progress(0)
        progress_bar.progress(10)

        st.write("- Initializing Embeddings and Database")
        embeddings = OpenAIEmbeddings()
        db = Chroma(persist_directory="/home/jabez/rizzbuzz with poetry/RAG-Optimization-System/vector_store", embedding_function=embeddings)
        progress_bar.progress(30)

        st.write("- Setting up Retriever")
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 6})
        progress_bar.progress(50)

        st.write("- Adding Answers to Test Data")
        synthetic_test_data_with_answer = evaluation.adding_answer_to_testdata(synthetic_test_data, pipelines.simple_pipeline, db, retriever)
        progress_bar.progress(70)

        st.write("- Evaluating the Test Data")
        simple_rag_evaluation_result = evaluation.ragas_evaluator(synthetic_test_data_with_answer)
        progress_bar.progress(90)

        # Calculate and display the result
        result = evaluation.evaluation_mean(simple_rag_evaluation_result)
        st.success(f"Evaluation Mean Result: {result}")
        progress_bar.progress(100)

    except ValueError:
        st.error("Please enter valid integer values for chunk size and chunk overlap.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

st.write("### Simple Rag Pipeline Evaluation with 1000 chunking size")
# Create a button to execute the function
if st.button("Execute Test"):
    try:
        # Add a progress bar
        progress_bar = st.progress(0)
        progress_bar.progress(10)

        st.write("- Initializing Embeddings and Database")
        embeddings = OpenAIEmbeddings()
        db = Chroma(persist_directory="/home/jabez/rizzbuzz with poetry/RAG-Optimization-System/vector_store", embedding_function=embeddings)
        progress_bar.progress(30)

        st.write("- Setting up Retriever")
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 6})
        progress_bar.progress(50)

        st.write("- Adding Answers to Test Data")
        synthetic_test_data_with_answer = evaluation.adding_answer_to_testdata(synthetic_test_data, pipelines.simple_pipeline, db, retriever)
        progress_bar.progress(70)

        st.write("- Evaluating the Test Data")
        simple_rag_evaluation_result = evaluation.ragas_evaluator(synthetic_test_data_with_answer)
        progress_bar.progress(90)

        # Calculate and display the result
        result = evaluation.evaluation_mean(simple_rag_evaluation_result)
        st.success(f"Evaluation Mean Result: {result}")
        progress_bar.progress(100)

    except ValueError:
        st.error("Please enter valid integer values for chunk size and chunk overlap.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

st.write("### Simple Rag Pipeline Evaluation with 500 chunking size")
# Create a button to execute the function
if st.button("Execute Test"):
    try:
        # Add a progress bar
        progress_bar = st.progress(0)
        progress_bar.progress(10)

        st.write("- Initializing Embeddings and Database")
        embeddings = OpenAIEmbeddings()
        db = Chroma(persist_directory="/home/jabez/rizzbuzz with poetry/RAG-Optimization-System/vector_store", embedding_function=embeddings)
        progress_bar.progress(30)

        st.write("- Setting up Retriever")
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 6})
        progress_bar.progress(50)

        st.write("- Adding Answers to Test Data")
        synthetic_test_data_with_answer = evaluation.adding_answer_to_testdata(synthetic_test_data, pipelines.simple_pipeline, db, retriever)
        progress_bar.progress(70)

        st.write("- Evaluating the Test Data")
        simple_rag_evaluation_result = evaluation.ragas_evaluator(synthetic_test_data_with_answer)
        progress_bar.progress(90)

        # Calculate and display the result
        result = evaluation.evaluation_mean(simple_rag_evaluation_result)
        st.success(f"Evaluation Mean Result: {result}")
        progress_bar.progress(100)

    except ValueError:
        st.error("Please enter valid integer values for chunk size and chunk overlap.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

st.write("### Simple Rag Pipeline Evaluation with semantic chunking")
# Create a button to execute the function
if st.button("Execute Test"):
    try:
        # Add a progress bar
        progress_bar = st.progress(0)
        progress_bar.progress(10)

        st.write("- Initializing Embeddings and Database")
        embeddings = OpenAIEmbeddings()
        db = Chroma(persist_directory="/home/jabez/rizzbuzz with poetry/RAG-Optimization-System/vector_store", embedding_function=embeddings)
        progress_bar.progress(30)

        st.write("- Setting up Retriever")
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 6})
        progress_bar.progress(50)

        st.write("- Adding Answers to Test Data")
        synthetic_test_data_with_answer = evaluation.adding_answer_to_testdata(synthetic_test_data, pipelines.simple_pipeline, db, retriever)
        progress_bar.progress(70)

        st.write("- Evaluating the Test Data")
        simple_rag_evaluation_result = evaluation.ragas_evaluator(synthetic_test_data_with_answer)
        progress_bar.progress(90)

        # Calculate and display the result
        result = evaluation.evaluation_mean(simple_rag_evaluation_result)
        st.success(f"Evaluation Mean Result: {result}")
        progress_bar.progress(100)

    except ValueError:
        st.error("Please enter valid integer values for chunk size and chunk overlap.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

st.write("### Multi Query Rag Pipeline Evaluation")
# Create a button to execute the function
if st.button("Execute Test"):
    try:
        # Add a progress bar
        progress_bar = st.progress(0)
        progress_bar.progress(10)

        st.write("- Initializing Embeddings and Database")
        embeddings = OpenAIEmbeddings()
        db = Chroma(persist_directory="/home/jabez/rizzbuzz with poetry/RAG-Optimization-System/vector_store", embedding_function=embeddings)
        progress_bar.progress(30)

        st.write("- Setting up Retriever")
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 6})
        progress_bar.progress(50)

        st.write("- Adding Answers to Test Data")
        synthetic_test_data_with_answer = evaluation.adding_answer_to_testdata(synthetic_test_data, pipelines.simple_pipeline, db, retriever)
        progress_bar.progress(70)

        st.write("- Evaluating the Test Data")
        simple_rag_evaluation_result = evaluation.ragas_evaluator(synthetic_test_data_with_answer)
        progress_bar.progress(90)

        # Calculate and display the result
        result = evaluation.evaluation_mean(simple_rag_evaluation_result)
        st.success(f"Evaluation Mean Result: {result}")
        progress_bar.progress(100)

    except ValueError:
        st.error("Please enter valid integer values for chunk size and chunk overlap.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

    
st.write("### Rag Fusion Rag Pipeline Evaluation")
# Create a button to execute the function
if st.button("Execute Test"):
    try:
        # Add a progress bar
        progress_bar = st.progress(0)
        progress_bar.progress(10)

        st.write("- Initializing Embeddings and Database")
        embeddings = OpenAIEmbeddings()
        db = Chroma(persist_directory="/home/jabez/rizzbuzz with poetry/RAG-Optimization-System/vector_store", embedding_function=embeddings)
        progress_bar.progress(30)

        st.write("- Setting up Retriever")
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 6})
        progress_bar.progress(50)

        st.write("- Adding Answers to Test Data")
        synthetic_test_data_with_answer = evaluation.adding_answer_to_testdata(synthetic_test_data, pipelines.simple_pipeline, db, retriever)
        progress_bar.progress(70)

        st.write("- Evaluating the Test Data")
        simple_rag_evaluation_result = evaluation.ragas_evaluator(synthetic_test_data_with_answer)
        progress_bar.progress(90)

        # Calculate and display the result
        result = evaluation.evaluation_mean(simple_rag_evaluation_result)
        st.success(f"Evaluation Mean Result: {result}")
        progress_bar.progress(100)

    except ValueError:
        st.error("Please enter valid integer values for chunk size and chunk overlap.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")