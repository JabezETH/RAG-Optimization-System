import streamlit as st
import pandas as pd
import sys

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

# Create input fields for the parameters
chunk_size = st.text_input("Enter the chunk size (integer)")
chunk_overlap = st.text_input("Enter the chunk overlap (integer)")

# Load synthetic test data
synthetic_test_data = pd.read_csv('/home/jabez/rizzbuzz with poetry/RAG-Optimization-System/test_data/syntetic_test_data.csv')

# Create a button to execute the function
if st.button("Execute Test"):
    try:
        # Convert input strings to integers
        chunk_size = int(chunk_size)
        chunk_overlap = int(chunk_overlap)

        # Call the function with the input parameters
        vectorstore_character = file_loader.character_text_splitter(data, chunk_size, chunk_overlap)
        retriever = vectorstore_character.as_retriever(search_type="similarity", search_kwargs={"k": 6})

        # Adding answer to test data from simple pipeline
        synthetic_test_data_with_answer = evaluation.adding_answer_to_testdata(synthetic_test_data, pipelines.simple_pipeline, vectorstore_character, retriever)

        # Evaluating the test data from simple pipeline
        simple_rag_evaluation_result = evaluation.ragas_evaluator(synthetic_test_data_with_answer)

        # Calculate and display the result
        result = evaluation.evaluation_mean(simple_rag_evaluation_result)
        st.success(f"Evaluation Mean Result: {result}")

    except ValueError:
        st.error("Please enter valid integer values for chunk size and chunk overlap.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
