# RAG-Optimization-System
Develop a Retrieval-Augmented Generation (RAG) system, benchmark its performance using RAGAS, and implement an optimization to improve the results.
---

### Goal
To create the best possible RAG system with the most effective strategies to provide optimized rag pipeline

---

### Evaluation
To evaluate the RAG pipeline RAGAS syntetic test data is used. 

By evaluating the RAG pipeline's performance on these datasets, we can assess the effectiveness of the current RAG strategies. Based on the evaluation results, we will update and refine the RAG strategies to improve the bot's question-answering capabilities.

The overall workflow is to use the RAGAS evaluation to assess the RAG pipeline, iterating on the RAG strategies based on the evaluation findings to continuously enhance the overall system.

---

### Variables Affecting RAG Performance
1. **User query**
2. **Chunking mechanism**
3. **Chunk Ranking**
4. **Retriver Performance**

By modifying these variables, we can improve the RAG performance.

---

### RAGAS Evaluation Metrics
- **Faithfulness:** Measures the factual consistency of the generated answer against the given context. The answer is scaled to a (0,1) range; higher is better.
- **Answer Relevancy:** Assesses how pertinent the generated answer is to the given prompt. Lower scores are assigned to incomplete or redundant answers, and higher scores indicate better relevancy.
- **Context Recall:** Measures the extent to which the retrieved context aligns with the annotated answer, treated as the ground truth. Values range between 0 and 1, with higher values indicating better performance.
- **Context Precision:** Evaluates whether all of the ground-truth relevant items present in the contexts are ranked higher or not.

Based on the metrics we have evaluated, we can determine which strategies to modify to improve overall performance.

---

### Strategies

- **Simple RAG system**
- **Multiquery RAG**
- **RAG Fusion**

---

### Installation
```sh
# Create a virtual environment
python -m venv env

# Activate the virtual environment
# For Windows
env\Scripts\activate
# For MacOS/Linux
source env/bin/activate

# Install backend dependencies
pip install -r requirements.txt

# Install frontend dependencies
pip install -r frontend_requirements.txt
```
---
### Usage 
```sh
git clone https://github.com/JabezETH/RAG-Optimization-System.git
cd RAG-Optimization-System
poetry init
```
