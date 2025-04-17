# AI Project Dashboard

This repository implements a Streamlit app that showcases four AI tasks:

1. **Regression**: Train and evaluate a linear regression model on user-uploaded data.
2. **Clustering**: Perform K-Means clustering with interactive visualization.
3. **Neural Network**: Train a feedforward neural network for classification with real-time training metrics.
4. **LLM Q&A (RAG)**: A Retrieval-Augmented Generation system using Hugging Face embeddings and a pre-trained Mistral-7B-Instruct model.

---

## Getting Started

### Prerequisites

- Python 3.11
- `venv` (virtual environment)

### Install Dependencies

```bash
python3.11 -m venv venv
source venv/bin/activate      # macOS/Linux
# or venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### Running the App

```bash
streamlit run app.py
```

Use the sidebar to navigate between tasks: Regression, Clustering, Neural Network, and LLM Q&A.

---

##  1. Regression Section

1. Upload a CSV file.
2. Select the **target column** (must be numeric).
3. Preview data and train a simple Linear Regression model.
4. View **Mean Absolute Error (MAE)** and **R² score**.
5. See an **Actual vs. Predicted** scatter plot (with regression line if only one feature).
6. Enter custom feature values to make new predictions.

---

##  2. Clustering Section

1. Upload a CSV file with numeric features.
2. Adjust the **number of clusters** via a slider.
3. Perform K-Means clustering.
4. View a 2D or 3D cluster plot with centroids.
5. Download the clustered dataset as CSV.

---

##  3. Neural Network Section

1. Upload a classification dataset (CSV).
2. Select the **target column**.
3. Choose **epochs** and **learning rate**.
4. Train a feedforward neural network (TensorFlow/Keras).
5. Monitor **training/validation loss and accuracy** graphs.
6. Input custom feature values for on-the-fly prediction.

---

##  4. LLM Q&A Section (RAG with Hugging Face)

### Dataset & Model Details

- **Dataset**: `Ghana_Election_Result.csv` (fields: Region, Constituency, Valid\_Votes).
- **Embedding Model**: `all-MiniLM-L6-v2` from Hugging Face Transformers.
- **LLM**: `mistralai/Mistral-7B-Instruct-v0.1` served via Hugging Face Hub.

### Methodology

1. **Load Data**: Read CSV into a Pandas DataFrame.
2. **Summarize Structure**: Compute number of constituencies per region.
3. **Context Generation**: Combine summary and row-level strings into a text block.
4. **Chunking**: Split into 1,000-character chunks with overlap.
5. **Embedding**: Create vector embeddings for each chunk.
6. **Indexing**: Build a FAISS in-memory vector store.
7. **Retrieve**: Use RetrievalQA chain to find relevant chunks.
8. **Generate**: Query Mistral-7B for final answer.
9. **Display**: Show answer and, optionally, confidence/metadata.

### Architecture Diagram

```mermaid
flowchart LR
    A[Streamlit UI] --> B[Data Loader (Pandas/DataFrameLoader)]
    B --> C[Text Splitter (CharacterTextSplitter)]
    C --> D[FAISS Vector Store]
    D --> E[Retriever]
    E --> F[LLM (Mistral-7B via HuggingFaceHub)]
    F --> G[Answer]
    G --> A
```

### Usage

1. Optionally upload a different CSV or use the default.
2. Expand **"Show context sent to model"** to inspect the text block.
3. Enter a question (e.g., "How many constituencies are in Ashanti Region?").
4. View the model’s answer in real time.

### Sample Questions

- "What is the valid vote count for Bantama?"
- "Which region has the most constituencies?"
- "Ashanti Region has how many constituencies?"

---

##  5. Comparison with ChatGPT

**Question:** "How many constituencies are in Ashanti Region?"

- **Our RAG Model:** "Ashanti Region has 18 constituencies."
- **ChatGPT Response:** "Ashanti Region has 47 constituencies."

**Analysis:** Our model pulls directly from the provided dataset, ensuring factual alignment. ChatGPT’s answer is based on external or outdated knowledge and may not match custom data.

---

##  6. Documentation & Collaboration

- **Repository Name:** `ai_10022200200` 
- **Collaborator:** Add `godwin.danso@acity.edu.gh` on GitHub.
- **Deployment:** Streamlit Cloud or similar; share the live URL.

---



