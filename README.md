# Kofi Boateng_10022200200 Streamlit AI Application

## Project Overview
This Streamlit-based application demonstrates core Artificial Intelligence concepts and tools through interactive interfaces for:

- **Regression**: Simple linear regression with user-uploaded CSV datasets.
- **Clustering**: K-Means clustering on multi-feature data with adjustable cluster count.
- **Neural Network**: Feedforward neural network for classification using TensorFlow.
- **Large Language Model (LLM) Q&A**: Retrieval-Augmented Generation (RAG) approach for question-and-answer tasks on custom data sources.

## Author
- **Name**: Kofi Boateng
- **Index Number**: 10022200200

## Repository Name
```
ai_10022200200
```

---

## Prerequisites
- Python 3.8 or higher
- `pip` package installer

## Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/Kofiteli/ai__10022200200.git
   cd ai_10022200200
   ```
2. **Create and activate a virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate    # Windows
   ```
3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Configuration
1. **Secrets**: Create a `secrets.toml` in the project root for the Gemini API key:
   ```toml
   [GEMINI]
   GEMINI_API_KEY = "<your_gemini_api_key_here>"
   ```
2. **Data files**: Place any default datasets in the project root (e.g., `Ghana_Election_Result.csv`, `2025-Budget-Statement-and-Economic-Policy_v4.pdf`).

## Running the Application
Launch the Streamlit dashboard:
```bash
streamlit run app.py
```
Navigate to the displayed URL (usually `http://localhost:8501`) to access the unified dashboard.

---

## Application Sections

### 1️⃣ Regression
- **Upload**: CSV file and specify the **target column** name.
- **Model**: Simple linear regression (sklearn).
- **Output**:
  - Preview of dataset.
  - Performance metrics: Mean Absolute Error, R² score.
  - Scatter plot: Predictions vs. Actual.
  - Custom input form for real-time predictions.

### 2️⃣ Clustering
- **Upload**: CSV file with numeric features.
- **Algorithm**: K-Means clustering.
- **Controls**: Slider to select number of clusters.
- **Visualization**: 2D scatter (and 3D if three features).
- **Export**: Download clustered dataset with labels.

### 3️⃣ Neural Network
- **Upload**: Classification CSV and specify **target column**.
- **Model**: TensorFlow Feedforward Neural Network.
- **Hyperparameters**: Epochs, learning rate, batch size.
- **Visualization**: Live training/validation loss and accuracy plots.
- **Inference**: Input fields for custom samples, real-time predictions.

### 4️⃣ Large Language Model (LLM) Q&A
- **Approach**: Retrieval-Augmented Generation (RAG) using Google Generative AI or Hugging Face model.
- **Datasets**:
  - `2025-Budget-Statement-and-Economic-Policy_v4.pdf`
  - `Ghana_Election_Result.csv`
- **Architecture**: PDF/CSV → Text splitters → Vector store (FAISS/Chroma) → LLM.
- **Interface**:
  - Input box for user queries.
  - Display generated answers with confidence scores.

---

## Project Structure
```
├── app.py                # Main Streamlit dashboard
├── regression.py         # Regression module
├── clustering.py         # Clustering module
├── neural_network.py     # Neural network module
├── llm.py                # LLM Q&A module
├── requirements.txt      # Python dependencies
├── 2025-Budget-Statement-and-Economic-Policy_v4.pdf  # Sample PDF data
└── Ghana_Election_Result.csv                         # Sample CSV data
```

### Usage

1. Optionally upload a different pdf or use the default.
2. Expand **"Show context sent to model"** to inspect the text block.
3. Enter a question (e.g., “What is the theme of the 2025 Budget Statement and Economic Policy?”).
4. View the model’s answer in real time.

### Sample Questions

- “What is the theme of the 2025 Budget Statement and Economic Policy?”
- “List the main fiscal policy objectives for 2025.”
- “What was Ghana’s end‑period inflation in 2024 and how did it compare to the target?”
- “Summarize the state of Ghana’s economy at the end of 2024 under the IMF‑supported programme.”
- “How much were total central government payables at end‑2024, and what were the biggest categories?”
- “What are the key energy‑sector fiscal risks highlighted in the document?”
- “Describe Ghana’s performance on the ECOWAS convergence criteria as of December 2024.”
- “What are the projected debt‑service obligations for 2025–2028?”
- “Which stalled bilateral‑loan projects were identified and how long will it take to complete them?”
- “What medium‑term macroeconomic targets has government set for 2025–2026?”

---

## Deployment
1. **GitHub**: https://github.com/Kofiteli/ai__10022200200.git
2. **Collaborator**:`godwin.danso@acity.edu.gh` or `GodwinDansoAcity` 
3. **Cloud Deployment**: ai--10022200200.streamlit.app/

---


