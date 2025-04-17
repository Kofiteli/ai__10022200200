# Name: Kofi Boateng Index_number: 10022200200
import os
import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import pipeline


def run():
    st.subheader("🤖 LLM - Q&A from Ghana Election Dataset (Hugging Face)")

    st.markdown("We’re using Hugging Face’s `distilbert-base-cased-distilled-squad` for question answering.")

    # Load dataset
    local_path = os.path.join("assets", "Ghana_Election_Result.csv")
    df = pd.read_csv(local_path)
    st.markdown("#### Preview of Ghana Election Dataset")
    st.dataframe(df.head())

    st.markdown("### 🧠 Ask a question about the dataset")

    # Pre-process the dataset into readable text
    # Combine key columns into plain text for the model
    context = ""
    for index, row in df.iterrows():
        row_text = f"Region: {row.get('Region', '')}, Constituency: {row.get('Constituency', '')}, Valid Votes: {row.get('Valid_Votes', '')}"
        context += row_text + "\n"

    # Load Hugging Face Question-Answering pipeline
    with st.spinner("Loading model..."):
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
        model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")
        qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, framework="pt")

    # Question input
    question = st.text_input("Type your question below 👇")

    if question:
        with st.spinner("Searching for the answer..."):
            result = qa_pipeline({
                'context': context,
                'question': question
            })
            st.success(f"Answer: {result['answer']}")
            st.caption(f"Confidence Score: {result['score']:.2f}")


