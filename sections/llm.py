# Name: Kofi Boateng Index_number: 10022200200
import os
import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import pipeline
from PyPDF2 import PdfReader
import google.generativeai as genai

def run():
    st.subheader("🤖 LLM - Q&A from Budget Statement and Economic Policy")

    st.markdown("We’re using Gemini API for question and answering.")

    # 1️⃣ Grab the key from secrets
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except KeyError:
        st.error("🔑 Missing GEMINI_API_KEY in secrets.toml")
        st.stop()

    # 2️⃣ Configure Gemini
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-pro")
    except Exception as e:
        st.error(f"❌ Gemini init error: {e}")
        st.stop()

    st.success("✅ Gemini API configured")

    # 3️⃣ Document upload / default load
    uploaded = st.file_uploader("Upload a PDF document", type="pdf")
    if uploaded:
        reader = PdfReader(uploaded)
    else:
        try:
            reader = PdfReader("2025-Budget-Statement-and-Economic-Policy_v4.pdf")
            st.info("ℹ️ Using default budget PDF")
        except FileNotFoundError:
            st.error("⚠️ No PDF found; please upload one.")
            st.stop()

    text = "".join(page.extract_text() or "" for page in reader.pages)
    st.write(f"📄 Document has {len(reader.pages)} pages")

    # 4️⃣ Question input & call
    question = st.text_area("Ask your question about the document:")
    if st.button("Get Answer") and question.strip():
        prompt = f"""
        Answer only using the text below. If you don’t see the answer, say “Not in document.”

        DOCUMENT:
        {text[:15000]}

        QUESTION:
        {question}

        ANSWER:
        """
        with st.spinner("🔍 Thinking..."):
            try:
                response = model.generate_content(prompt)
                st.markdown(f"**Answer:** {response.text}")
            except Exception as e:
                st.error(f"❌ Generation error: {e}")
