# # Name: Kofi Boateng Index_number: 10022200200

import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai

def run():
    st.subheader("📚 LLM Question Answering with Gemini")
    st.markdown("We’re using Gemini API for question and answering.")

    # 1️⃣ Load your Gemini key from Streamlit secrets.toml
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except KeyError:
        st.error("🔒 Missing `GEMINI_API_KEY` in your secrets.toml.")
        st.stop()

    # 2️⃣ Configure Gemini
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-pro")
        st.success("✅ Gemini API connected successfully")
    except Exception as e:
        st.error(f"❌ Failed to configure Gemini: {e}")
        st.stop()

    # 3️⃣ Document upload / default load
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    if uploaded_file:
        reader = PdfReader(uploaded_file)
        st.success(f"📄 Processed {len(reader.pages)} pages from your upload")
    else:
        try:
            reader = PdfReader("2025-Budget-Statement-and-Economic-Policy_v4.pdf")
            st.info("ℹ️ Using default Ghana Budget document")
        except FileNotFoundError:
            st.error("⚠️ Default PDF not found – please upload one.")
            st.stop()

    # Extract text
    text = "".join(page.extract_text() or "" for page in reader.pages)

    # 4️⃣ Question input & call
    st.subheader("💬 Ask Your Question")
    question = st.text_area("Enter your question about the document:")
    if st.button("Get Answer") and question.strip():
        prompt = f"""
Answer ONLY using the text below. If you don’t see the answer, say "This information is not in the document."

DOCUMENT CONTEXT:
{text[:15000]}

QUESTION:
{question}

ANSWER:
"""
        with st.spinner("🔍 Analyzing document…"):
            try:
                response = model.generate_content(prompt)
                st.markdown(f"**Answer:** {response.text}")
            except Exception as e:
                st.error(f"❌ Generation error: {e}")
