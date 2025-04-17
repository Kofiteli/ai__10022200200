# Name: Kofi Boateng Index_number: 10022200200
import streamlit as st
from sections import regression, clustering, neural_network, llm

st.set_page_config(page_title="AI Project", layout="wide")
st.title("AI Project Dashboard")

# Navigation
page = st.sidebar.selectbox("Choose a Task", ["Regression", "Clustering", "Neural Network", "LLM"])

if page == "Regression":
    regression.run()

elif page == "Clustering":
    clustering.run()

elif page == "Neural Network":
    neural_network.run()

elif page == "LLM":
    llm.run()
