import os
import pickle
import time
import streamlit as st
from langchain import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# Load API Key
load_dotenv()
openai_key = os.getenv("OPENAI_KEY")
if not openai_key:
    raise ValueError("OPENAI_KEY is not set in the environment variables.")

# Set page config
st.set_page_config(
    page_title="NewsGPT - AI Research Tool",
    page_icon="📈",
    layout="centered",
)

# Styling
st.markdown(
    """
    <style>
        .main-title { text-align: center; font-size: 36px; font-weight: bold; }
        .sidebar-title { font-size: 20px; font-weight: bold; }
        .footer { text-align: center; font-size: 14px; margin-top: 20px; }
        .info-box { padding: 10px; border-radius: 10px; background-color: #f1f3f6; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown('<p class="main-title">NewsGPT: AI-Powered News Research 📊</p>', unsafe_allow_html=True)
st.markdown("---")

st.sidebar.markdown('<p class="sidebar-title">News Article URL</p>', unsafe_allow_html=True)

# URL Input
url = st.sidebar.text_input(f"🔗 Enter URL")


# Process Button
process_url_clicked = st.sidebar.button("⚡ Process URL")

# File path for FAISS store
file_path = "faiss_store_openai.pkl"

# Model Initialization
llm = OpenAI(api_key=openai_key, temperature=0.9, max_tokens=500)
embeddings = OpenAIEmbeddings(openai_api_key=openai_key)

# Process URL
if process_url_clicked:
    with st.spinner("🔄 Processing the URL... Please wait."):
        try:
            # Load data
            loader = UnstructuredURLLoader(urls=urls)
            data = loader.load()

            # Split data
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            docs = text_splitter.split_documents(data)

            # Create embeddings and save to FAISS
            vectorstore_openai = FAISS.from_documents(docs, embeddings)

            # Save the FAISS index
            with open(file_path, "wb") as f:
                pickle.dump(vectorstore_openai, f)

            st.success("✅ Data processing completed successfully!")
        except Exception as e:
            st.error(f"❌ Error processing the URL: {str(e)}")

st.markdown('<p class="footer">Made with ❤️ by Vashishth</p>', unsafe_allow_html=True)