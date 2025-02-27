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
from textblob import TextBlob  # For tone detection (sentiment analysis)
import requests  # For related articles and truthfulness check

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

# Function to detect tone (sentiment analysis)
def detect_tone(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return "Positive"
    elif sentiment < 0:
        return "Negative"
    else:
        return "Neutral"

# Function to get related articles based on the URL
def get_related_articles(url):
    # Use an external API or scraping method here to fetch related articles
    # For demonstration purposes, let's assume we're using a simple API or search
    related_articles = []
    try:
        search_url = f"https://api.example.com/related?url={url}"  # Placeholder API endpoint
        response = requests.get(search_url)
        related_articles = response.json().get("articles", [])
    except Exception as e:
        print(f"Error fetching related articles: {e}")
    return related_articles

# Function to check article truthfulness (fact-checking API)
def check_truthfulness(text):
    # Placeholder: You can integrate APIs like FactCheck API for truthfulness check
    # Example logic here could involve checking for known fact database or trustworthiness scores
    try:
        fact_check_url = f"https://api.factcheck.org/check?text={text}"  # Placeholder API endpoint
        response = requests.get(fact_check_url)
        fact_data = response.json()
        return fact_data.get("truthfulness", "Unknown")
    except Exception as e:
        print(f"Error checking truthfulness: {e}")
        return "Unknown"

# Process URL
if process_url_clicked:
    with st.spinner("🔄 Processing the URL... Please wait."):
        try:
            # Load data
            loader = UnstructuredURLLoader(urls=[url])
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

            # Show related articles
            related_articles = get_related_articles(url)
            if related_articles:
                st.subheader("Suggested Articles:")
                for article in related_articles:
                    st.write(f"🔗 [ {article['title']} ]({article['url']})")

            # Show tone of the article
            tone = detect_tone(data[0].page_content)  # Assuming `data[0]` is the first article loaded
            st.subheader("Article Tone:")
            st.write(f"📝 Tone of the article: **{tone}**")

            # Check truthfulness of the article
            truthfulness = check_truthfulness(data[0].page_content)
            st.subheader("Article Truthfulness:")
            st.write(f"✅ Truthfulness: **{truthfulness}**")

        except Exception as e:
            st.error(f"❌ Error processing the URL: {str(e)}")

st.markdown('<p class="footer">Made with ❤️ by Vashishth</p>', unsafe_allow_html=True)
