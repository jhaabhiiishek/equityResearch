import os
import streamlit as st
import pickle
import time
import requests
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv
from langchain.llms.base import BaseLLM
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(find_dotenv(), override=True)
genai.configure(api_key=os.getenv("GENAI_API_KEY"))
# Load environment variables

class GeminiEmbeddings:
    def embed_documents(self, texts):
        result = genai.embed_content(
            model="models/text-embedding-004", content=texts, output_dimensionality=10
        )
        return result["embedding"]
    def embed_query(self, query):
        """Get embeddings for a single query."""
        return self.embed_documents([query])[0]  # Get embeddings and return the first one


st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_gemini.pkl"

main_placeholder = st.empty()

if process_url_clicked:
    data = []
    main_placeholder.text("Data Loading...Started...✅✅✅")
    for url in urls:
        response = requests.get(url)
        if response.status_code == 200:
            data.append(response.text)
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    # create embeddings and save it to FAISS index
    embeddings = GeminiEmbeddings()  # You may want to adjust this for Gemini embeddings if needed
    vectorstore_gemini = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_gemini, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(query)
            print(response.text)
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)
