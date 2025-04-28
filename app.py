import os
import streamlit as st
import pickle
import time
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

st.title("Inquira: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()

# Setup LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  
    temperature=0,         
    max_tokens=None,           
    api_key=os.getenv("GOOGLE_GEMINI_API_KEY")  
)

# Initialize embeddings globally
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv('GOOGLE_GEMINI_API_KEY')
)

if process_url_clicked:
    # Load Data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading... Started ✅")
    data = loader.load()
    
    # Split Data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n','\n','.'],
        chunk_size=500,  # 🔥 reduced to capture finer details
        chunk_overlap=50
    )
    main_placeholder.text("Text Splitting... Started ✅")
    docs = text_splitter.split_documents(data)

    # Create Embeddings and Vectorstore
    vectorstore_gemini = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Building Embeddings... Started ✅")
    time.sleep(1)
    
    vectorstore_gemini.save_local("faiss_store_gemini") 

# Take user query
query = main_placeholder.text_input("Question: ")

if query:
    if os.path.exists("faiss_store_gemini"):
        # Load VectorStore
        x = FAISS.load_local("faiss_store_gemini", embeddings, allow_dangerous_deserialization=True)
        
        # 🔥 Use a stronger retriever (retrieve more documents)
        retriever = x.as_retriever(search_kwargs={"k": 10})  # retrieve top 10 chunks
        
        # Setup the QA Chain
        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=retriever
        )

        result = chain({"question": query}, return_only_outputs=True)

        # Display Answer
        st.header("Answer")
        st.write(result["answer"])

        # Display Sources
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            for source in sources.split("\n"):
                st.write(source)
