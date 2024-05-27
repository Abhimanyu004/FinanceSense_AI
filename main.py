import os
import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=os.getenv('google_api_key'))
embeddings = GooglePalmEmbeddings()

st.title("Fin-News Research Tool")
st.sidebar.title("News Article URLs")
main_placeholder = st.empty()

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)
    
url_clicked = st.sidebar.button("Process URLs")


if url_clicked:
    #loading the data from the urls
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading Started . . .")
    data = loader.load()
    
    #splitting the data and saving in docs
    main_placeholder.text("Splitting Document Started . . .")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n','.',','],
        chunk_size = 1000
    )
    docs = text_splitter.split_documents(data)
    
    #create embedding and save it in faiss
    main_placeholder.text("Embedding Started . . .")
    vectorstore_genai = FAISS.from_documents(docs, embeddings) 
    
    #saving the faiss index to pickle file
    main_placeholder.text("Saving to Pikle File Started . . .")
    vectorstore_genai.save_local('faiss_index')
    
query = main_placeholder.text_input("Question: ")

if query:
    new_faiss_index = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=new_faiss_index.as_retriever(search_kwargs={"k": 10}),
    )
    result = qa.invoke(query)
    st.header('Answer')
    st.subheader(result['result'])

    

        
    