import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from docx import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the Google Generative AI API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to extract text from .txt files
def get_txt_text(txt_docs):
    text = ""
    for txt in txt_docs:
        text += txt.read().decode("utf-8")  # Reading and decoding the .txt file
    return text

# Function to extract text from .docx files
def get_docx_text(docx_docs):
    text = ""
    for docx in docx_docs:
        doc = Document(docx)
        for para in doc.paragraphs:
            text += para.text + "\n"
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=1000)  # Adjust chunk size if necessary
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save a FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")  # Save the FAISS index
    return vector_store

# Function to create a conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details. 
    If the answer is not in the provided context, then just say "answer is not available in the context". 
    Do not provide the wrong answer.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# Function to handle user input and query the vector store
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Load the FAISS vector store with the safe deserialization flag
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return

    # Perform a similarity search
    docs = new_db.similarity_search(user_question, k=3)  # Get top 3 results for better context

    if not docs:
        st.write("No relevant documents found.")
        return

    # Get the conversational chain
    chain = get_conversational_chain()

    # Get the response
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True,
    )

    # Display the response
    st.write("Reply: ", response["output_text"])

# Main application
def main():
    st.set_page_config(page_title="DoQueryLm")  # Set Streamlit page configuration
    st.header("Upload Documents and Work Through Them with Simple Queries")

    # User input for queries
    user_question = st.text_input("Ask any question from your uploaded files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        uploaded_files = st.file_uploader(
            "Upload your PDF, DOCX, or TXT files and click on the Submit & Process button",
            accept_multiple_files=True,
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = ""
                pdf_files = [file for file in uploaded_files if file.name.endswith(".pdf")]
                txt_files = [file for file in uploaded_files if file.name.endswith(".txt")]
                docx_files = [file for file in uploaded_files if file.name.endswith(".docx")]

                # Process PDF files
                if pdf_files:
                    raw_text += get_pdf_text(pdf_files)
                
                # Process TXT files
                if txt_files:
                    raw_text += get_txt_text(txt_files)
                
                # Process DOCX files
                if docx_files:
                    raw_text += get_docx_text(docx_files)

                # Split the combined raw text into chunks
                text_chunks = get_text_chunks(raw_text)

                # Create and save the FAISS vector store
                get_vector_store(text_chunks)
                st.success("Processing complete and vector store created.")

if __name__ == "__main__":
    main()
