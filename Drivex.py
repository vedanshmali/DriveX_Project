import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from PIL import Image
import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Configure Tesseract (if not in system PATH, set its location here)
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Modify path if needed

# Functions for Data Extraction
def extract_pdf_text(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_excel_data(file):
    try:
        df = pd.read_excel(file)
        return df
    except Exception as e:
        return str(e)

def extract_image_text(file):
    try:
        image = Image.open(file)
        text = pytesseract.image_to_string(image)  # Extract text using Tesseract
        return text
    except Exception as e:
        return f"Error extracting text from image: {str(e)}"

# Process Uploaded Files
def process_files(files):
    all_text = ""
    for file in files:
        if file.type == "application/pdf":
            all_text += extract_pdf_text(file)
        elif file.type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
            df = extract_excel_data(file)
            all_text += df.to_string() if isinstance(df, pd.DataFrame) else str(df)
        elif file.type.startswith("image/"):
            all_text += extract_image_text(file)
    return all_text

# Chunk Text
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

# Create Vector Store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# DocuVision QA Chain
def get_qa_chain():
    prompt_template = """
    Use the context to answer the question accurately. If not in the context, reply "Not available in the provided context".
    Context:
    {context}
    Question: {question}
    Answer:
    """
    # Model renamed to DocuVision
    model = ChatGoogleGenerativeAI(model="DocuVision", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def generate_answer(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)
    chain = get_qa_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# Main Function
def main():
    st.set_page_config(page_title="DocuVision: Intelligent Document Processing", layout="wide")
    
    # Dark/Light Mode Toggle
    dark_mode = st.checkbox("Enable Dark Mode")
    if dark_mode:
        st.markdown(
            """
            <style>
                body { background-color: #121212; color: white; }
                .stButton > button { background-color: #333; color: white; border: 1px solid white; }
                .stTextInput > div { background-color: #333; color: white; }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
                body { background-color: white; color: black; }
                .stButton > button { background-color: #4caf50; color: white; border: none; }
                .stTextInput > div { background-color: white; color: black; }
            </style>
            """,
            unsafe_allow_html=True,
        )

    # Title
    st.title("DocuVision: Intelligent Document Processing")

    # File Upload
    st.subheader("Upload Files")
    files = st.file_uploader(
        "Upload PDF, Excel, or Images", accept_multiple_files=True, type=["pdf", "xlsx", "xls", "png", "jpg", "jpeg"]
    )

    # Process Files
    if st.button("Process Files"):
        if files:
            with st.spinner("Processing files..."):
                raw_text = process_files(files)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Files processed successfully!")
        else:
            st.warning("Please upload files before processing.")

    # Auto-Completion & Question Input
    st.subheader("Ask a Question")
    question_col, suggestions_col = st.columns([2, 1])

    with question_col:
        user_question = st.text_input("Type your question here:")
    with suggestions_col:
        st.markdown("### Suggestions:")
        suggestions = ["What is the summary?", "Can you provide insights?", "What are the key points?"]
        for suggestion in suggestions:
            if st.button(suggestion):
                user_question = suggestion

    # Generate Answer
    if st.button("Get Answer"):
        if user_question:
            with st.spinner("Generating answer..."):
                answer = generate_answer(user_question)
                st.subheader("Answer:")
                st.success(answer)
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
