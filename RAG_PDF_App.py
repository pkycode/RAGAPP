import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os
import uuid
import re
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables from .env file
load_dotenv()
# Get API key from environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY")

# MongoDB setup - Using environment variables
def get_database():
    try:
        # Get MongoDB connection string directly from .env file
        mongo_uri = os.getenv("MONGODB_URI")
        
        if not mongo_uri:
            st.error("Missing MongoDB URI in environment variables")
            return None
            
        client = MongoClient(mongo_uri)
        # Test the connection
        client.server_info()  # This will raise an exception if connection fails
        return client.rag_app_db  # database name
    except Exception as e:
        st.error(f"Failed to connect to database: {str(e)}")
        return None
        return client.rag_app_db  # database name
    except Exception as e:
        st.error(f"Failed to connect to database: {str(e)}")
        return None

def validate_email(email):
    """Validate email format using regex"""
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None

def log_email(email):
    """Log email with timestamp to MongoDB"""
    try:
        db = get_database()
        if db is not None:  # Proper way to check if database connection exists
            user_collection = db.users
            timestamp = datetime.now()
            result = user_collection.insert_one({
                "email": email,
                "timestamp": timestamp
            })
            # Check if insertion was successful
            if result.inserted_id:
                return True
        return False
    except Exception as e:
        st.error(f"Failed to log email: {str(e)}")
        return False

class RAGApplication:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        self.vector_store = None
        self.persist_directory = "vector_store"

    def process_pdf(self, pdf_file):
        """Process uploaded PDF file and create vector store"""
        try:
            # Create a unique temporary file name
            temp_pdf_path = f"temp_{uuid.uuid4()}.pdf"
            # Save uploaded file temporarily
            with open(temp_pdf_path, "wb") as f:
                f.write(pdf_file.getvalue())
            
            # Load and split the PDF
            loader = PyPDFLoader(temp_pdf_path)
            documents = loader.load()
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=100,
                length_function=len
            )
            texts = text_splitter.split_documents(documents)
            
            # Create vector store
            self.vector_store = FAISS.from_documents(
                documents=texts,
                embedding=self.embeddings
            )
            
            # Clean up temporary file
            os.remove(temp_pdf_path)
            return len(texts)
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return 0

    def get_answer(self, query):
        """Get answer for the query using RAG"""
        try:
            if not self.vector_store:
                return "Please upload a PDF document first."
            
            # Create retrieval chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_kwargs={"k": 3}
                )
            )
            
            # Get answer
            response = qa_chain.invoke({"query": query})
            return response["result"]
        except Exception as e:
            return f"Error generating answer: {str(e)}"

def main():
    # Page configuration
    st.set_page_config(page_title="RAG APP", layout="wide")

    # Company logo and header
    col1, col2 = st.columns([1, 3])
    with col1:
        # Add logo with click functionality
        logo_path = "images/bindspacelogo.jpg"
        if os.path.exists(logo_path):
            st.markdown(
                f'<a href="https://www.bindspacetech.com" target="_blank">'
                f'<img src="images/bindspacelogo.jpg" width="150">'
                f'</a>',
                unsafe_allow_html=True
            )

    with col2:
        st.markdown("# PDF Question Answering System")
        st.markdown("##### Built Using RAG and powered by OpenAI - GPT 4")

    # Initialize session state for email verification
    if 'email_verified' not in st.session_state:
        st.session_state.email_verified = False

    # Email verification section
    if not st.session_state.email_verified:
        st.write("Please enter your email to continue:")
        email = st.text_input("Email Address")
        
        if st.button("Submit Email"):
            if validate_email(email):
                if log_email(email):
                    st.session_state.email_verified = True
                    st.success("Email verified successfully!")
                    st.rerun()
                else:
                    st.error("Failed to save email. Please try again.")
            else:
                st.error("Please enter a valid email address")
        return

    # Initialize RAG application in session state
    if 'rag_app' not in st.session_state:
        st.session_state.rag_app = RAGApplication()

    # Main application (only shown after email verification)
    st.write("Upload a PDF and ask questions about its content!")
    
    # File upload
    pdf_file = st.file_uploader("Upload your PDF", type=['pdf'])
    if pdf_file:
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                num_chunks = st.session_state.rag_app.process_pdf(pdf_file)
                if num_chunks > 0:
                    st.success(f"PDF processed successfully! Created {num_chunks} text chunks.")

    # Query input
    query = st.text_input("Ask a question about your PDF:")
    if query:
        with st.spinner("Getting answer..."):
            answer = st.session_state.rag_app.get_answer(query)
            st.write("Answer:", answer)

if __name__ == "__main__":
    main()
