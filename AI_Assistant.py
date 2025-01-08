# app.py
import streamlit as st
import os
import re
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
import importlib.util
import sys

# Load environment variables
load_dotenv()

def init_session_state():
    if 'email_verified' not in st.session_state:
        st.session_state.email_verified = False
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = 'home'

def get_database():
    try:
        mongo_uri = os.getenv("MONGODB_URI")
        if not mongo_uri:
            st.error("Missing MongoDB URI in environment variables")
            return None
        client = MongoClient(mongo_uri)
        client.server_info()
        return client.rag_app_db
    except Exception as e:
        st.error(f"Failed to connect to database: {str(e)}")
        return None

def validate_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None

def log_email(email):
    try:
        db = get_database()
        if db is not None:
            user_collection = db.users
            timestamp = datetime.now()
            result = user_collection.insert_one({
                "email": email,
                "timestamp": timestamp
            })
            return bool(result.inserted_id)
        return False
    except Exception as e:
        st.error(f"Failed to log email: {str(e)}")
        return False

def load_module_from_file(filepath):
    spec = importlib.util.spec_from_file_location("module", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def local_css():
    st.markdown("""
    <style>
        /* Modern clean background and colors */
        .stApp {
            background: #ffffff;
            color: #333333;
        }
        
        /* Card styling */
        .app-card {
            background: #ffffff;
            border-radius: 8px;
            padding: 15px;
            margin: 8px 0;
            border: 1px solid #e0e0e0;
            transition: all 0.2s ease;
        }
        
        .app-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
        }
        
        .app-card h3 {
            color: #333333;
            margin-bottom: 8px;
            font-size: 1.2em;
        }
        
        .app-card p {
            color: #666666;
            font-size: 0.95em;
            margin-bottom: 10px;
        }
        
        /* Button styling */
        .stButton button {
            background: #ffffff;
            color: #333333;
            border: 1px solid #d0d0d0;
            padding: 0.4rem 0.8rem;
            border-radius: 4px;
            font-weight: normal;
            font-size: 0.9em;
            transition: all 0.2s ease;
        }
        
        .stButton button:hover {
            background: #333333;
            color: #ffffff;
            border-color: #333333;
        }
        
        /* List styling */
        ul {
            list-style-type: none;
            padding-left: 0;
            margin-top: 5px;
        }
        
        ul li {
            margin: 5px 0;
            color: #666666;
            font-size: 0.9em;
        }
        
        /* Header styling */
        .main-header {
            padding: 30px 15px;
            margin-bottom: 30px;
            text-align: center;
        }
        
        /* Home button */
        .home-button {
            position: fixed;
            top: 10px;
            right: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

def render_home_button():
    if st.session_state.app_mode != 'home':
        if st.button("🏠 Back to Home", key='home_button'):
            st.session_state.app_mode = 'home'
            st.query_params.clear()
            st.rerun()

def main():
    # Initialize session state
    init_session_state()
    
    # Get current mode from query params
    current_mode = st.query_params.get('mode', 'home')
    
    if current_mode != st.session_state.app_mode:
        st.session_state.app_mode = current_mode
        st.rerun()

    # Render home button if not on home page
    render_home_button()

    # Handle different modes
    if st.session_state.app_mode == 'pdf':
        pdf_app = load_module_from_file("RAG_PDF_App.py")
        if hasattr(pdf_app, 'RAGApplication'):
            app = pdf_app.RAGApplication()
            # Remove page config from PDF app and call its main logic
            original_main = pdf_app.main
            def wrapped_main():
                st.markdown("# AI PDF Assistant")
                if hasattr(original_main, '__code__'):
                    original_main()
            wrapped_main()
    
    elif st.session_state.app_mode == 'excel':
        excel_app = load_module_from_file("RAG_XLCSV_APP.py")
        if hasattr(excel_app, 'main'):
            original_main = excel_app.main
            def wrapped_main():
                st.markdown("# AI Excel/CSV Assistant")
                if hasattr(original_main, '__code__'):
                    original_main()
            wrapped_main()
    
    else:  # Home page
        # Header
        st.markdown("""
            <div class="main-header">
                <h1 style='color: #333333; margin-bottom: 5px; font-size: 2.5em; text-align: center; font-weight: bold;'>AI Document Analysis Hub</h1>
                <p style='color: #666666; margin-top: 0; font-size: 1.25em; text-align: center; font-weight: normal;'>Built Using RAG and powered by OpenAI - GPT 4</p>
            </div>
        """, unsafe_allow_html=True)

        # Email verification section
        if not st.session_state.email_verified:
            st.markdown("""
                <p style='color: #333333; font-size: 1em;'>Please enter your email to continue</p>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                email = st.text_input("Email Address")
                if st.button("Submit Email"):
                    if validate_email(email):
                        if log_email(email):
                            st.session_state.email_verified = True
                            st.rerun()
                        else:
                            st.error("Failed to save email. Please try again.")
                    else:
                        st.error("Please enter a valid email address")
            
            with col2:
                if st.button("🔧 Testing Bypass"):
                    st.session_state.email_verified = True
                    st.rerun()
            return

        # Application cards
        if st.session_state.email_verified:
            st.markdown("""
                <div style='text-align: center; margin-bottom: 20px;'>
                    <h2 style='color: #333333; font-size: 1.2em;'>Choose Your AI Assistant</h2>
                    <p style='color: #666666; font-size: 0.9em;'>Select the tool that best fits your needs</p>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                    <div class='app-card'>
                        <h3>AI PDF Assistant</h3>
                        <p>Transform your PDF documents into interactive knowledge bases</p>
                        <ul>
                            <li>Ask questions about your documents</li>
                            <li>Extract key information</li>
                            <li>Powered by GPT-4</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
                if st.button("Launch PDF Assistant"):
                    st.session_state.app_mode = 'pdf'
                    st.query_params['mode'] = 'pdf'
                    st.rerun()

            with col2:
                st.markdown("""
                    <div class='app-card'>
                        <h3>AI Excel/CSV Assistant</h3>
                        <p>Analyze your data with natural language queries</p>
                        <ul>
                            <li>Data analysis made simple</li>
                            <li>Natural language queries</li>
                            <li>Instant insights</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
                if st.button("Launch Excel/CSV Assistant"):
                    st.session_state.app_mode = 'excel'
                    st.query_params['mode'] = 'excel'
                    st.rerun()

if __name__ == "__main__":
    # Set page config only once at the very beginning
    st.set_page_config(
        page_title="AI Document Analysis Hub",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    local_css()  # Apply custom CSS
    main()