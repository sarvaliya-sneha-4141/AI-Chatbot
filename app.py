import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import tempfile
import time

# Load environment variables from .env file
load_dotenv()

# Configure Streamlit page settings
st.set_page_config(
    page_title="Neura Talk : ChatBot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a Grok-like UI with updated text color for input fields
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main container */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }

    /* Header styling */
    .header-container {
        text-align: center;
        margin-bottom: 3rem;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.3);
    }

    .subtitle {
        font-size: 1.2rem;
        color: #a0a0a0;
        font-weight: 400;
        margin-bottom: 0;
    }

    /* Card styling */
    .custom-card {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }

    .custom-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        border-color: rgba(102, 126, 234, 0.3);
    }

    /* Enhanced Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-radius: 15px;
        padding: 1rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(102, 126, 234, 0.2);
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }

    .stTabs [data-baseweb="tab"] {
        height: 70px;
        padding: 0 35px;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        color: rgba(255, 255, 255, 0.9);
        font-weight: 700;
        font-size: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.15);
        transition: all 0.3s ease;
        min-width: 250px;
        display: flex;
        align-items: center;
        justify-content: center;
        letter-spacing: 0.5px;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.15);
        border-color: rgba(102, 126, 234, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.2);
        color: #ffffff;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border: 1px solid #667eea !important;
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.4) !important;
        transform: translateY(-2px) !important;
    }

    /* Input styling for question fields */
    .stTextInput > div > div > input {
        background: #ffffff !important; /* White background for visibility */
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px !important;
        color: #000000 !important; /* Black text for readability */
        font-size: 16px !important;
        padding: 12px 16px !important;
        transition: all 0.3s ease !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
        background: #ffffff !important; /* Maintain white background on focus */
        color: #000000 !important; /* Ensure text remains black on focus */
    }

    /* Placeholder text styling */
    .stTextInput > div > div > input::placeholder {
        color: rgba(0, 0, 0, 0.6) !important; /* Darker placeholder text */
        opacity: 1 !important;
    }

    /* Input label styling */
    .stTextInput > label {
        color: white !important;
        font-weight: 500 !important;
    }

    /* Text input container */
    .stTextInput > div {
        color: white !important;
    }

    /* Additional input field styling */
    input[type="text"] {
        color: white !important;
        background: rgba(255, 255, 255, 0.1) !important;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }

    /* File uploader styling */
    .stFileUploader > div {
        background: rgba(255, 255, 255, 0.05);
        border: 2px dashed rgba(255, 255, 255, 0.2);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
        color: #ffffff !important;
    }

    .stFileUploader > div:hover {
        border-color: #667eea;
        background: rgba(102, 126, 234, 0.1);
    }

    /* File uploader text styling */
    .stFileUploader label {
        color: #ffffff !important;
        font-weight: 500 !important;
    }

    .stFileUploader div, .stFileUploader p, .stFileUploader span {
        color: #ffffff !important;
    }

    /* File uploader help text */
    .stFileUploader small {
        color: rgba(255, 255, 255, 0.7) !important;
    }

    /* Drag and drop text */
    .stFileUploader [data-testid="stFileUploaderDropzone"] {
        color: #ffffff !important;
    }

    .stFileUploader [data-testid="stFileUploaderDropzone"] * {
        color: #ffffff !important;
    }

    /* Chat message styling */
    .chat-message {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        backdrop-filter: blur(10px);
    }

    .chat-question {
        color: #ffffff !important;
        font-weight: 600 !important;
        margin-bottom: 0.8rem !important;
        font-size: 1.1rem !important;
        background: rgba(102, 126, 234, 0.3) !important;
        padding: 1rem !important;
        border-radius: 10px !important;
        border-left: 4px solid #667eea !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.5) !important;
        display: block !important;
    }

    /* More specific selectors for chat questions */
    div.chat-question, .chat-message .chat-question {
        color: #ffffff !important;
        font-weight: 600 !important;
        background: rgba(102, 126, 234, 0.3) !important;
        padding: 1rem !important;
        border-radius: 10px !important;
        border-left: 4px solid #667eea !important;
        margin-bottom: 0.8rem !important;
        font-size: 1.1rem !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.5) !important;
        display: block !important;
    }

    /* Ensure question text is always visible */
    .chat-question *, .chat-message .chat-question * {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* Spinner text styling */
    .stSpinner > div > div {
        color: #ffffff !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
    }

    /* Spinner container styling */
    div[data-testid="stSpinner"] {
        color: #ffffff !important;
    }

    /* Spinner text specific styling */
    div[data-testid="stSpinner"] > div {
        color: #ffffff !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
    }

    /* Alternative spinner selectors */
    .stSpinner div, .stSpinner span, .stSpinner p {
        color: #ffffff !important;
        font-weight: 500 !important;
    }

    /* More comprehensive spinner styling */
    [data-testid="stSpinner"] * {
        color: #ffffff !important;
    }

    /* Streamlit spinner text */
    .stSpinner {
        color: #ffffff !important;
    }

    /* Spinner loading text */
    div[data-testid="stSpinner"] div[data-testid="stMarkdownContainer"] {
        color: #ffffff !important;
    }

    /* Ensure all spinner related text is visible */
    .stSpinner *, [data-testid="stSpinner"] *,
    div[class*="spinner"] *, div[class*="loading"] * {
        color: #ffffff !important;
        font-weight: 500 !important;
    }

    /* Global spinner text override */
    .stApp [data-testid="stSpinner"] {
        color: #ffffff !important;
    }

    .stApp [data-testid="stSpinner"] > div {
        color: #ffffff !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
    }

    /* Spinner markdown content */
    [data-testid="stSpinner"] [data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
    }

    .chat-answer {
        color: #ffffff !important;
        line-height: 1.7 !important;
        font-size: 1rem !important;
        font-weight: 400 !important;
        margin-top: 0.8rem !important;
        padding: 1rem !important;
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 10px !important;
        border-left: 4px solid #4CAF50 !important;
    }

    /* Summary content styling */
    .summary-content {
        color: #ffffff;
        line-height: 1.6;
    }

    .summary-content h2, .summary-content h3, .summary-content h4 {
        color: #667eea;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }

    .summary-content strong {
        color: #e0e0e0;
        font-weight: 600;
    }

    /* Success/Info/Warning/Error styling */
    .stSuccess, .stInfo, .stWarning, .stError {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        backdrop-filter: blur(10px);
        color: #ffffff !important;
    }

    .stSuccess div, .stInfo div, .stWarning div, .stError div {
        color: #ffffff !important;
    }

    /* Error message specific styling */
    .stError {
        background: rgba(244, 67, 54, 0.15) !important;
        border: 1px solid rgba(244, 67, 54, 0.3) !important;
        color: #ffffff !important;
    }

    .stError div, .stError p, .stError span {
        color: #ffffff !important;
        font-weight: 500 !important;
    }

    /* Warning message specific styling */
    .stWarning {
        background: rgba(255, 152, 0, 0.15) !important;
        border: 1px solid rgba(255, 152, 0, 0.3) !important;
        color: #ffffff !important;
    }

    .stWarning div, .stWarning p, .stWarning span {
        color: #ffffff !important;
        font-weight: 500 !important;
    }

    /* Success message specific styling */
    .stSuccess {
        background: rgba(76, 175, 80, 0.15) !important;
        border: 1px solid rgba(76, 175, 80, 0.3) !important;
        color: #ffffff !important;
    }

    .stSuccess div, .stSuccess p, .stSuccess span {
        color: #ffffff !important;
        font-weight: 500 !important;
    }

    /* Info message specific styling */
    .stInfo {
        background: rgba(33, 150, 243, 0.15) !important;
        border: 1px solid rgba(33, 150, 243, 0.3) !important;
        color: #ffffff !important;
    }

    .stInfo div, .stInfo p, .stInfo span {
        color: #ffffff !important;
        font-weight: 500 !important;
    }

    /* Alert content styling */
    [data-testid="stAlert"] {
        color: #ffffff !important;
    }

    [data-testid="stAlert"] div {
        color: #ffffff !important;
    }

    /* More specific alert styling */
    div[data-testid="stAlert"] * {
        color: #ffffff !important;
        font-weight: 500 !important;
    }

    /* General text styling */
    .stMarkdown, .stText {
        color: #ffffff !important;
    }

    /* Markdown content styling */
    .stMarkdown p, .stMarkdown div, .stMarkdown span {
        color: #ffffff !important;
    }

    /* Ensure all text is visible */
    p, div, span {
        color: inherit;
    }

    /* Form elements text color */
    .stForm {
        color: white !important;
    }

    /* All text elements */
    * {
        color: inherit;
    }

    /* Streamlit specific text elements */
    .stTextInput label, .stTextArea label, .stSelectbox label, .stFileUploader label {
        color: white !important;
        font-weight: 500 !important;
    }

    /* Help text styling */
    .help {
        color: rgba(255, 255, 255, 0.7) !important;
    }

    /* File uploader specific text elements */
    [data-testid="stFileUploader"] {
        color: #ffffff !important;
    }

    [data-testid="stFileUploader"] * {
        color: #ffffff !important;
    }

    [data-testid="stFileUploader"] label {
        color: #ffffff !important;
        font-weight: 500 !important;
    }

    [data-testid="stFileUploader"] small {
        color: rgba(255, 255, 255, 0.7) !important;
    }

    /* Drag and drop zone styling */
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 2px dashed rgba(255, 255, 255, 0.3) !important;
        border-radius: 16px !important;
        color: #ffffff !important;
    }

    [data-testid="stFileUploaderDropzone"] * {
        color: #ffffff !important;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #667eea !important;
        background: rgba(102, 126, 234, 0.1) !important;
    }

    /* Exception and error text styling */
    .stException, .stException div, .stException p {
        color: #ffffff !important;
        background: rgba(244, 67, 54, 0.15) !important;
        border: 1px solid rgba(244, 67, 54, 0.3) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }

    /* Toast messages */
    .stToast {
        color: #ffffff !important;
    }

    .stToast div {
        color: #ffffff !important;
    }

    /* Status messages */
    [data-testid="stStatusWidget"] {
        color: #ffffff !important;
    }

    [data-testid="stStatusWidget"] * {
        color: #ffffff !important;
    }

    /* Question input specific styling */
    .question-input {
        color: white !important;
        background: rgba(255, 255, 255, 0.1) !important;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        color: white;
    }

    /* Spinner styling */
    .stSpinner > div {
        border-top-color: #667eea;
    }

    /* Divider styling */
    .stDivider {
        margin: 2rem 0;
    }

    /* Custom animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .fade-in-up {
        animation: fadeInUp 0.6s ease-out;
    }

    /* Comprehensive text visibility fixes */
    .stApp, .stApp * {
        color: inherit !important;
    }

    /* Force visibility for all possible text containers */
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] div,
    div[data-testid="stMarkdownContainer"] span,
    div[data-testid="stText"] p,
    div[data-testid="stText"] div,
    div[data-testid="stText"] span {
        color: #ffffff !important;
        font-weight: 500 !important;
    }

    /* Error and exception containers */
    .element-container div,
    .stAlert div,
    [data-testid="stAlert"] div,
    [data-testid="stException"] div,
    [data-testid="stError"] div {
        color: #ffffff !important;
    }

    /* All possible text elements */
    .stApp p, .stApp div, .stApp span, .stApp li, .stApp td, .stApp th {
        color: inherit !important;
    }

    /* Ensure error messages are visible */
    .stApp [data-testid="stAlert"] p,
    .stApp [data-testid="stAlert"] div,
    .stApp [data-testid="stAlert"] span {
        color: #ffffff !important;
        font-weight: 500 !important;
    }

    /* Universal text visibility - catch all approach */
    .stApp label,
    .stApp small,
    .stApp .help,
    .stApp [class*="help"],
    .stApp [data-testid] label,
    .stApp [data-testid] small,
    .stApp [data-testid] p,
    .stApp [data-testid] div,
    .stApp [data-testid] span {
        color: #ffffff !important;
    }

    /* File uploader comprehensive styling */
    .stFileUploader,
    .stFileUploader *,
    [data-testid="stFileUploader"],
    [data-testid="stFileUploader"] *,
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploaderDropzone"] * {
        color: #ffffff !important;
    }

    /* File uploader button - Browse files button styling */
    .stFileUploader button,
    [data-testid="stFileUploader"] button,
    .stFileUploader [role="button"],
    [data-testid="stFileUploader"] [role="button"],
    [data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3) !important;
    }

    .stFileUploader button:hover,
    [data-testid="stFileUploader"] button:hover,
    .stFileUploader [role="button"]:hover,
    [data-testid="stFileUploader"] [role="button"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
        color: #ffffff !important;
    }

    /* Ensure all form elements are visible */
    .stForm label,
    .stForm small,
    .stForm p,
    .stForm div,
    .stForm span {
        color: #ffffff !important;
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'pdf_summary' not in st.session_state:
    st.session_state.pdf_summary = ""
if 'document_chat_history' not in st.session_state:
    st.session_state.document_chat_history = []
if 'document_direct_chat_history' not in st.session_state:
    st.session_state.document_direct_chat_history = []
if 'general_chat_history' not in st.session_state:
    st.session_state.general_chat_history = []

def get_llm():
    """
    Retrieve the Google Gemini LLM instance using the API key from environment variables.

    Returns:
        ChatGoogleGenerativeAI: Initialized LLM instance or None if API key is missing.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Please set GOOGLE_API_KEY in your .env file")
        return None
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key
    )

def process_pdfs(pdf_files):
    """
    Process multiple PDF files and create a combined vector store for document analysis.

    Args:
        pdf_files (list): List of uploaded PDF files.

    Returns:
        tuple: (vectorstore, chunks) or (None, None) if processing fails.
    """
    try:
        all_chunks = []
        
        for i, pdf_file in enumerate(pdf_files):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Load PDF content
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            # Split documents into manageable chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)
            
            # Add metadata to chunks
            for chunk in chunks:
                chunk.metadata['source'] = pdf_file.name
                chunk.metadata['file_index'] = i
            
            all_chunks.extend(chunks)
        
        if not all_chunks:
            st.error("No content extracted from PDFs")
            return None, None
        
        # Create embeddings and vector store
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = FAISS.from_documents(all_chunks, embeddings)
        
        return vectorstore, all_chunks
        
    except Exception as e:
        st.error(f"Error processing PDFs: {str(e)}")
        return None, None

def generate_summary(chunks, llm):
    """
    Generate a concise summary of the content from all processed PDF chunks.

    Args:
        chunks (list): List of document chunks.
        llm (ChatGoogleGenerativeAI): Initialized language model.

    Returns:
        str: Formatted summary of the documents.
    """
    try:
        # Group chunks by source file
        files_content = {}
        for chunk in chunks:
            source = chunk.metadata.get('source', 'Unknown')
            if source not in files_content:
                files_content[source] = []
            files_content[source].append(chunk.page_content)
        
        # Create brief summaries for each file
        summaries = []
        for filename, content_list in files_content.items():
            file_text = "\n".join(content_list)
            if len(file_text) > 3000:
                file_text = file_text[:3000]
            
            file_summary_prompt = PromptTemplate(
                input_variables=["filename", "text"],
                template="""Provide a very brief summary (2-3 sentences) of the main content from {filename}.
                
                Content: {text}
                
                Brief Summary:"""
            )
            
            file_summary_chain = file_summary_prompt | llm
            file_summary = file_summary_chain.invoke({"filename": filename, "text": file_text})
            summaries.append(f"**{filename}**: {file_summary.content if hasattr(file_summary, 'content') else str(file_summary)}")
        
        # Generate overall summary
        overall_prompt = PromptTemplate(
            input_variables=["summaries"],
            template="""Based on these document summaries, provide a very brief overview (3-4 sentences) highlighting the main themes.
            
            Document Summaries:
            {summaries}
            
            Brief Overview:"""
        )
        
        overall_chain = overall_prompt | llm
        overall_summary = overall_chain.invoke({"summaries": "\n\n".join(summaries)})
        
        final_summary = "## Document Summaries\n\n" + "\n\n".join(summaries)
        final_summary += "\n\n## Key Themes\n\n" + (overall_summary.content if hasattr(overall_summary, 'content') else str(overall_summary))
        
        return final_summary
        
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return "Error generating summary."

def answer_question(question, vectorstore, llm):
    """
    Answer questions based on the content of processed PDFs using vector search.

    Args:
        question (str): User’s question.
        vectorstore (FAISS): Vector store containing PDF content.
        llm (ChatGoogleGenerativeAI): Initialized language model.

    Returns:
        tuple: (answer_text, source_files) or error message.
    """
    try:
        # Search for relevant documents
        docs = vectorstore.similarity_search(question, k=5)

        if not docs:
            return "I couldn't find relevant information in the PDFs to answer this question.", []

        # Create context from relevant documents
        context = "\n".join([doc.page_content for doc in docs])

        # Get source files for relevant documents
        source_files = list(set([doc.metadata.get('source', 'Unknown') for doc in docs]))

        # Define QA prompt
        qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""Based on the following context from multiple PDF documents, please answer the question.
            If the question cannot be answered from the provided context, respond with "This question is out of the PDF context."

            Context: {context}

            Question: {question}

            Answer:"""
        )

        # Process QA chain
        qa_chain = qa_prompt | llm
        answer = qa_chain.invoke({"context": context, "question": question})

        answer_text = answer.content if hasattr(answer, 'content') else str(answer)
        return answer_text, source_files

    except Exception as e:
        st.error(f"Error answering question: {str(e)}")
        return "Error processing your question.", []

def answer_direct_question(question, llm):
    """
    Answer direct questions without relying on PDF context.

    Args:
        question (str): User’s question.
        llm (ChatGoogleGenerativeAI): Initialized language model.

    Returns:
        str: Answer to the question or error message.
    """
    try:
        # Define direct QA prompt
        direct_qa_prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are a helpful AI assistant. Please answer the following question to the best of your knowledge.
            Be informative, accurate, and concise in your response.

            Question: {question}

            Answer:"""
        )

        # Process direct QA chain
        direct_qa_chain = direct_qa_prompt | llm
        answer = direct_qa_chain.invoke({"question": question})

        answer_text = answer.content if hasattr(answer, 'content') else str(answer)
        return answer_text

    except Exception as e:
        st.error(f"Error answering direct question: {str(e)}")
        return "Error processing your question."

# Main application function
def main():
    """
    Main function to run the AI Assistant Streamlit app with document upload and chat features.
    """
    # Header section with modern styling
    st.markdown("""
    <div class="header-container fade-in-up">
        <h1 class="main-title">🤖 AI Assistant</h1>
        <p class="subtitle">Powered by Neura Talk : ChatBot • Upload PDFs or ask direct questions</p>
    </div>
    """, unsafe_allow_html=True)

    # Create tabs immediately after header
    st.markdown("""
    <div style="margin: 2rem 0;">
        <p style="text-align: center; color: rgba(255, 255, 255, 0.8); margin-bottom: 2rem; font-size: 1.1rem;">Choose how you want to interact with the AI assistant</p>
    </div>
    """, unsafe_allow_html=True)

    # Create tabs with larger, more prominent labels
    tab1, tab2 = st.tabs(["📄  Document Q&A", "💬  General Chat"])

    with tab1:
        # Document Upload Section
        st.markdown("""
        <div style="margin-top: 2rem;">
            <h2 style="text-align: center; color: #ffffff; margin-bottom: 2rem; font-weight: 600; font-size: 1.8rem;">📄 Document Upload & Analysis</h2>
        </div>
        """, unsafe_allow_html=True)

        # Create columns for better layout
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            # File upload section with custom styling
            st.markdown("""
            <div class="custom-card fade-in-up">
                <h3 style="color: #667eea; margin-bottom: 1rem; font-weight: 600;">📄 Document Upload</h3>
            </div>
            """, unsafe_allow_html=True)

            # Add specific CSS for file uploader visibility
            st.markdown("""
            <style>
            /* Force file uploader text visibility */
            .stFileUploader label,
            .stFileUploader small,
            .stFileUploader p,
            .stFileUploader div,
            .stFileUploader span,
            [data-testid="stFileUploader"] label,
            [data-testid="stFileUploader"] small,
            [data-testid="stFileUploader"] p,
            [data-testid="stFileUploader"] div,
            [data-testid="stFileUploader"] span,
            [data-testid="stFileUploaderDropzone"] *,
            .stFileUploader .help,
            [data-testid="stFileUploader"] .help {
                color: #ffffff !important;
                font-weight: 500 !important;
            }

            /* File uploader button styling - Browse files button */
            .stFileUploader button,
            [data-testid="stFileUploader"] button,
            .stFileUploader input[type="file"] + label,
            [data-testid="stFileUploaderDropzone"] button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                color: #ffffff !important;
                border: none !important;
                border-radius: 8px !important;
                padding: 8px 16px !important;
                font-weight: 600 !important;
                font-size: 14px !important;
                transition: all 0.3s ease !important;
                box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3) !important;
            }

            .stFileUploader button:hover,
            [data-testid="stFileUploader"] button:hover,
            [data-testid="stFileUploaderDropzone"] button:hover {
                transform: translateY(-1px) !important;
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
                color: #ffffff !important;
            }

            /* Alternative selectors for browse button */
            .stFileUploader [role="button"],
            [data-testid="stFileUploader"] [role="button"] {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                color: #ffffff !important;
                border: none !important;
                border-radius: 8px !important;
                padding: 8px 16px !important;
                font-weight: 600 !important;
            }
            </style>
            """, unsafe_allow_html=True)

            uploaded_files = st.file_uploader(
                "Choose PDF files to analyze",
                type=['pdf'],
                accept_multiple_files=True,
                help="Upload one or multiple PDF files to analyze together",
                label_visibility="collapsed"
            )

            if uploaded_files:
                # Display uploaded files with modern styling
                st.markdown("""
                <div class="custom-card" style="margin-top: 1rem;">
                    <h4 style="color: #667eea; margin-bottom: 1rem;">📁 Selected Files</h4>
                </div>
                """, unsafe_allow_html=True)

                for i, file in enumerate(uploaded_files):
                    st.markdown(f"""
                    <div style="background: rgba(255, 255, 255, 0.1); padding: 0.8rem; margin: 0.5rem 0; border-radius: 8px; border-left: 3px solid #667eea;">
                        <strong style="color: #ffffff; font-size: 1rem;">{i+1}. {file.name}</strong>
                        <span style="color: #b0b0b0; margin-left: 1rem; font-size: 0.9rem;">({file.size / 1024:.1f} KB)</span>
                    </div>
                    """, unsafe_allow_html=True)

            # Process button with enhanced styling
            if st.button("🚀 Process Documents", type="primary", use_container_width=True):
                # Add custom styling for spinner visibility
                st.markdown("""
                <style>
                div[data-testid="stSpinner"] * {
                    color: #ffffff !important;
                    font-weight: 500 !important;
                    font-size: 1rem !important;
                }
                </style>
                """, unsafe_allow_html=True)

                with st.spinner("🔄 Processing your documents..."):
                    # Get LLM instance
                    llm = get_llm()
                    if llm is None:
                        return

                    # Process PDFs
                    vectorstore, chunks = process_pdfs(uploaded_files)

                    if vectorstore and chunks:
                        st.session_state.vectorstore = vectorstore
                        st.session_state.pdf_processed = True

                        # Generate summary
                        st.markdown("""
                        <style>
                        div[data-testid="stSpinner"] * {
                            color: #ffffff !important;
                            font-weight: 500 !important;
                            font-size: 1rem !important;
                        }
                        </style>
                        """, unsafe_allow_html=True)

                        with st.spinner("✨ Generating intelligent summary..."):
                            summary = generate_summary(chunks, llm)
                            st.session_state.pdf_summary = summary

                        st.success("🎉 Documents processed successfully!")
                        time.sleep(0.5)  # Small delay for better UX
                        st.rerun()

    # Display summary if available with modern styling
    if st.session_state.pdf_summary:
        st.markdown("""
        <div class="custom-card fade-in-up" style="margin-top: 2rem;">
            <h3 style="color: #667eea; margin-bottom: 1rem; font-weight: 600;">📊 Document Summary</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="chat-message summary-content">
            {st.session_state.pdf_summary}
        </div>
        """, unsafe_allow_html=True)

        # PDF Question Section - only show if documents are processed
        if st.session_state.pdf_processed:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
                padding: 2rem;
                border-radius: 15px;
                border: 1px solid rgba(102, 126, 234, 0.3);
                margin-top: 3rem;
            ">
                <h3 style="color: #ffffff; margin-bottom: 1rem; font-weight: 600; font-size: 1.4rem; text-align: center;">❓ Ask Questions About Your Documents</h3>
                <p style="color: rgba(255, 255, 255, 0.8); text-align: center; margin: 0;">Get specific answers from your uploaded PDFs</p>
            </div>
            """, unsafe_allow_html=True)

            # PDF question input
            with st.container():
                st.markdown("""
                <div style="
                    background: rgba(255, 255, 255, 0.05);
                    padding: 1.5rem;
                    border-radius: 12px;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    margin: 1rem 0;
                ">
                </div>
                """, unsafe_allow_html=True)

                pdf_question = st.text_input(
                    "Ask a question about your documents:",
                    placeholder="What are the main topics discussed? Summarize the key findings...",
                    key="pdf_question",
                    label_visibility="collapsed"
                )

            # Always show the button
            st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)
            if st.button("🔍 Analyze Documents", type="primary", use_container_width=True, key="pdf_answer_btn"):
                if not pdf_question:
                    st.warning("⚠️ Please enter a question first!")
                else:
                    # Add custom styling for spinner visibility
                    st.markdown("""
                    <style>
                    div[data-testid="stSpinner"] * {
                        color: #ffffff !important;
                        font-weight: 500 !important;
                        font-size: 1rem !important;
                    }
                    </style>
                    """, unsafe_allow_html=True)

                    with st.spinner("🧠 Analyzing your documents..."):
                        llm = get_llm()
                        if llm:
                            answer, source_files = answer_question(
                                pdf_question,
                                st.session_state.vectorstore,
                                llm
                            )

                            st.markdown(f"""
                            <div class="chat-message">
                                <div class="chat-question" style="color: #ffffff !important; background: rgba(102, 126, 234, 0.3) !important; padding: 1rem !important; border-radius: 10px !important; border-left: 4px solid #667eea !important; font-weight: 600 !important; font-size: 1.1rem !important; margin-bottom: 0.8rem !important;">
                                    <strong style="color: #ffffff !important;">Q:</strong> <span style="color: #ffffff !important; font-weight: 600 !important;">{pdf_question}</span>
                                </div>
                                <div class="chat-answer">{answer}</div>
                            </div>
                            """, unsafe_allow_html=True)

                            if source_files:
                                st.markdown("**📚 Sources:**")
                                for source in source_files:
                                    st.markdown(f"""
                                    <div style="background: rgba(102, 126, 234, 0.1); padding: 0.5rem; margin: 0.3rem 0; border-radius: 8px; border-left: 3px solid #667eea;">
                                        📄 {source}
                                    </div>
                                    """, unsafe_allow_html=True)

                            # Add to document chat history (PDF-related questions)
                            st.session_state.document_chat_history.append((pdf_question, answer))

                            if "out of context" in answer.lower() or "cannot be answered" in answer.lower():
                                st.markdown("""
                                <div style="background: rgba(255, 152, 0, 0.1); border: 1px solid rgba(255, 152, 0, 0.3); border-radius: 12px; padding: 1rem; margin-top: 1rem;">
                                    <p style="color: #ff9800; margin: 0;">⚠️ This question appears to be outside the document context.</p>
                                </div>
                                """, unsafe_allow_html=True)

        # Display document chat history (PDF-related questions)
        if st.session_state.document_chat_history:
            st.markdown("""
            <div style="
                background: rgba(255, 255, 255, 0.05);
                padding: 1.5rem;
                border-radius: 12px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                margin: 2rem 0;
            ">
                <h4 style="color: #ffffff; margin-bottom: 1rem; font-weight: 600; text-align: center; font-size: 1.3rem;">📄 Recent Document Questions</h4>
                <p style="color: rgba(255, 255, 255, 0.7); text-align: center; margin: 0;">Your last 5 PDF-related questions</p>
            </div>
            """, unsafe_allow_html=True)

            for i, (q, a) in enumerate(reversed(st.session_state.document_chat_history[-5:])):
                with st.expander(f"📄 {q[:60]}..." if len(q) > 60 else f"📄 {q}", expanded=False):
                    st.markdown(f"""
                    <div class="chat-message">
                        <div class="chat-question" style="color: #ffffff !important; background: rgba(102, 126, 234, 0.3) !important; padding: 1rem !important; border-radius: 10px !important; border-left: 4px solid #667eea !important; font-weight: 600 !important; font-size: 1.1rem !important; margin-bottom: 0.8rem !important;">
                            <strong style="color: #ffffff !important;">Q:</strong> <span style="color: #ffffff !important; font-weight: 600 !important;">{q}</span>
                        </div>
                        <div class="chat-answer">{a}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # Direct question input in Document tab
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(76, 175, 80, 0.15) 0%, rgba(139, 195, 74, 0.15) 100%);
            padding: 2rem;
            border-radius: 15px;
            border: 1px solid rgba(76, 175, 80, 0.3);
            margin-top: 3rem;
        ">
            <h3 style="color: #ffffff; margin-bottom: 1rem; font-weight: 600; font-size: 1.4rem; text-align: center;">💬 Or Ask Any General Question</h3>
            <p style="color: rgba(255, 255, 255, 0.8); text-align: center; margin: 0;">Get answers to any question - no document required!</p>
        </div>
        """, unsafe_allow_html=True)

        with st.container():
            st.markdown("""
            <div style="
                background: rgba(255, 255, 255, 0.05);
                padding: 1.5rem;
                border-radius: 12px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                margin: 1rem 0;
            ">
            </div>
            """, unsafe_allow_html=True)

            direct_question_tab1 = st.text_input(
                "Ask any general question:",
                placeholder="What is artificial intelligence? How does machine learning work? Explain quantum computing...",
                key="direct_question_tab1",
                label_visibility="collapsed"
            )

        # Always show the button
        st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)
        if st.button("🚀 Get AI Response", type="primary", use_container_width=True, key="direct_answer_btn_tab1"):
            if not direct_question_tab1:
                st.warning("⚠️ Please enter a question first!")
            else:
                # Add custom styling for spinner visibility
                st.markdown("""
                <style>
                div[data-testid="stSpinner"] * {
                    color: #ffffff !important;
                    font-weight: 500 !important;
                    font-size: 1rem !important;
                }
                </style>
                """, unsafe_allow_html=True)

                with st.spinner("🤖 AI is thinking..."):
                    llm = get_llm()
                    if llm:
                        answer = answer_direct_question(direct_question_tab1, llm)

                        st.markdown(f"""
                        <div class="chat-message fade-in-up">
                            <div class="chat-question" style="color: #ffffff !important; background: rgba(76, 175, 80, 0.3) !important; padding: 1rem !important; border-radius: 10px !important; border-left: 4px solid #4CAF50 !important; font-weight: 600 !important; font-size: 1.1rem !important; margin-bottom: 0.8rem !important;">
                                <strong style="color: #ffffff !important;">Q:</strong> <span style="color: #ffffff !important; font-weight: 600 !important;">{direct_question_tab1}</span>
                            </div>
                            <div class="chat-answer">{answer}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Add to document direct chat history (not document_chat_history)
                        st.session_state.document_direct_chat_history.append((direct_question_tab1, answer))

        # Display document direct chat history (direct questions in Document tab)
        if st.session_state.document_direct_chat_history:
            st.markdown("""
            <div style="
                background: rgba(255, 255, 255, 0.05);
                padding: 1.5rem;
                border-radius: 12px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                margin: 2rem 0;
            ">
                <h4 style="color: #ffffff; margin-bottom: 1rem; font-weight: 600; text-align: center; font-size: 1.3rem;">💬 Recent General Questions (Document Tab)</h4>
                <p style="color: rgba(255, 255, 255, 0.7); text-align: center; margin: 0;">Your last 5 general questions in this tab</p>
            </div>
            """, unsafe_allow_html=True)

            for i, (q, a) in enumerate(reversed(st.session_state.document_direct_chat_history[-5:])):
                with st.expander(f"💬 {q[:60]}..." if len(q) > 60 else f"💬 {q}", expanded=False):
                    st.markdown(f"""
                    <div class="chat-message">
                        <div class="chat-question" style="color: #ffffff !important; background: rgba(76, 175, 80, 0.3) !important; padding: 1rem !important; border-radius: 10px !important; border-left: 4px solid #4CAF50 !important; font-weight: 600 !important; font-size: 1.1rem !important; margin-bottom: 0.8rem !important;">
                            <strong style="color: #ffffff !important;">Q:</strong> <span style="color: #ffffff !important; font-weight: 600 !important;">{q}</span>
                        </div>
                        <div class="chat-answer">{a}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # Clear document chat history button
        if st.session_state.document_chat_history or st.session_state.document_direct_chat_history:
            st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
            if st.button("🗑️ Clear Document Chat History", type="secondary", use_container_width=True, key="clear_doc_history"):
                # Clear both document chat histories
                st.session_state.document_chat_history = []
                st.session_state.document_direct_chat_history = []

                # Clear document tab input fields
                if 'pdf_question' in st.session_state:
                    del st.session_state['pdf_question']
                if 'direct_question_tab1' in st.session_state:
                    del st.session_state['direct_question_tab1']

                st.success("✨ Document chat history cleared!")
                time.sleep(0.5)
                st.rerun()

    with tab2:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(156, 39, 176, 0.15) 0%, rgba(233, 30, 99, 0.15) 100%);
            padding: 2rem;
            border-radius: 15px;
            border: 1px solid rgba(156, 39, 176, 0.3);
            margin-bottom: 2rem;
            text-align: center;
        ">
            <h3 style="color: #ffffff; margin-bottom: 1rem; font-weight: 600; font-size: 1.5rem;">💬 General AI Assistant</h3>
            <p style="color: rgba(255, 255, 255, 0.8); margin: 0; font-size: 1.1rem;">Ask any question - no document upload required!</p>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.general_chat_history:
            st.markdown("""
            <div style="
                background: rgba(255, 255, 255, 0.05);
                padding: 1.5rem;
                border-radius: 12px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                margin: 2rem 0;
            ">
                <h4 style="color: #ffffff; margin-bottom: 1rem; font-weight: 600; text-align: center; font-size: 1.3rem;">💭 Recent General Conversations</h4>
                <p style="color: rgba(255, 255, 255, 0.7); text-align: center; margin: 0;">Your last 5 general chat conversations</p>
            </div>
            """, unsafe_allow_html=True)

            for i, (q, a) in enumerate(reversed(st.session_state.general_chat_history[-5:])):
                with st.expander(f"💬 {q[:60]}..." if len(q) > 60 else f"💬 {q}", expanded=False):
                    st.markdown(f"""
                    <div class="chat-message">
                        <div class="chat-question" style="color: #ffffff !important; background: rgba(102, 126, 234, 0.3) !important; padding: 1rem !important; border-radius: 10px !important; border-left: 4px solid #667eea !important; font-weight: 600 !important; font-size: 1.1rem !important; margin-bottom: 0.8rem !important;">
                            <strong style="color: #ffffff !important;">Q:</strong> <span style="color: #ffffff !important; font-weight: 600 !important;">{q}</span>
                        </div>
                        <div class="chat-answer">{a}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # Enhanced input section
        st.markdown("""
        <div style="
            background: rgba(102, 126, 234, 0.1);
            padding: 2rem;
            margin: 2rem 0;
            border-radius: 15px;
            border: 1px solid rgba(102, 126, 234, 0.3);
            backdrop-filter: blur(10px);
        ">
            <h4 style="color: #ffffff; margin-bottom: 1rem; font-weight: 600; text-align: center; font-size: 1.3rem;">🤔 Ask Anything</h4>
            <p style="color: rgba(255, 255, 255, 0.8); text-align: center; margin-bottom: 1.5rem; font-size: 1rem;">Get instant answers to any question!</p>
        </div>
        """, unsafe_allow_html=True)

        # Create a container for better input visibility
        with st.container():
            st.markdown("""
            <div style="
                background: rgba(255, 255, 255, 0.05);
                padding: 1.5rem;
                border-radius: 12px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                margin: 1rem 0;
            ">
            </div>
            """, unsafe_allow_html=True)

            direct_question = st.text_input(
                "Ask any question:",
                placeholder="What is artificial intelligence? How does machine learning work? Explain quantum computing...",
                key="direct_question",
                label_visibility="collapsed"
            )

        # Always show the button
        if st.button("🚀 Get AI Response", type="primary", use_container_width=True, key="direct_answer_btn"):
            if not direct_question:
                st.warning("⚠️ Please enter a question first!")
            else:
                # Add custom styling for spinner visibility
                st.markdown("""
                <style>
                div[data-testid="stSpinner"] * {
                    color: #ffffff !important;
                    font-weight: 500 !important;
                    font-size: 1rem !important;
                }
                </style>
                """, unsafe_allow_html=True)

                with st.spinner("🤖 AI is thinking..."):
                    llm = get_llm()
                    if llm:
                        answer = answer_direct_question(direct_question, llm)

                        st.markdown(f"""
                        <div class="chat-message fade-in-up">
                            <div class="chat-question" style="color: #ffffff !important; background: rgba(102, 126, 234, 0.3) !important; padding: 1rem !important; border-radius: 10px !important; border-left: 4px solid #667eea !important; font-weight: 600 !important; font-size: 1.1rem !important; margin-bottom: 0.8rem !important;">
                                <strong style="color: #ffffff !important;">Q:</strong> <span style="color: #ffffff !important; font-weight: 600 !important;">{direct_question}</span>
                            </div>
                            <div class="chat-answer">{answer}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        st.session_state.general_chat_history.append((direct_question, answer))

                        if len(st.session_state.general_chat_history) > 10:
                            st.session_state.general_chat_history = st.session_state.general_chat_history[-10:]

        if st.session_state.general_chat_history:
            st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
            if st.button("🗑️ Clear General Chat History", type="secondary", use_container_width=True):
                # Clear general chat history only
                st.session_state.general_chat_history = []

                # Clear input fields by removing them from session state
                if 'direct_question' in st.session_state:
                    del st.session_state['direct_question']

                st.success("✨ General chat history cleared!")
                time.sleep(0.5)
                st.rerun()

    # Footer with modern styling
    st.markdown("""
    <div style="margin-top: 4rem; text-align: center; padding: 2rem; background: rgba(255, 255, 255, 0.05); border-radius: 16px; backdrop-filter: blur(10px);">
        <p style="color: #a0a0a0; margin: 0; font-size: 0.9rem;">
            🤖 Powered by <strong style="color: #667eea;">Neura Talk : ChatBot</strong> •
            Built with <strong style="color: #667eea;">Streamlit</strong> & <strong style="color: #667eea;">LangChain</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()