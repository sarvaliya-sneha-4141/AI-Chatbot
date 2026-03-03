# AI-Chatbot

# 📚 PDF AI Assistant

A powerful Streamlit-based application that uses LangChain and AI models to analyze PDF documents, generate summaries, and answer questions with context awareness.

## ✨ Features

- **PDF Processing**: Upload and process PDF documents
- **AI-Powered Summary**: Generate comprehensive summaries using Grok or Gemini models
- **Smart Q&A**: Ask questions and get context-aware answers
- **Out-of-Context Detection**: Automatically identifies when questions are outside the PDF scope
- **Free Embeddings**: Uses Sentence Transformers for cost-effective document embeddings
- **Multiple LLM Support**: Choose between Grok (via Groq) and Gemini models
- **Modern UI**: Beautiful Streamlit interface with responsive design

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd simple
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up API Keys

1. Copy `env_example.txt` to `.env`
2. Get your API keys:
   - **Groq API Key**: Visit [Groq Console](https://console.groq.com/)
   - **Google API Key**: Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
3. Add your keys to the `.env` file:

```env
GROQ_API_KEY=your_actual_groq_key
GOOGLE_API_KEY=your_actual_google_key
```

### 5. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🎯 How It Works

### PDF Processing Pipeline

1. **Upload**: User uploads a PDF file
2. **Text Extraction**: PyPDF2 extracts text content
3. **Chunking**: Text is split into manageable chunks with overlap
4. **Embeddings**: Sentence Transformers create vector representations
5. **Vector Store**: FAISS stores embeddings for similarity search

### AI Analysis

1. **Summary Generation**: LLM creates comprehensive document summary
2. **Question Processing**: User questions are converted to embeddings
3. **Context Retrieval**: Similarity search finds relevant document chunks
4. **Answer Generation**: LLM generates answers based on retrieved context
5. **Context Validation**: System detects out-of-context questions

## 🔧 Configuration

### Model Selection

- **Grok (via Groq)**: Uses Mixtral-8x7b model for high-quality responses
- **Gemini**: Google's Gemini Pro model for comprehensive analysis

### Embedding Model

- **Sentence Transformers**: Uses `all-MiniLM-L6-v2` for fast, accurate embeddings
- **Chunk Size**: 1000 characters with 200 character overlap
- **Vector Store**: FAISS for efficient similarity search

## 📁 Project Structure

```
simple/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── venv/              # Virtual environment
```

## 🛠️ Dependencies

- **Streamlit**: Web interface framework
- **LangChain**: LLM orchestration and chains
- **Sentence Transformers**: Free embedding models
- **FAISS**: Vector similarity search
- **PyPDF2**: PDF text extraction
- **Python-dotenv**: Environment variable management

## 🎨 Usage Guide

### 1. Upload PDF
- Click "Choose a PDF file" to upload your document
- Click "Process PDF" to analyze the content

### 2. View Summary
- After processing, the AI-generated summary appears automatically
- Summary focuses on main points and key concepts

### 3. Ask Questions
- Type your question in the text input
- Click "Get Answer" to receive context-aware responses
- System automatically detects out-of-context questions

### 4. Model Selection
- Use the sidebar to switch between Grok and Gemini models
- Each model has different strengths and response styles

## 🔍 Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure your `.env` file is properly configured
2. **PDF Processing Failures**: Check if PDF is text-based (not scanned images)
3. **Memory Issues**: Large PDFs may require more RAM
4. **Model Timeouts**: Some models may take time for complex queries

### Performance Tips

- Use smaller PDFs for faster processing
- Close other applications to free up memory
- Ensure stable internet connection for API calls

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **LangChain**: For the amazing LLM orchestration framework
- **Sentence Transformers**: For free, high-quality embeddings
- **Streamlit**: For the beautiful web interface framework
- **FAISS**: For efficient vector similarity search

---

**Happy PDF Analysis! 📖✨** 
