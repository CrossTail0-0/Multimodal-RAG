# Multimodal RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system that processes and queries multimodal documents containing text, tables, and images. This system extracts information from PDF documents, creates summaries using AI models, and enables intelligent question-answering across different content types.

## 🚀 Features

- **Multimodal Document Processing**: Handles PDFs with text, tables, and images
- **Intelligent Content Extraction**: Automatically partitions documents into structured chunks
- **AI-Powered Summarization**: Creates concise summaries using Hugging Face models
- **Vector Database**: Stores document embeddings for efficient retrieval
- **RAG Pipeline**: Combines retrieval and generation for accurate Q&A
- **Multi-Modal Context**: Integrates text, table, and image information in responses

## 🏗️ Architecture

The system follows a sophisticated pipeline:

1. **Document Partitioning**: Uses `unstructured` library to extract text, tables, and images from PDFs
2. **Content Summarization**: 
   - Text/Table summaries using BART-large-CNN model
   - Image descriptions using LLaVA-1.5-7B model
3. **Vector Storage**: ChromaDB for storing document embeddings
4. **Multi-Vector Retrieval**: Links summaries to original content for context preservation
5. **RAG Generation**: GPT-4o-mini for generating comprehensive answers

## 📋 Requirements

### System Dependencies
```bash
# Install Poppler utilities for PDF processing
winget install poppler  # Windows
# OR
apt-get install poppler-utils tesseract-ocr libmagic-dev  # Linux/Ubuntu
```

### Python Dependencies
```bash
pip install -Uq "unstructured[all-docs]" pillow lxml
pip install -Uq chromadb tiktoken
pip install -Uq langchain langchain-community langchain-openai langchain-groq
pip install -Uq python_dotenv
pip install -Uq langchain-huggingface
pip install -Uq transformers
```

## 🔑 API Keys Required

Set the following environment variables:
- `OPENAI_API_KEY`: For GPT-4o-mini model
- `GROQ_API_KEY`: For alternative LLM options
- `LANGCHAIN_API_KEY`: For LangChain tracing
- `HF_TOKEN`: For Hugging Face model access

## 📖 Usage

### 1. Document Processing
```python
from unstructured.partition.pdf import partition_pdf

chunks = partition_pdf(
    filename="document.pdf",
    infer_table_structure=True,
    strategy="hi_res",
    extract_image_block_types=["Image"],
    extract_image_block_to_payload=True,
    chunking_strategy="by_title",
    max_characters=3000
)
```

### 2. Content Summarization
```python
# Text and table summarization
summarization_pipeline = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=-1
)

# Image description
pipe = pipeline("image-text-to-text", model="llava-hf/llava-1.5-7b-hf")
```

### 3. Vector Database Setup
```python
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever

vectorstore = Chroma(collection_name="multi_modal_rag", embedding_function=OpenAIEmbeddings())
store = InMemoryStore()
retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key="doc_id")
```

### 4. Query the System
```python
response = chain.invoke("What is the attention mechanism?")
print(response)
```

## 🔍 Key Components

- **Content Extractor**: Processes PDFs into structured chunks
- **Summarizer**: Creates concise summaries of text, tables, and images
- **Vector Store**: ChromaDB for efficient similarity search
- **Retriever**: Multi-vector approach linking summaries to original content
- **Generator**: GPT-4o-mini for comprehensive answer generation

## 📊 Supported Content Types

- **Text**: Extracted and summarized using BART model
- **Tables**: Converted to HTML and summarized
- **Images**: Described using LLaVA vision-language model
- **Metadata**: Preserved for context and source tracking

## 🎯 Use Cases

- Research paper analysis and Q&A
- Document intelligence and information extraction
- Multimodal content understanding
- Educational content processing
- Technical documentation analysis

## 🚨 Notes

- GPU acceleration recommended for image processing
- Large models require sufficient memory
- API rate limits apply for OpenAI and other services
- Document processing time scales with file size and complexity

## 📁 Project Structure

```
Mltimodal-RAG/
├── multimodal_rag.ipynb    # Main notebook with complete implementation
├── content/                 # Sample PDF documents
├── pyproject.toml          # Project dependencies
├── README.md               # This file
└── uv.lock                 # Lock file for reproducible builds
```

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve this multimodal RAG system.

## 📄 License

This project is open source and available under the MIT License.
