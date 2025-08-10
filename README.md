## Multimodal RAG
- Unstructured data (pdf files, html, excel..)
- Structured data (text, tables, images...)
- Summarize the structured data using an LLM
- link doc_id to summary as metadata (same thing with doc)
- Create vector database for the summaries with text embedding and a document database for docs

## Requirements
```
>> winget install poppler

>> pip install -Uq "unstructured[all-docs]" pillow lxml pillow

>> pip install -Uq chromadb tiktoken

>>pip install -Uq langchain langchain-community langchain-openai langchain-groq

>>pip install -Uq python_dotenv

```
