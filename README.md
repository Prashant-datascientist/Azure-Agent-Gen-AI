# Azure-Agent
### Detailed Documentation

This project is a Streamlit-based web app that allows users to interact with a PDF document via a conversational interface powered by Azure OpenAI and Azure Cognitive Search. Below is a breakdown of the key components and functionality of the code.

---

### 1. **Environment Setup**

```python
import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex, SimpleField, SearchableField, SearchField, VectorSearch, VectorSearchProfile, HnswAlgorithmConfiguration
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents.models import VectorizedQuery
import openai
from openai import AzureOpenAI
from dotenv import load_dotenv
import PyPDF2
```

**Explanation:**
- **Environment Variables**: `load_dotenv()` loads Azure credentials and keys from a `.env` file for use throughout the script.
- **Packages**: This imports necessary packages including `Streamlit`, Azure Search clients, Langchain for text chunking, and PyPDF2 for extracting PDF content.

---

### 2. **Environment Variables Setup**

```python
# Load environment variables
load_dotenv()

# Azure OpenAI and Azure Search credentials
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_version = os.getenv('API_VERSION')
azure_openai_embedding_deployment = "text-embedding-ada-002"
azure_openai_chat_model = "gpt-35-turbo"
```

**Explanation:**
- The environment variables, such as the Azure Search and OpenAI credentials, are fetched from the `.env` file. These variables are essential for connecting to the Azure services.
- **Embedding Models**: The Azure OpenAI models are defined (`text-embedding-ada-002` for embeddings and `gpt-35-turbo` for chat completions).

---

### 3. **PDF Extraction and Text Chunking**

#### PDF Extraction:
```python
@st.cache_data
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text
```
- **`extract_text_from_pdf()`**: Extracts raw text from the provided PDF file using PyPDF2. The text is read page by page and returned as a concatenated string.

#### Text Chunking:
```python
@st.cache_data
def get_text_chunks(pdf_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(pdf_text)
    return chunks
```
- **`get_text_chunks()`**: Splits the extracted text into smaller, manageable chunks using Langchain's `RecursiveCharacterTextSplitter`. This prepares the text for vectorization and indexing.

---

### 4. **Azure Cognitive Search Index Management**

#### Create Index:
```python
def create_index():
    index_client = SearchIndexClient(SEARCH_ENDPOINT, AzureKeyCredential(SEARCH_ADMIN_KEY))
    # ...
```

- **`create_index()`**: This function sets up a vectorized search index in Azure Cognitive Search.
- **Fields and Vector Search**: The index is created with fields like `id`, `fileName`, `content`, and `contentEmbeddings`. `contentEmbeddings` stores the vector embeddings of the text.
- **Vector Search**: Uses HnswAlgorithm for approximate nearest neighbor searches.

---

### 5. **Document Indexing in Azure Cognitive Search**

```python
def index_documents_to_azure_search(text_chunks):
    client = AzureOpenAI(
        azure_deployment=azure_openai_embedding_deployment,
        azure_endpoint=azure_openai_endpoint,
        api_key=azure_openai_key,
        azure_ad_token_provider=DefaultAzureCredential() if not azure_openai_key else None,
        api_version=azure_openai_version
    )
    # ...
```

- **`index_documents_to_azure_search()`**: This function loops through the text chunks and generates embeddings for each using the Azure OpenAI service. The embeddings are then indexed into Azure Cognitive Search for future retrieval.
- **Uploading Documents**: The embeddings and the content chunks are uploaded to the Azure Search index.

---

### 6. **Embedding-based Search Query**

```python
def search_embeddings(user_question):
    client_search = SearchClient(endpoint=SEARCH_ENDPOINT, index_name=SEARCH_INDEX_NAME, credential=AzureKeyCredential(SEARCH_ADMIN_KEY))
    # Generate embeddings for the user's question using Azure OpenAI
    # ...
```

- **`search_embeddings()`**: 
  - Takes a user question, generates its embeddings using the OpenAI `text-embedding-ada-002` model, and performs a vector search on the `contentEmbeddings` field in Azure Cognitive Search.
  - Returns the top 3 relevant chunks based on nearest neighbors.

---

### 7. **Generating Responses with GPT-35-turbo**

```python
def get_chat_response(user_question):
    search_results = search_embeddings(user_question)
    prompt = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
        If the answer is not in the provided context, just say, "answer is not available in the context", don't provide the wrong answer.\n\n
        Answer:
    """
    # Use search results to generate a response...
```

- **`get_chat_response()`**: After performing a vector search based on the user question, this function constructs a prompt using the retrieved content. It then uses the `gpt-35-turbo` model to generate a detailed response to the user's query.
- If the relevant information is found in the search results, it is added to the prompt for GPT-3 to generate a response. Otherwise, it defaults to a message that the answer isn't available.

---

### 8. **Streamlit Front-End**

```python
def main():
    st.header("Chat with PDF using Azure OpenAI GPT-35-turbo")
    user_question = st.text_input("Ask a question based on the PDF content")
    # ...
```

- **Streamlit UI**: The app is configured with a simple text input for user questions. It displays chat history and processes new questions via the `get_chat_response()` function.
- **Index Creation**: If the search index does not exist, the app processes the PDF file, chunks the text, creates the index, and uploads the chunks into Azure Cognitive Search.

---

### 9. **Main Workflow Execution**

```python
if __name__ == "__main__":
    main()
```
- **Execution**: The `main()` function is the entry point of the Streamlit app, allowing the user to interact with the PDF and ask questions in a conversational manner.

---

### Conclusion

This code demonstrates how to build a PDF-based conversational AI interface using Azure OpenAI for embeddings and GPT-based responses, combined with Azure Cognitive Search for vector-based search. The core functionalities include:
- Extracting and chunking PDF text.
- Setting up a vector search index in Azure Cognitive Search.
- Generating vector embeddings for both content and queries.
- Retrieving relevant content and generating responses using OpenAI's GPT model.
- A simple UI using Streamlit to interact with the system.

This setup is ideal for querying large documents (e.g., clinical trials, research papers) and getting answers based on pre-indexed content.
