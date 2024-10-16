import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex, SimpleField, SearchableField, VectorSearch, VectorSearchProfile, HnswAlgorithmConfiguration
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
# from azure.ai.openai import AzureOpenAI
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure OpenAI and Azure Search credentials
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
azure_openai_embedding_deployment = "text-embedding-ada-002"
azure_openai_chat_model = "gpt-35-turbo"

# Constants
SEARCH_INDEX_NAME = "pdf-search-index"

# Step 1: Extract Text from PDF (Simulate PDF Chunking)
def get_text_chunks(pdf_text):
    """Splits extracted PDF text into smaller chunks for better indexing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(pdf_text)
    return chunks

# Step 2: Create the Azure Search Index
def create_index():
    """Creates an Azure Cognitive Search index with vector search."""
    index_fields = [
        SimpleField(name="id", type="Edm.String", key=True, filterable=True, sortable=True),
        SearchableField(name="fileName", type="Edm.String", filterable=True),
        SearchableField(name="content", type="Edm.String", filterable=True),
        SearchField(name="contentEmbeddings", type="Collection(Edm.Single)", vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile")
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(name="myHnsw")
        ],
        profiles=[
            VectorSearchProfile(name="myHnswProfile", algorithm_configuration_name="myHnsw")
        ]
    )

    index = SearchIndex(name=SEARCH_INDEX_NAME, fields=index_fields, vector_search=vector_search)
    index_client = SearchIndexClient(SEARCH_ENDPOINT, AzureKeyCredential(SEARCH_ADMIN_KEY))

    try:
        existing_index = index_client.get_index(SEARCH_INDEX_NAME)
        print(f"Index '{SEARCH_INDEX_NAME}' already exists. Deleting the existing index...")
        index_client.delete_index(SEARCH_INDEX_NAME)
        print(f"Index '{SEARCH_INDEX_NAME}' has been deleted.")
    except Exception as e:
        print(f"Index '{SEARCH_INDEX_NAME}' not found.")

    try:
        print(f"Creating index '{SEARCH_INDEX_NAME}'...")
        index_client.create_index(index)
        print(f"Index '{SEARCH_INDEX_NAME}' has been created successfully.")
    except Exception as e:
        print(f"Error creating index '{SEARCH_INDEX_NAME}': {e}")

# Step 3: Index Documents with Embeddings to Azure Cognitive Search
def index_documents_to_azure_search(text_chunks):
    """Generates embeddings and indexes chunks of text into Azure Cognitive Search."""
    openai_credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(openai_credential, "https://cognitiveservices.azure.com/.default")
    client = AzureOpenAI(
        azure_deployment=azure_openai_embedding_deployment,
        azure_endpoint=azure_openai_endpoint,
        api_key=azure_openai_key,
        azure_ad_token_provider=token_provider if not azure_openai_key else None
    )

    client_search = SearchClient(endpoint=SEARCH_ENDPOINT, index_name=SEARCH_INDEX_NAME, credential=AzureKeyCredential(SEARCH_ADMIN_KEY))

    for i, chunk in enumerate(text_chunks):
        document = {
            "id": str(i),
            "fileName": "sample_pdf_file.pdf",  # Replace this with the actual file name
            "content": chunk,
            "contentEmbeddings": []  # Placeholder for embeddings
        }

        # Create embeddings for the content chunk
        try:
            response = client.embeddings.create(input=chunk, model="text-embedding-ada-002")
            embeddings = response.data[0].embedding
            document["contentEmbeddings"] = embeddings  # Store the embeddings
        except Exception as e:
            print(f"Error generating embeddings for chunk {i}: {e}")

        # Upload the document with embeddings to Azure Search
        try:
            client_search.upload_documents(documents=[document])
            print(f"Document chunk '{document['id']}' indexed successfully.")
        except Exception as e:
            print(f"Failed to index document chunk '{document['id']}': {e}")

# Step 4: Chatbot Functionality
def get_chat_response(user_question):
    """Generates a response from Azure OpenAI GPT-35-turbo based on the user's question."""
    try:
        # Simulate the chat model with GPT-35-turbo
        client = AzureOpenAI(
            azure_endpoint=azure_openai_endpoint,
            api_key=azure_openai_key,
            deployment=azure_openai_chat_model,
        )

        # Construct the prompt for the GPT-35-turbo model
        response = client.chat(messages=[{"role": "user", "content": user_question}], model=azure_openai_chat_model)
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error generating chat response: {e}")
        return "Error generating response. Please try again."

# Step 5: Streamlit UI
def main():
    st.set_page_config(page_title="Chat with PDF via Azure AI", layout="wide")
    st.header("Chat with PDF using Azure OpenAI GPT-35-turbo")

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Clear chat history button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Chat history cleared.")

    # Input box for user question
    user_question = st.text_input("Ask a question based on the PDF content")

    if user_question:
        with st.spinner("Generating a response..."):
            response = get_chat_response(user_question)
            st.session_state.chat_history.append((user_question, response))

    # Display chat history in reverse order (most recent first)
    if st.session_state.chat_history:
        st.subheader("Chat History")
        for i, (question, response) in enumerate(reversed(st.session_state.chat_history)):
            st.write(f"**Q{i+1}:** {question}")
            st.write(f"**A{i+1}:** {response}")
            st.write("---")

    # Backend processing for indexing the PDF content (run once)
    if not os.path.exists("search_index"):
        with st.spinner("Processing PDF..."):
            raw_text = "Simulated PDF content here..."  # Simulate PDF content or call a PDF reader function
            text_chunks = get_text_chunks(raw_text)
            create_index()
            index_documents_to_azure_search(text_chunks)
            st.success("PDF processed and indexed successfully.")

if __name__ == "__main__":
    main()