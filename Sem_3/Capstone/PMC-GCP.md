# 📘 Clinical Treatment Summary Generator using Gemini on GCP
# Objective: Use Gemini Pro for prompt-based clinical summarization with Agentic AI and RAG

# ✅ Step 1: Enable Generative Language API
# (Done via Google Cloud Console)
# - Enable API: https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com
# - Create Service Account or use API key for authentication

## Agentic.py
```py
# ✅ Step 2: Install Required SDKs
# pip install google-cloud-aiplatform langchain faiss-cpu tiktoken

from vertexai.preview.generative_models import GenerativeModel
from langchain.tools import tool
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatVertexAI
from langchain.vectorstores import FAISS, VectorStore
from langchain.embeddings import VertexAIEmbeddings
from langchain.docstore.document import Document
from langchain.vectorstores.google_vertexai import GoogleVertexAIVectorStore
import os

# ✅ Step 3: Initialize Gemini Pro model via LangChain interface
llm = ChatVertexAI(model_name="gemini-pro", temperature=0.3)

# ✅ Step 4: Create Prompt Template and Function Tool
@tool
def summarize_treatment(context: str) -> str:
    """Summarizes treatment plan from clinical notes."""
    prompt = f"""
    You are a clinical AI assistant.
    Summarize the treatment approach based on the following patient case:

    {context}

    Give the treatments in clear bullet points.
    """
    model = GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text

# ✅ Step 5: Setup Retrieval-Augmented Generation (RAG)

# Sample corpus
documents = [
    Document(page_content="Treatment involved slow prone positioning with oxygen therapy."),
    Document(page_content="Breathing exercises were adapted to avoid desaturation and coughing."),
    Document(page_content="Patient was monitored during low-intensity walking sessions.")
]

# Use VertexAI embedding + FAISS
embedding_model = VertexAIEmbeddings()

# ✅ Store embeddings in Google Cloud Vertex AI Vector DB
vectorstore = GoogleVertexAIVectorStore.from_documents(
    documents=documents,
    embedding=embedding_model,
    project_id="your-gcp-project-id",
    location="us-central1",
    index_id="treatment-index",
    endpoint_id="treatment-index-endpoint"
)

# Simulate a clinical query
query = "How was the patient's oxygen level stabilized during COVID-19 treatment?"
retrieved_docs = vectorstore.similarity_search(query, k=2)
retrieved_chunks = "\n".join([doc.page_content for doc in retrieved_docs])

# ✅ Step 6: Final Prompt Construction for RAG
rag_prompt = f"""
Summarize the treatments provided based on the following notes:
{retrieved_chunks}

Answer the query: {query}
"""

# Generate using Gemini
gemini_model = GenerativeModel("gemini-pro")
rag_response = gemini_model.generate_content(rag_prompt)
print("\n✅ Gemini RAG Output:\n")
print(rag_response.text)

# ✅ Step 7: Agent with Tools
tools = [
    Tool(
        name="summarize_treatment",
        func=summarize_treatment,
        description="Summarize treatment from patient note"
    )
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# ✅ Step 8: Agentic Function Calling Example
clinical_text = """
This 60-year-old male was hospitalized due to moderate ARDS from COVID-19. Any deep breathing or position changes led to coughing and desaturation.
He was treated with slow prone positioning, breathing modifications, and low-exertion physical therapy.
"""

agent.run(f"Summarize treatment plan for this patient: {clinical_text}")
```

## With VectorStore.py
```py
# 📘 Clinical Treatment Summary Generator using Gemini on GCP
# Objective: Use Gemini Pro for prompt-based clinical summarization with Agentic AI and RAG

# ✅ Step 1: Enable Generative Language API
# (Done via Google Cloud Console)
# - Enable API: https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com
# - Create Service Account or use API key for authentication

# ✅ Step 2: Install Required SDKs
# pip install google-cloud-aiplatform langchain faiss-cpu tiktoken

from vertexai.preview.generative_models import GenerativeModel
from langchain.tools import tool
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatVertexAI
from langchain.vectorstores import FAISS, VectorStore
from langchain.docstore.document import Document
import os

# ✅ Custom Implementation of VertexAIEmbeddings
from langchain.embeddings.base import Embeddings
from google.cloud import aiplatform

class VertexAIEmbeddings(Embeddings):
    def __init__(self, project_id: str, location: str, model: str = "textembedding-gecko"):
        aiplatform.init(project=project_id, location=location)
        self.model = model

    def embed_documents(self, texts):
        from vertexai.language_models import TextEmbeddingModel
        model = TextEmbeddingModel.from_pretrained(self.model)
        return model.get_embeddings(texts)

    def embed_query(self, text: str):
        return self.embed_documents([text])[0]

# ✅ Custom Implementation of GoogleVertexAIVectorStore
from langchain.vectorstores.base import VectorStore

class GoogleVertexAIVectorStore(VectorStore):
    def __init__(self, index_id, endpoint_id, project_id, location, embedding):
        self.project_id = project_id
        self.location = location
        self.index_id = index_id
        self.endpoint_id = endpoint_id
        self.embedding = embedding
        self._init_vertex_client()

    def _init_vertex_client(self):
        from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import MatchingEngineIndexEndpoint
        self.endpoint = MatchingEngineIndexEndpoint(index_endpoint_name=self.endpoint_id)

    @classmethod
    def from_documents(cls, documents, embedding, project_id, location, index_id, endpoint_id):
        # Convert documents to vectors
        texts = [doc.page_content for doc in documents]
        vectors = embedding.embed_documents(texts)
        # Assume that vector storage happens elsewhere (index creation via GCP console or API)
        return cls(index_id=index_id, endpoint_id=endpoint_id, project_id=project_id, location=location, embedding=embedding)

    def similarity_search(self, query: str, k: int = 4):
        query_vector = self.embedding.embed_query(query)
        response = self.endpoint.find_neighbors(query_vector, num_neighbors=k)
        # Mock conversion to documents
        return [Document(page_content=f"Retrieved chunk {i+1}") for i in range(len(response.nearest_neighbors))]

# ✅ Step 3: Initialize Gemini Pro model via LangChain interface
llm = ChatVertexAI(model_name="gemini-pro", temperature=0.3)

# ✅ Step 4: Create Prompt Template and Function Tool
@tool
def summarize_treatment(context: str) -> str:
    """Summarizes treatment plan from clinical notes."""
    prompt = f"""
    You are a clinical AI assistant.
    Summarize the treatment approach based on the following patient case:

    {context}

    Give the treatments in clear bullet points.
    """
    model = GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text

# ✅ Step 5: Setup Retrieval-Augmented Generation (RAG)

# Sample corpus
documents = [
    Document(page_content="Treatment involved slow prone positioning with oxygen therapy."),
    Document(page_content="Breathing exercises were adapted to avoid desaturation and coughing."),
    Document(page_content="Patient was monitored during low-intensity walking sessions.")
]

# Use VertexAI embedding + Google Vector DB
embedding_model = VertexAIEmbeddings(project_id="your-gcp-project-id", location="us-central1")

# ✅ Store embeddings in Google Cloud Vertex AI Vector DB
vectorstore = GoogleVertexAIVectorStore.from_documents(
    documents=documents,
    embedding=embedding_model,
    project_id="your-gcp-project-id",
    location="us-central1",
    index_id="treatment-index",
    endpoint_id="treatment-index-endpoint"
)

# Simulate a clinical query
query = "How was the patient's oxygen level stabilized during COVID-19 treatment?"
retrieved_docs = vectorstore.similarity_search(query, k=2)
retrieved_chunks = "\n".join([doc.page_content for doc in retrieved_docs])

# ✅ Step 6: Final Prompt Construction for RAG
rag_prompt = f"""
Summarize the treatments provided based on the following notes:
{retrieved_chunks}

Answer the query: {query}
"""

# Generate using Gemini
gemini_model = GenerativeModel("gemini-pro")
rag_response = gemini_model.generate_content(rag_prompt)
print("\n✅ Gemini RAG Output:\n")
print(rag_response.text)

# ✅ Step 7: Agent with Tools
tools = [
    Tool(
        name="summarize_treatment",
        func=summarize_treatment,
        description="Summarize treatment from patient note"
    )
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# ✅ Step 8: Agentic Function Calling Example
clinical_text = """
This 60-year-old male was hospitalized due to moderate ARDS from COVID-19. Any deep breathing or position changes led to coughing and desaturation.
He was treated with slow prone positioning, breathing modifications, and low-exertion physical therapy.
"""

agent.run(f"Summarize treatment plan for this patient: {clinical_text}")
```