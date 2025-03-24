---
title: "Setting up an AI/LLM Stack in Haiku: A Practical Guide part II"
date: 2025-03-19T18:30:00-00:00
categories:
  - New Adventures In AI
tags:
  - AI
  - Haiku
  - Python
comments:
  id: 114216921183626942
  #host: mastodon.social
  #user: nexus_6
---

## Using AI Components in Haiku

In the [previous post]({% post_url 2025-19-03-Setup-an-environment-for-AI-in-Haiku-Part-1 %}) we have seen which tools and frameworks are available in Haiku and how to install them.
Now let's see how to use these with some practical examples. You can find the code used in this post in [this repository](https://github.com/nexus6-haiku/ai-stack-haiku-examples){:target="_blank"}.

### Connecting to llama.cpp via OpenAI API and LangChain

```python
from langchain_openai import OpenAI
from langchain.llms.llamacpp import LlamaCpp
import os

# Option 1: Connect to local llama.cpp
def connect_local_llama():
    # Path to the downloaded model (e.g., Mistral 7B)
    model_path = "/Dati/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

    # Parameters optimized for CPU
    llm = LlamaCpp(
        model_path=model_path,
        temperature=0.7,
        max_tokens=512,
        n_ctx=2048,
        n_batch=512,  # Reduced batch size for CPU
        verbose=False
    )

    return llm

# Option 2: Connect to an OpenAI-compatible API
def connect_openai_compatible(base_url="<http://localhost:8000/v1>"):
    # Configuration for an OpenAI-compatible API server
    os.environ["OPENAI_API_KEY"] = "not-needed-for-local"
    os.environ["OPENAI_API_BASE"] = base_url

    llm = OpenAI(
        temperature=0.7,
        model_name="mistral-7b"  # The name must match the one configured in the server
    )

    return llm

# Test the connection
def test_connection(llm):
    result = llm.invoke("What are Large Language Models?")
    print(result)

# Example of usage
if __name__ == "__main__":
    # Choose one of the options
    llm = connect_local_llama()
    # llm = connect_openai_compatible()

    test_connection(llm)
```

In this first example, I'm demonstrating how to establish a connection with a llama.cpp model. There are two main approaches:

1. **Direct connection to local llama.cpp**: This option uses the model directly from the llama-cpp-python library, which we installed previously. It's useful when we want to run the model completely locally.
2. **Connection to an OpenAI-compatible API**: If we have a local or remote server that exposes an OpenAI-compatible API (like llama.cpp with the server option), we can use this configuration.

The parameters have been optimized for CPU execution, reducing the batch size and limiting the context to improve performance. Obviously, on Haiku, the absence of GPU acceleration will impose limitations on the size of usable models, but lighter models like quantized Mistral 7B can work acceptably well.

### Creating Embeddings Using llama.cpp and FAISS

```python
import numpy as np
from langchain_community.embeddings import LlamaCppEmbeddings
import faiss

# Configure embeddings
def setup_embeddings():
    # Path to the embedding model
    embed_model_path = "/Dati/models/ggml-all-MiniLM-L6-v2-f16.gguf"

    # Initialize the embedding model
    embeddings = LlamaCppEmbeddings(
        model_path=embed_model_path,
        n_ctx=512,  # Reduced context to improve performance
        n_batch=32,  # Reduced batch for CPU
        verbose=False
    )

    return embeddings

# Create a FAISS index
def create_faiss_index(documents, embeddings):
    # Get embeddings for each document
    embedded_documents = []
    for doc in documents:
        embed = embeddings.embed_query(doc)
        embedded_documents.append(embed)

    # Convert to numpy array
    vectors = np.array(embedded_documents).astype('float32')

    # Create the FAISS index
    dimension = len(vectors[0])
    index = faiss.IndexFlatL2(dimension)  # Simple L2 index, efficient on CPU

    # Add vectors to the index
    index.add(vectors)

    return index, documents

# Example usage
if __name__ == "__main__":
    # Sample documents
    documents = [
        "Haiku is an open source operating system inspired by BeOS",
        "Python is a versatile and powerful programming language",
        "Large Language Models are neural network-based models",
        "FAISS is a library for vector similarity search",
        "The CPU (Central Processing Unit) is the brain of a computer"
    ]

    # Configure embeddings
    embeddings = setup_embeddings()

    # Create the FAISS index
    index, docs = create_faiss_index(documents, embeddings)

    print(f"FAISS index created with {index.ntotal} vectors of dimension {index.d}")
```

In this example, I'm demonstrating how to create embeddings (vector representations of text) using llama.cpp and how to use FAISS to index them. Embeddings are fundamental for many AI applications, such as semantic search, recommendation systems, and RAG (Retrieval Augmented Generation).

I'm using a lightweight embedding model based on MiniLM, which is efficient even on CPUs. For Haiku, it's important to select quantized embedding models (like those in GGUF format) to reduce memory consumption and improve performance.

The FAISS configuration has been kept simple, using a flat L2 index that works well on CPU without requiring excessive memory. For larger collections, it might be necessary to explore other types of FAISS indices that offer a better trade-off between speed and precision.

### Content Retrieval via Semantic Query

```python
import numpy as np
from langchain_community.embeddings import LlamaCppEmbeddings
import faiss

def semantic_search(query, index, documents, embeddings, k=3):
    """
    Performs a semantic search using FAISS

    Args:
        query (str): The search query
        index (faiss.Index): The FAISS index
        documents (list): List of original documents
        embeddings (LlamaCppEmbeddings): Embedding model
        k (int): Number of results to return

    Returns:
        list: The k most relevant documents
    """
    # Create the query embedding
    query_vector = embeddings.embed_query(query)

    # Convert the embedding to a numpy array
    query_vector = np.array([query_vector]).astype('float32')

    # Search in the FAISS index
    distances, indices = index.search(query_vector, k)

    # Collect the results
    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1:  # -1 indicates that not enough results were found
            results.append({
                'document': documents[idx],
                'score': float(distances[0][i])  # Convert to standard Python float
            })

    return results

# Example usage
def demo_semantic_search():
    # Sample documents
    documents = [
        "Haiku is an open source operating system inspired by BeOS",
        "Python is a versatile and powerful programming language",
        "Large Language Models are models based on neural networks",
        "FAISS is a library for vector similarity search",
        "The CPU (Central Processing Unit) is the brain of a computer",
        "Neural networks are the foundation of modern artificial intelligence",
        "BeOS was an advanced multimedia operating system for its time",
        "Artificial intelligence requires powerful computing capabilities"
    ]

    # Configure embeddings
    embed_model_path = "/Dati/models/ggml-all-MiniLM-L6-v2-f16.gguf"
    embeddings = LlamaCppEmbeddings(
        model_path=embed_model_path,
        n_ctx=512,
        n_batch=32,
        verbose=False
    )

    # Create embeddings and FAISS index
    embedded_documents = []
    for doc in documents:
        embed = embeddings.embed_query(doc)
        embedded_documents.append(embed)

    vectors = np.array(embedded_documents).astype('float32')
    dimension = len(vectors[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    # Run a sample query
    query = "What is the relationship between AI and computing power?"
    results = semantic_search(query, index, documents, embeddings, k=3)

    print("Query:", query)
    print("Most relevant results:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['document']} (score: {result['score']:.4f})")

if __name__ == "__main__":
    demo_semantic_search()
```

In this third example, I'm showing how to use embeddings and FAISS to implement semantic search. Semantic search allows us to find relevant documents based on meaning rather than simple keyword matching.

The process is relatively simple:

1. We convert the query into an embedding vector
2. We use FAISS to find the most similar vectors in the index
3. We retrieve the original documents corresponding to the found vectors

An interesting feature of FAISS is that it also returns a distance score (in this case, the L2 distance), which indicates how relevant the document is to the query. The lower the score, the more semantically similar the document is to the query.

On Haiku, this type of semantic search works well even with the CPU, especially for moderately sized document collections.

## Complete Implementation: Simple RAG System

Now let's put together all the components we've seen so far to create a simple but functional RAG (Retrieval Augmented Generation) system:

```python
import numpy as np
import os
from langchain.llms.llamacpp import LlamaCpp
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain.prompts import PromptTemplate
import faiss

class SimpleRAGSystem:
    def __init__(self, llm_model_path, embed_model_path):
        """
        Initializes the RAG system

        Args:
            llm_model_path (str): Path to the LLM model
            embed_model_path (str): Path to the embedding model
        """
        # Initialize the LLM model
        self.llm = LlamaCpp(
            model_path=llm_model_path,
            temperature=0.7,
            max_tokens=512,
            n_ctx=2048,
            n_batch=512,
            verbose=False
        )

        # Initialize the embedding model
        self.embeddings = LlamaCppEmbeddings(
            model_path=embed_model_path,
            n_ctx=512,
            n_batch=32,
            verbose=False
        )

        # Prompt template for RAG
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            Use the following context to answer the question.

            Context:
            {context}

            Question: {question}

            Answer:
            """
        )

        # Storage for documents and FAISS index
        self.documents = []
        self.index = None
        self.is_index_built = False

    def add_documents(self, documents):
        """
        Adds documents to the knowledge base

        Args:
            documents (list): List of documents to add
        """
        self.documents.extend(documents)
        self.is_index_built = False

    def build_index(self):
        """
        Builds the FAISS index from documents
        """
        if not self.documents:
            print("No documents to index")
            return

        # Create embeddings for all documents
        embedded_documents = []
        for doc in self.documents:
            embed = self.embeddings.embed_query(doc)
            embedded_documents.append(embed)

        # Convert to numpy array
        vectors = np.array(embedded_documents).astype('float32')

        # Create the FAISS index
        dimension = len(vectors[0])
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(vectors)

        self.is_index_built = True
        print(f"FAISS index built with {len(self.documents)} documents")

    def retrieve(self, query, top_k=3):
        """
        Retrieves the most relevant documents for the query

        Args:
            query (str): The search query
            top_k (int): Number of documents to retrieve

        Returns:
            list: The most relevant documents
        """
        if not self.is_index_built:
            self.build_index()

        # Create the query embedding
        query_vector = self.embeddings.embed_query(query)
        query_vector = np.array([query_vector]).astype('float32')

        # Search in the FAISS index
        distances, indices = self.index.search(query_vector, top_k)

        # Collect the results
        retrieved_docs = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                retrieved_docs.append(self.documents[idx])

        return retrieved_docs

    def answer(self, question):
        """
        Answers a question using the RAG system

        Args:
            question (str): The question

        Returns:
            str: The generated answer
        """
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(question)

        if not retrieved_docs:
            return "I couldn't find relevant information to answer your question."

        # Join the retrieved documents into a single context
        context = "\\n\\n".join(retrieved_docs)

        # Create the complete prompt
        prompt = self.prompt_template.format(context=context, question=question)

        # Generate the response
        response = self.llm.invoke(prompt)

        return response

# Example usage
if __name__ == "__main__":
    # Paths to models
    llm_model_path = "/Dati/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    embed_model_path = "/Dati/models/ggml-all-MiniLM-L6-v2-f16.gguf"

    # Create the RAG system
    rag_system = SimpleRAGSystem(llm_model_path, embed_model_path)

    # Add sample documents
    documents = [
        "Haiku is an open-source operating system inspired by BeOS, born in 2001 as a spiritual continuation project of BeOS.",
        "BeOS was a highly advanced multimedia operating system for the '90s, developed by Be Inc.",
        "Haiku uses a block-level file system called BFS (Be File System) that supports extended attributes and indexing.",
        "The CPU (Central Processing Unit) is the main component of a computer responsible for executing calculations.",
        "GPUs (Graphics Processing Units) are specialized processors for graphics rendering and parallel calculations.",
        "Large Language Models (LLMs) are artificial intelligence models trained on enormous amounts of text.",
        "FAISS (Facebook AI Similarity Search) is a library for efficient vector similarity search in large datasets.",
        "Python is an interpreted, high-level, and general-purpose programming language created by Guido van Rossum.",
        "Machine learning is a subset of artificial intelligence that focuses on automatic learning from data.",
        "RAG (Retrieval Augmented Generation) is a technique that combines information retrieval with text generation."
    ]

    rag_system.add_documents(documents)

    # Test the RAG system
    questions = [
        "What is Haiku and what is its origin?",
        "How do Large Language Models work?",
        "What are the differences between CPU and GPU?",
        "What is RAG and how does it work?"
    ]

    for q in questions:
        print("\\nQuestion:", q)
        print("\\nAnswer:", rag_system.answer(q))
```

In this complete example, I've created a `SimpleRAGSystem` class that:

1. Initializes both an LLM and an embedding model
2. Manages a collection of documents and builds a FAISS index for fast retrieval
3. Provides methods to retrieve relevant documents based on a query
4. Generates responses to questions using the retrieved context

The RAG process works in two main steps:

1. **Retrieval**: When a question is asked, the system finds the most relevant documents in its knowledge base using the semantic similarity between the question and the documents.
2. **Generation**: The system then uses these relevant documents as context to help the LLM generate a more informed and accurate response.

This approach is particularly useful on Haiku because it allows us to leverage external knowledge without loading all the information into the LLM's context, which would require more memory and processing power.

## Final Considerations

I've shown how to configure and use an AI/LLM stack in Haiku, demonstrating that it's possible to run language models and semantic retrieval systems even on a lightweight operating system without GPU acceleration.

LLM models, even quantized ones, require a significant amount of memory. For an acceptable experience, I recommend:

1. **Using quantized models**: Models in GGUF format with 4-bit or 3-bit quantization can significantly reduce memory consumption.
2. **Limiting the context**: Reducing the context size (`n_ctx`) is one of the most effective ways to reduce memory usage.
3. **Considering cloud services**: For larger models or intensive workloads, it's always possible to use AI cloud services through their APIs.

From my experiments, the memory footprint when running the `SimpleRAGSystem` is around 1.7GB.

Despite the limitations, Haiku offers a stable and responsive environment for AI experimentation, especially for educational projects or developing applications that require moderate-sized language models.