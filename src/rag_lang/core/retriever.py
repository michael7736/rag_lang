from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_core.embeddings import Embeddings
from typing import Optional

from .vector_store import load_vector_store, get_embedding_function
from .config import logger, app_config

import os

def get_retriever(vector_store: Optional[VectorStore] = None,
                  embeddings: Optional[Embeddings] = None,
                  search_type: str = "similarity",
                  search_kwargs: dict = {"k": 3},
                  collection_name_override: Optional[str] = None,
                  persist_directory_override: Optional[str] = None) -> BaseRetriever:
    """Creates a retriever from a vector store.

    Args:
        vector_store: An existing VectorStore instance. If None, attempts to load from config.
        embeddings: An existing Embeddings instance. If None, calls get_embedding_function().
        search_type: The type of search to perform (e.g., "similarity", "mmr").
        search_kwargs: Keyword arguments for the search (e.g., {"k": 3} for top 3 results).
        collection_name_override: Optional collection name to override config.
        persist_directory_override: Optional persist directory to override config (for Chroma).

    Returns:
        A LangChain BaseRetriever instance.
    """
    if embeddings is None:
        logger.debug("Embeddings not provided to get_retriever, initializing...")
        embeddings = get_embedding_function()

    if vector_store is None:
        logger.info("No vector store provided to get_retriever, attempting to load...")
        try:
            vector_store = load_vector_store(embeddings=embeddings,
                                             collection_name_override=collection_name_override,
                                             persist_directory_override=persist_directory_override)
            logger.info("Successfully loaded vector store for retriever.")
        except FileNotFoundError as e:
            logger.error(f"Failed to load vector store: {e}. Cannot create retriever. Please run ingestion first.")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading vector store for retriever: {e}")
            raise

    logger.info(f"Creating retriever with search_type='{search_type}' and search_kwargs={search_kwargs}")
    retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )
    logger.info("Retriever created successfully.")
    return retriever

# Example Usage (for testing purposes)
if __name__ == '__main__':
    # This example assumes a vector store exists at the default location
    # Ensure OPENAI_API_KEY_EMBEDDING is set and ingestion has been run at least once.
    
    TEST_PERSIST_DIR = app_config.chromadb.persist_directory
    
    if not app_config.embedding.api_key:
        print("OPENAI_API_KEY_EMBEDDING not set in .env")
        exit(1)

    if app_config.vector_store_type == "chroma" and not os.path.exists(TEST_PERSIST_DIR):
        print(f"Error: Default Chroma vector store directory '{TEST_PERSIST_DIR}' not found. Run ingestion first.")
        exit(1)
    # For other DBs, this test relies on them being available and having the default collection/index from config

    logger.info("--- Testing Retriever Creation ---")
    try:
        retriever_instance = get_retriever()
        logger.info(f"Retriever instance created: {type(retriever_instance)}")
        
        test_query = "What is LangChain?" 
        if app_config.vector_store_type == "milvus": test_query = "What is Milvus?"
        elif app_config.vector_store_type == "weaviate": test_query = "What is Weaviate?"
        elif app_config.vector_store_type == "qdrant": test_query = "What is Qdrant?"
        
        logger.info(f"Testing retriever with query: '{test_query}'")
        retrieved_docs = retriever_instance.invoke(test_query)
        logger.info(f"Retrieved {len(retrieved_docs)} documents.")
        for i, doc in enumerate(retrieved_docs):
             logger.info(f"Doc {i+1}: {doc.page_content[:80]}... (Source: {doc.metadata.get('source')})")
        assert len(retrieved_docs) > 0 

    except Exception as e:
        logger.error(f"Error during retriever testing: {e}", exc_info=True) 