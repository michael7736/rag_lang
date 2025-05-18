import os
import shutil
from typing import List, Optional

import weaviate
from langchain_community.vectorstores import Chroma, Milvus, Weaviate
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings

from .config import app_config, logger

# --- Embedding Function --- 
# Initialize the embedding model based on the configuration
# Ensure API key and potentially base URL are configured in .env
def get_embedding_function() -> Embeddings:
    """Initializes and returns the embedding function based on config."""
    if not app_config.embedding.api_key:
        msg = "OpenAI API Key for embeddings is not configured. Set OPENAI_API_KEY_EMBEDDING in your environment."
        logger.error(msg)
        raise ValueError(msg)
    
    logger.info(f"Initializing OpenAI Embeddings with model: {app_config.embedding.model_name}")
    return OpenAIEmbeddings(
        model=app_config.embedding.model_name,
        openai_api_key=app_config.embedding.api_key,
        openai_api_base=app_config.embedding.api_base
    )

# --- Vector Store Operations --- 

def create_vector_store(documents: List[Document],
                        embeddings: Embeddings,
                        collection_name_override: Optional[str] = None, 
                        persist_directory_override: Optional[str] = None) -> VectorStore:
    """Creates and persists a vector store from documents based on AppConfig."""
    if not documents:
        msg = "Cannot create vector store from an empty list of documents."
        logger.error(msg)
        raise ValueError(msg)

    vector_store_type = app_config.vector_store_type
    logger.info(f"Attempting to create '{vector_store_type}' vector store.")
    logger.info(f"Number of documents to add: {len(documents)}")
    vector_store: VectorStore

    if vector_store_type == "chroma":
        persist_dir = persist_directory_override if persist_directory_override is not None else app_config.chromadb.persist_directory
        col_name = collection_name_override if collection_name_override is not None else app_config.chromadb.collection_name
        logger.info(f"Creating Chroma vector store at '{persist_dir}' with collection '{col_name}'.")
        os.makedirs(persist_dir, exist_ok=True)
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_dir,
            collection_name=col_name
        )
        logger.info(f"Successfully created and persisted ChromaDB store with {vector_store._collection.count()} entries.")
    elif vector_store_type == "milvus":
        milvus_cfg = app_config.milvus
        col_name = collection_name_override if collection_name_override is not None else milvus_cfg.collection_name
        logger.info(f"Creating Milvus vector store: host={milvus_cfg.host}, port={milvus_cfg.port}, collection='{col_name}'.")
        try:
            vector_store = Milvus.from_documents(
                documents,
                embedding=embeddings,
                collection_name=col_name,
                connection_args={"host": milvus_cfg.host, "port": milvus_cfg.port},
                index_params=milvus_cfg.index_params,
            )
            logger.info(f"Successfully created/updated Milvus collection '{col_name}'.")
        except Exception as e:
            logger.error(f"Failed to create Milvus vector store: {e}", exc_info=True)
            raise
    elif vector_store_type == "weaviate":
        wv_cfg = app_config.weaviate
        idx_name = collection_name_override if collection_name_override is not None else wv_cfg.index_name
        logger.info(f"Creating Weaviate vector store (v3 client): url={wv_cfg.get_url()}, index='{idx_name}'.")
        
        auth_client_secret = None
        if wv_cfg.api_key:
            auth_client_secret = weaviate.AuthApiKey(api_key=wv_cfg.api_key)
            logger.info("Weaviate API key configured.")

        client = weaviate.Client(
            url=wv_cfg.get_url(),
            auth_client_secret=auth_client_secret,
            additional_headers={
                 "X-OpenAI-Api-Key": app_config.embedding.api_key 
            }
        )
        try:
            vector_store = Weaviate.from_documents(
                client=client, 
                documents=documents,
                embedding=embeddings,
                index_name=idx_name,
                text_key=wv_cfg.text_key,
            )
            logger.info(f"Successfully added documents to Weaviate class/index '{idx_name}'.")
        except Exception as e:
            logger.error(f"Failed to create/update Weaviate vector store: {e}", exc_info=True)
            raise
    elif vector_store_type == "qdrant":
        qdrant_cfg = app_config.qdrant
        col_name = collection_name_override if collection_name_override is not None else qdrant_cfg.collection_name
        logger.info(f"Creating Qdrant vector store with langchain-qdrant: host={qdrant_cfg.host}, port={qdrant_cfg.port}, collection='{col_name}'.")
        try:
            vector_store = Qdrant.from_documents(
                documents=documents,
                embedding=embeddings,
                host=qdrant_cfg.host,
                port=qdrant_cfg.port,
                grpc_port=qdrant_cfg.grpc_port,
                prefer_grpc=qdrant_cfg.prefer_grpc,
                api_key=qdrant_cfg.api_key,
                collection_name=col_name,
            )
            logger.info(f"Successfully created/updated Qdrant collection '{col_name}' using langchain-qdrant.")
        except Exception as e:
            logger.error(f"Failed to create Qdrant vector store with langchain-qdrant: {e}", exc_info=True)
            raise
    else:
        msg = f"Unsupported vector_store_type: {vector_store_type}. Supported types are 'chroma', 'milvus', 'weaviate', and 'qdrant'."
        logger.error(msg)
        raise ValueError(msg)
    
    return vector_store

def load_vector_store(embeddings: Embeddings, 
                      collection_name_override: Optional[str] = None, 
                      persist_directory_override: Optional[str] = None) -> VectorStore:
    """Loads an existing vector store based on AppConfig."""
    vector_store_type = app_config.vector_store_type
    logger.info(f"Attempting to load existing '{vector_store_type}' vector store.")
    vector_store: VectorStore

    if vector_store_type == "chroma":
        persist_dir = persist_directory_override if persist_directory_override is not None else app_config.chromadb.persist_directory
        col_name = collection_name_override if collection_name_override is not None else app_config.chromadb.collection_name
        if not os.path.isdir(persist_dir) or not os.listdir(persist_dir):
            msg = f"ChromaDB persist directory '{persist_dir}' does not exist or is empty. Cannot load vector store."
            logger.error(msg)
            raise FileNotFoundError(msg)
        logger.info(f"Loading existing ChromaDB store from '{persist_dir}' with collection '{col_name}'.")
        vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=col_name
        )
        logger.info(f"Successfully loaded ChromaDB store with {vector_store._collection.count()} entries.")
    elif vector_store_type == "milvus":
        milvus_cfg = app_config.milvus
        col_name = collection_name_override if collection_name_override is not None else milvus_cfg.collection_name
        logger.info(f"Loading existing Milvus vector store: host={milvus_cfg.host}, port={milvus_cfg.port}, collection='{col_name}'.")
        try:
            vector_store = Milvus(
                embedding_function=embeddings,
                collection_name=col_name,
                connection_args={"host": milvus_cfg.host, "port": milvus_cfg.port},
                index_params=milvus_cfg.index_params, 
                search_params=milvus_cfg.search_params
            )
            logger.info(f"Successfully initialized Milvus client for collection '{col_name}'.")
        except Exception as e:
            logger.error(f"Failed to load Milvus vector store: {e}", exc_info=True)
            raise
    elif vector_store_type == "weaviate":
        wv_cfg = app_config.weaviate
        idx_name = collection_name_override if collection_name_override is not None else wv_cfg.index_name
        logger.info(f"Loading existing Weaviate vector store (v3 client): url={wv_cfg.get_url()}, index='{idx_name}'.")
        
        auth_client_secret = None
        if wv_cfg.api_key:
            auth_client_secret = weaviate.AuthApiKey(api_key=wv_cfg.api_key)

        client = weaviate.Client(
            url=wv_cfg.get_url(),
            auth_client_secret=auth_client_secret,
            additional_headers={
                 "X-OpenAI-Api-Key": app_config.embedding.api_key
            }
        )
        try:
            vector_store = Weaviate(
                client=client, 
                index_name=idx_name,
                text_key=wv_cfg.text_key,
                embedding=embeddings, 
                attributes=None 
            )
            logger.info(f"Successfully initialized LangChain Weaviate wrapper for class/index '{idx_name}'.")
        except Exception as e:
            logger.error(f"Failed to load Weaviate vector store: {e}", exc_info=True)
            raise
    elif vector_store_type == "qdrant":
        qdrant_cfg = app_config.qdrant
        col_name = collection_name_override if collection_name_override is not None else qdrant_cfg.collection_name
        logger.info(f"Loading existing Qdrant vector store with langchain-qdrant: host={qdrant_cfg.host}, port={qdrant_cfg.port}, collection='{col_name}'.")
        try:
            qdrant_client = QdrantClient(
                host=qdrant_cfg.host,
                port=qdrant_cfg.port,
                grpc_port=qdrant_cfg.grpc_port,
                prefer_grpc=qdrant_cfg.prefer_grpc,
                api_key=qdrant_cfg.api_key
            )
            vector_store = Qdrant(
                client=qdrant_client,
                collection_name=col_name,
                embeddings=embeddings 
            )
            logger.info(f"Successfully initialized Qdrant wrapper for collection '{col_name}' using langchain-qdrant.")
        except Exception as e:
            logger.error(f"Failed to load Qdrant vector store with langchain-qdrant: {e}", exc_info=True)
            raise
    else:
        msg = f"Unsupported vector_store_type: {vector_store_type}. Supported types are 'chroma', 'milvus', 'weaviate', and 'qdrant'."
        logger.error(msg)
        raise ValueError(msg)
        
    return vector_store

# --- Example Usage --- 
if __name__ == '__main__':
    # Ensure OPENAI_API_KEY is set in .env
    # To test Milvus, set VECTOR_STORE_TYPE="milvus" in .env or AppConfig
    # And ensure Milvus server is running at configured host/port (e.g., rtx4080:19530)
    
    logger.info(f"Running vector_store.py example with VECTOR_STORE_TYPE='{app_config.vector_store_type}'")

    try:
        embedding_function = get_embedding_function()
        logger.info("Successfully initialized embedding function.")
    except ValueError as e:
        logger.error(f"Failed to initialize embedding function: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during embedding initialization: {e}")
        exit(1)
        
    test_docs = [
        Document(page_content="Document for testing vector store type: " + app_config.vector_store_type, metadata={"source": "test1.txt", "id": "doc1"}),
        Document(page_content="LangChain makes building LLM applications easier.", metadata={"source": "test2.txt", "id": "doc2"}),
        Document(page_content="Qdrant is a vector similarity search engine.", metadata={"source": "test3.txt", "id": "doc3"})
    ]

    # Specific test directories/collection names for Chroma to avoid conflicts with default
    CHROMA_TEST_PERSIST_DIR = "./chroma_vs_test_store"
    CHROMA_TEST_COLLECTION = "vs_test_collection_chroma"
    
    # For Milvus, the collection name is handled by its config in app_config.milvus.collection_name
    # We will use a distinct collection name for testing if needed, or just the default.
    # For this example, we'll use the default from app_config for Milvus.

    original_chroma_persist_dir = app_config.chromadb.persist_directory
    original_chroma_collection = app_config.chromadb.collection_name
    original_weaviate_index = app_config.weaviate.index_name
    original_qdrant_collection = app_config.qdrant.collection_name

    if app_config.vector_store_type == "chroma":
        app_config.chromadb.persist_directory = CHROMA_TEST_PERSIST_DIR
        app_config.chromadb.collection_name = CHROMA_TEST_COLLECTION
        if os.path.exists(CHROMA_TEST_PERSIST_DIR):
            logger.info(f"Removing existing Chroma test directory: {CHROMA_TEST_PERSIST_DIR}")
            shutil.rmtree(CHROMA_TEST_PERSIST_DIR)
    elif app_config.vector_store_type == "weaviate":
        # For Weaviate, use a test-specific index name to avoid conflicts
        app_config.weaviate.index_name = "VsTestWeaviateCollection"
        # Cleanup for Weaviate would typically involve deleting the class via the client.
        # For this example, we'll rely on a distinct name and manual cleanup if persistent.
        logger.info(f"Using Weaviate test index/class: {app_config.weaviate.index_name}")
    elif app_config.vector_store_type == "qdrant":
        app_config.qdrant.collection_name = "vs_test_qdrant_collection"
        logger.info(f"Using Qdrant test collection: {app_config.qdrant.collection_name}")
        # Cleanup for Qdrant would involve deleting the collection. For simplicity, rely on distinct name.

    logger.info("--- Testing Vector Store Creation ---")
    created_store = None
    try:
        created_store = create_vector_store(
            documents=test_docs, 
            embeddings=embedding_function,
            collection_name_override=CHROMA_TEST_COLLECTION if app_config.vector_store_type == "chroma" else None,
            persist_directory_override=CHROMA_TEST_PERSIST_DIR if app_config.vector_store_type == "chroma" else None
        )
        logger.info(f"Vector store creation successful for type '{app_config.vector_store_type}'.")
    except Exception as e:
        logger.error(f"Error during vector store creation: {e}", exc_info=True)
        # Revert any config changes before exiting
        if app_config.vector_store_type == "chroma":
            app_config.chromadb.persist_directory = original_chroma_persist_dir
            app_config.chromadb.collection_name = original_chroma_collection
        elif app_config.vector_store_type == "weaviate":
            app_config.weaviate.index_name = original_weaviate_index
        elif app_config.vector_store_type == "qdrant":
            app_config.qdrant.collection_name = original_qdrant_collection
        exit(1)

    if created_store:
        logger.info("--- Testing Vector Store Loading ---")
        try:
            loaded_store = load_vector_store(
                embeddings=embedding_function,
                collection_name_override=CHROMA_TEST_COLLECTION if app_config.vector_store_type == "chroma" else None,
                persist_directory_override=CHROMA_TEST_PERSIST_DIR if app_config.vector_store_type == "chroma" else None
            )
            logger.info(f"Vector store loading successful for type '{app_config.vector_store_type}'.")
            
            query_text = "What is Qdrant?"
            if app_config.vector_store_type == "milvus": query_text = "What is Milvus?"
            if app_config.vector_store_type == "weaviate": query_text = "What is Weaviate?"

            results = loaded_store.similarity_search(query_text, k=1)
            logger.info(f"Similarity search result for '{query_text}': {results}")
            if app_config.vector_store_type == "chroma":
                 assert loaded_store._collection.count() > 0
            # Add specific assertions for Milvus/Weaviate if possible (e.g., checking if collection/class has items)
            if results:
                expected_content = "testing vector store"
                if app_config.vector_store_type == "weaviate": expected_content = "Weaviate"
                elif app_config.vector_store_type == "milvus": expected_content = "Milvus"
                elif app_config.vector_store_type == "qdrant": expected_content = "Qdrant"
                assert expected_content in results[0].page_content or "testing vector store" in results[0].page_content
            else:
                logger.warning("Similarity search returned no results.")

        except Exception as e:
            logger.error(f"An unexpected error occurred during vector store loading or search: {e}", exc_info=True)

    # Cleanup and revert config changes
    if app_config.vector_store_type == "chroma":
        if os.path.exists(CHROMA_TEST_PERSIST_DIR):
            logger.info(f"Removing Chroma test directory: {CHROMA_TEST_PERSIST_DIR}")
            shutil.rmtree(CHROMA_TEST_PERSIST_DIR)
        app_config.chromadb.persist_directory = original_chroma_persist_dir
        app_config.chromadb.collection_name = original_chroma_collection
    elif app_config.vector_store_type == "weaviate":
        # We would typically delete the test class here if it was created by the test
        # client = weaviate.Client(app_config.weaviate.get_url()) # Re-init client if needed
        # if client.schema.exists(app_config.weaviate.index_name):
        #     logger.info(f"Deleting Weaviate test class: {app_config.weaviate.index_name}")
        #     client.schema.delete_class(app_config.weaviate.index_name)
        app_config.weaviate.index_name = original_weaviate_index
    elif app_config.vector_store_type == "qdrant":
        # To fully clean up Qdrant, you'd connect with a qdrant_client and delete the collection
        # For this example, we just reset the config name.
        app_config.qdrant.collection_name = original_qdrant_collection
    
    logger.info("Vector store example script finished.") 