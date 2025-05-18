from langchain_core.runnables import Runnable
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseChatModel
from typing import Optional # Ensure Optional is imported

from ..core.loader import load_documents
from ..core.splitter import split_documents
from ..core.vector_store import create_vector_store, get_embedding_function
from ..core.retriever import get_retriever
from ..core.generator import get_rag_chain, get_llm
from ..core.config import logger, app_config

# --- Global instances (cached for efficiency within a run) --- 
# We can cache these to avoid re-initializing on every query within the same process.
# Be mindful of state if the application becomes more complex.
_embedding_function_instance = None
_llm_instance = None
_retriever_instance = None
_rag_chain_instance = None

def _get_cached_embedding_function():
    global _embedding_function_instance
    if _embedding_function_instance is None:
        logger.debug("Initializing and caching embedding function...")
        _embedding_function_instance = get_embedding_function()
    return _embedding_function_instance

def _get_cached_llm():
    global _llm_instance
    if _llm_instance is None:
        logger.debug("Initializing and caching LLM...")
        _llm_instance = get_llm()
    return _llm_instance

def _get_cached_retriever(force_reload: bool = False, 
                          collection_name_override: Optional[str] = None, 
                          persist_directory_override: Optional[str] = None) -> BaseRetriever:
    global _retriever_instance
    if _retriever_instance is None or force_reload:
        logger.debug(f"Initializing retriever (force_reload={force_reload})...")
        embeddings = _get_cached_embedding_function()
        # Pass overrides to get_retriever, which will pass them to load_vector_store
        _retriever_instance = get_retriever(embeddings=embeddings, 
                                          collection_name_override=collection_name_override,
                                          persist_directory_override=persist_directory_override)
    return _retriever_instance

def get_baseline_rag_pipeline(force_reload_retriever: bool = False, 
                              collection_name_override: Optional[str] = None, 
                              persist_directory_override: Optional[str] = None) -> Optional[Runnable]: # Added Optional return
    """Constructs and returns the baseline RAG query pipeline."""
    global _rag_chain_instance
    if _rag_chain_instance is None or force_reload_retriever:
        logger.info(f"Constructing baseline RAG pipeline (force_reload_retriever={force_reload_retriever})...")
        try:
            retriever = _get_cached_retriever(force_reload=force_reload_retriever, 
                                              collection_name_override=collection_name_override,
                                              persist_directory_override=persist_directory_override)
            llm = _get_cached_llm()
            _rag_chain_instance = get_rag_chain(retriever=retriever, llm=llm)
            logger.info("Baseline RAG pipeline constructed successfully.")
        except FileNotFoundError as e:
             logger.error(f"Failed to construct RAG pipeline: Could not load vector store. {e}")
             logger.error("Please ensure you have run the ingestion pipeline first or configured an existing one.")
             return None 
        except ValueError as e:
             logger.error(f"Failed to construct RAG pipeline due to configuration error: {e}")
             return None 
        except Exception as e:
             logger.error(f"Failed to construct RAG pipeline due to an unexpected error: {e}", exc_info=True)
             return None 
             
    return _rag_chain_instance

def run_ingestion_pipeline(source_path: str,
                           persist_directory: Optional[str] = None,
                           collection_name: Optional[str] = None):
    """Runs the complete document ingestion pipeline: Load -> Split -> Embed -> Store."""
    logger.info(f"Starting ingestion pipeline for source: {source_path}")
    
    vs_type = app_config.vector_store_type
    if persist_directory is None:
        if vs_type == "chroma":
            persist_directory = app_config.chromadb.persist_directory # Corrected: chromadb
        # For Milvus/Weaviate/Qdrant, persist_directory is not directly used by LangChain wrapper in the same way for remote servers.
        # The host/port/URL is the primary locator.
        # If they were configured for local file-based persistence, this would need adjustment.

    if collection_name is None:
        if vs_type == "chroma":
            collection_name = app_config.chromadb.collection_name # Corrected: chromadb
        elif vs_type == "milvus":
            collection_name = app_config.milvus.collection_name
        elif vs_type == "weaviate":
            collection_name = app_config.weaviate.index_name 
        elif vs_type == "qdrant":
            collection_name = app_config.qdrant.collection_name

    logger.info(f"Target vector store type: {vs_type}")
    # Log the actual names/paths that will be used by create_vector_store
    final_collection_name_to_log = collection_name # Use the override if present
    if final_collection_name_to_log is None: # If no override, get from config
        if vs_type == "chroma": final_collection_name_to_log = app_config.chromadb.collection_name
        elif vs_type == "milvus": final_collection_name_to_log = app_config.milvus.collection_name
        elif vs_type == "weaviate": final_collection_name_to_log = app_config.weaviate.index_name
        elif vs_type == "qdrant": final_collection_name_to_log = app_config.qdrant.collection_name
            
    if final_collection_name_to_log:
        logger.info(f"Target collection/index for creation: {final_collection_name_to_log}")
    
    if vs_type == "chroma":
        final_persist_dir_to_log = persist_directory if persist_directory is not None else app_config.chromadb.persist_directory # Corrected: chromadb
        if final_persist_dir_to_log:
            logger.info(f"Target Chroma persist directory for creation: {final_persist_dir_to_log}")

    try:
        logger.info("Step 1: Loading documents...")
        documents = load_documents(source_path)
        if not documents:
            logger.warning(f"No documents found or loaded from source: {source_path}. Aborting ingestion.")
            return
        logger.info(f"Successfully loaded {len(documents)} document(s).")

        logger.info("Step 2: Splitting documents...")
        split_docs = split_documents(documents)
        if not split_docs:
            logger.warning("Document splitting resulted in zero chunks. Aborting ingestion.")
            return
        logger.info(f"Successfully split documents into {len(split_docs)} chunks.")

        logger.info("Step 3: Initializing embedding function...")
        embedding_function = _get_cached_embedding_function() 
        logger.info("Successfully initialized or retrieved embedding function.")

        logger.info("Step 4: Creating/updating vector store...")
        create_vector_store(
            documents=split_docs,
            embeddings=embedding_function,
            collection_name_override=collection_name, # Pass CLI override
            persist_directory_override=persist_directory    # Pass CLI override (relevant for Chroma)
        )
        logger.info("Successfully created/updated vector store.")
        
        global _retriever_instance, _rag_chain_instance
        _retriever_instance = None 
        _rag_chain_instance = None
        logger.info("Invalidated cached retriever and RAG chain due to ingestion.")
        
        logger.info("Ingestion pipeline completed successfully.")

    except ValueError as ve:
        logger.error(f"Configuration error during ingestion: {ve}", exc_info=True)
    except FileNotFoundError as fnfe:
        logger.error(f"File/Directory not found during ingestion: {fnfe}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred during the ingestion pipeline: {e}", exc_info=True)

# Example Usage (for testing purposes)
if __name__ == '__main__':
    import os
    import shutil
    
    if not app_config.embedding.api_key:
        print("Error: OPENAI_API_KEY not found in environment variables. Please set it in .env")
        exit(1)
        
    TEST_SOURCE_DIR = "./temp_ingest_data"
    
    CHROMA_TEST_PERSIST_DIR = "./temp_chroma_store_pipeline"
    CHROMA_TEST_COLLECTION = "ingest_test_collection_pipeline_chroma"
    MILVUS_TEST_COLLECTION = "ingest_test_pipeline_milvus"
    WEAVIATE_TEST_INDEX = "IngestTestPipelineWeaviate"
    QDRANT_TEST_COLLECTION = "ingest_test_pipeline_qdrant"

    os.makedirs(TEST_SOURCE_DIR, exist_ok=True)
    with open(os.path.join(TEST_SOURCE_DIR, "ingest_doc1.txt"), "w") as f:
        f.write("This is the first document for ingestion testing. LangChain is useful.")
    with open(os.path.join(TEST_SOURCE_DIR, "ingest_doc2.md"), "w") as f:
        f.write("# Ingestion Test\nThis is the second document. RAG enhances LLMs.")

    original_vs_type = app_config.vector_store_type
    original_chroma_persist_dir = app_config.chromadb.persist_directory
    original_chroma_collection = app_config.chromadb.collection_name
    original_milvus_collection = app_config.milvus.collection_name
    original_weaviate_index = app_config.weaviate.index_name
    original_qdrant_collection = app_config.qdrant.collection_name

    # --- Test with Chroma ----
    test_vs_type = "chroma"
    print(f"--- Testing Ingestion & Query Pipeline with {test_vs_type.upper()} ---")
    app_config.vector_store_type = test_vs_type
    # CLI args would override these if provided
    test_persist_dir_param = CHROMA_TEST_PERSIST_DIR
    test_collection_param = CHROMA_TEST_COLLECTION 
    if os.path.exists(test_persist_dir_param):
        shutil.rmtree(test_persist_dir_param)
    
    run_ingestion_pipeline(source_path=TEST_SOURCE_DIR, persist_directory=test_persist_dir_param, collection_name=test_collection_param)
    query_pipeline_chroma = get_baseline_rag_pipeline(force_reload_retriever=True, persist_directory_override=test_persist_dir_param, collection_name_override=test_collection_param)
    if query_pipeline_chroma:
        response = query_pipeline_chroma.invoke("What is LangChain?")
        print(f"Chroma Response: {response}")
    else:
        print("Chroma query pipeline creation failed.")
    if os.path.exists(test_persist_dir_param):
         shutil.rmtree(test_persist_dir_param)

    # Restore original config before next test or exit
    app_config.vector_store_type = original_vs_type
    app_config.chromadb.persist_directory = original_chroma_persist_dir
    app_config.chromadb.collection_name = original_chroma_collection
    app_config.milvus.collection_name = original_milvus_collection
    app_config.weaviate.index_name = original_weaviate_index
    app_config.qdrant.collection_name = original_qdrant_collection
        
    shutil.rmtree(TEST_SOURCE_DIR)
    print("Pipeline example script finished.") 