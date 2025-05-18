import os
from typing import List, Dict, Type
from urllib.parse import urlparse

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader,
    UnstructuredFileLoader
)
from langchain_core.documents import Document

from .config import logger

# Mapping from file extensions to their respective loaders and keyword arguments
DEFAULT_LOADER_MAPPING: Dict[str, Type[UnstructuredFileLoader]] = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader, # Using UnstructuredMarkdownLoader for better markdown parsing
}

def _get_loader(file_path: str) -> UnstructuredFileLoader:
    """Determines the appropriate loader based on the file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    loader_class = DEFAULT_LOADER_MAPPING.get(ext)
    if loader_class:
        # For PyPDFLoader and TextLoader, they expect a file_path string
        # UnstructuredMarkdownLoader also takes file_path
        return loader_class(file_path) 
    logger.warning(f"No specific loader for extension '{ext}', using generic TextLoader for {file_path}.")
    return TextLoader(file_path)

def load_documents(source_path: str) -> List[Document]:
    """Loads documents from a source path (directory, file, or URL)."""
    logger.info(f"--- Inside load_documents --- CWD: {os.getcwd()}, source_path: {source_path}") # DEBUG
    documents: List[Document] = []
    # Convert to absolute path early to resolve relative paths correctly
    abs_source_path = os.path.abspath(source_path)
    logger.info(f"Attempting to load documents from resolved source: {abs_source_path} (original: {source_path})")
    logger.info(f"Checking file: {abs_source_path}, IsFile: {os.path.isfile(abs_source_path)}, IsDir: {os.path.isdir(abs_source_path)}") # DEBUG

    parsed_url = urlparse(source_path) # Check original source_path for URL scheme
    if parsed_url.scheme in ["http", "https"]:
        try:
            logger.info(f"Loading documents from URL: {source_path}")
            loader = WebBaseLoader([source_path]) # WebBaseLoader expects a list of URLs
            documents = loader.load()
            logger.info(f"Successfully loaded {len(documents)} document(s) from URL {source_path}.")
        except Exception as e:
            logger.error(f"Error loading documents from URL {source_path}: {e}", exc_info=True)
        return documents

    if os.path.isdir(abs_source_path):
        logger.info(f"Loading documents from directory: {abs_source_path}")
        loaded_docs_count = 0
        for root, _, files in os.walk(abs_source_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    loader = _get_loader(file_path)
                    docs_from_file = loader.load()
                    if docs_from_file:
                        for doc in docs_from_file: 
                            doc.metadata["source"] = file_path # Use absolute file_path for source
                        documents.extend(docs_from_file)
                        loaded_docs_count += len(docs_from_file)
                        logger.info(f"Loaded {len(docs_from_file)} document(s) from file: {file_path}")
                except Exception as e:
                    logger.error(f"Error loading file {file_path}: {e}", exc_info=True)
        logger.info(f"Successfully loaded {loaded_docs_count} total document(s) from directory {abs_source_path}.")
        return documents

    if os.path.isfile(abs_source_path):
        logger.info(f"Loading document from single file: {abs_source_path}")
        try:
            loader = _get_loader(abs_source_path)
            documents = loader.load()
            if documents:
                 for doc in documents: 
                    doc.metadata["source"] = abs_source_path # Use absolute path
            logger.info(f"Successfully loaded {len(documents)} document(s) from file {abs_source_path}.")
        except Exception as e:
            logger.error(f"Error loading document from file {abs_source_path}: {e}", exc_info=True)
        return documents

    logger.warning(f"Source path '{source_path}' (resolved to '{abs_source_path}') is not a valid URL, directory, or file. Returning empty list.")
    return []

# Example Usage (for testing purposes, can be removed or moved to a test file)
if __name__ == '__main__':
    # Create dummy files for testing
    os.makedirs("temp_data/docs", exist_ok=True)
    with open("temp_data/docs/sample.txt", "w") as f:
        f.write("This is a sample text file.")
    with open("temp_data/docs/sample.md", "w") as f:
        f.write("# Sample Markdown\n\nThis is a sample markdown file.")
    # PDFs and Web URLs require actual files/URLs
    
    logger.info("--- Testing Directory Loader ---")
    dir_docs = load_documents("temp_data/docs")
    for doc in dir_docs:
        logger.info(f"Loaded: {doc.page_content[:50]}... Source: {doc.metadata.get('source')}")

    logger.info("--- Testing Single File Loader (txt) ---")
    txt_docs = load_documents("temp_data/docs/sample.txt")
    for doc in txt_docs:
        logger.info(f"Loaded: {doc.page_content[:50]}... Source: {doc.metadata.get('source')}")

    # logger.info("--- Testing Web Loader ---")
    # web_docs = load_documents("https://lilianweng.github.io/posts/2023-06-23-agent/") # Example URL
    # if web_docs:
    #     logger.info(f"Loaded {len(web_docs)} documents from web.")
    #     logger.info(f"Content from web: {web_docs[0].page_content[:100]}...")
    # else:
    #     logger.info("Failed to load from web or URL is invalid.")
    
    # Clean up dummy files
    os.remove("temp_data/docs/sample.txt")
    os.remove("temp_data/docs/sample.md")
    os.rmdir("temp_data/docs")
    os.rmdir("temp_data") 