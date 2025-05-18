from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .config import logger

# --- Default Splitter Configuration --- 
# These values can be adjusted based on the embedding model and use case.
# For models like text-embedding-ada-002, smaller chunk sizes (~512-1024 tokens) often work well.
DEFAULT_CHUNK_SIZE = 1000  # Number of characters per chunk
DEFAULT_CHUNK_OVERLAP = 200 # Number of characters to overlap between chunks

def split_documents(documents: List[Document],
                    chunk_size: int = DEFAULT_CHUNK_SIZE,
                    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[Document]:
    """Splits loaded documents into smaller chunks using RecursiveCharacterTextSplitter."""
    if not documents:
        logger.warning("Received empty list of documents to split. Returning empty list.")
        return []
    
    logger.info(f"Splitting {len(documents)} document(s) into chunks (size={chunk_size}, overlap={chunk_overlap})...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len, # Use character count for length
        add_start_index=True, # Add start index metadata to chunks
    )
    
    split_docs = text_splitter.split_documents(documents)
    
    logger.info(f"Successfully split {len(documents)} document(s) into {len(split_docs)} chunks.")
    
    # Log a sample of the split
    if split_docs:
        logger.debug(f"Sample chunk 1 metadata: {split_docs[0].metadata}")
        logger.debug(f"Sample chunk 1 content: {split_docs[0].page_content[:100]}...")
        if len(split_docs) > 1:
             logger.debug(f"Sample chunk 2 metadata: {split_docs[1].metadata}")
             logger.debug(f"Sample chunk 2 content: {split_docs[1].page_content[:100]}...")
             
    return split_docs

# Example Usage (for testing purposes)
if __name__ == '__main__':
    from langchain_core.documents import Document

    # Create dummy documents
    doc1 = Document(page_content="This is the first document. It has multiple sentences. We want to split it.", metadata={"source": "doc1.txt"})
    doc2 = Document(page_content="""This is the second document.
It contains multiple lines.
Newlines should be handled properly by the splitter.
""" * 50, metadata={"source": "doc2.txt"}) # Make it longer to ensure splitting

    logger.info("--- Testing Document Splitter ---")
    original_docs = [doc1, doc2]
    split_docs_result = split_documents(original_docs, chunk_size=100, chunk_overlap=20)

    logger.info(f"Original document count: {len(original_docs)}")
    logger.info(f"Split document count: {len(split_docs_result)}")

    for i, doc in enumerate(split_docs_result):
        logger.info(f"Chunk {i+1}: Source={doc.metadata.get('source')}, Start Index={doc.metadata.get('start_index')}, Content='{doc.page_content[:80]}...'") 