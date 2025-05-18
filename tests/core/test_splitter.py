import pytest
from langchain_core.documents import Document
from unittest.mock import patch, MagicMock

from src.rag_lang.core.splitter import split_documents, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from src.rag_lang.core.config import logger # Import logger to mock it

# Basic test document
SAMPLE_DOC_CONTENT = "This is a test document. It has several sentences. We want to split this text effectively."
SAMPLE_DOC_METADATA = {"source": "sample.txt"}
SAMPLE_DOC = Document(page_content=SAMPLE_DOC_CONTENT, metadata=SAMPLE_DOC_METADATA)

def test_split_documents_basic():
    """Test basic splitting of a single document."""
    docs = [SAMPLE_DOC]
    
    # Make the chunk size smaller than the document to ensure splitting
    chunk_size = 30
    chunk_overlap = 5
    
    split_docs = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    print("\nDEBUG: Basic Split Docs Info")
    for idx, d_chunk in enumerate(split_docs):
        print(f"  Chunk {idx}: start={d_chunk.metadata.get('start_index')}, len={len(d_chunk.page_content)}, content=\'{d_chunk.page_content}'")

    assert len(split_docs) > 1, "Document should have been split into multiple chunks."
    
    total_content_length = 0
    previous_end_index = -1
    
    for i, chunk in enumerate(split_docs):
        assert chunk.metadata["source"] == SAMPLE_DOC_METADATA["source"], "Source metadata should be preserved."
        assert "start_index" in chunk.metadata, "start_index metadata should be added."
        
        # Verify that the chunk's content matches the original document at its start_index
        current_chunk_start = chunk.metadata['start_index']
        current_chunk_len = len(chunk.page_content)
        assert chunk.page_content == SAMPLE_DOC_CONTENT[current_chunk_start : current_chunk_start + current_chunk_len], (
            f"Chunk {i} content '{chunk.page_content[:20]}...' does not match original document slice "
            f"'{SAMPLE_DOC_CONTENT[current_chunk_start : current_chunk_start + current_chunk_len][:20]}...'"
        )

        if i > 0:
            prev_chunk_obj = split_docs[i-1]
            prev_chunk_content = prev_chunk_obj.page_content
            prev_chunk_start_index = prev_chunk_obj.metadata['start_index']
            
            # 1. Current chunk must start before the previous one ends to have overlap
            assert current_chunk_start < (prev_chunk_start_index + len(prev_chunk_content)), (
                f"Chunk {i} (start {current_chunk_start}) should start before previous chunk (idx {i-1}) "
                f"ends (at {prev_chunk_start_index + len(prev_chunk_content)})"
            )
            
            # 2. Calculate actual overlap length and verify content
            overlap_len = (prev_chunk_start_index + len(prev_chunk_content)) - current_chunk_start
            assert overlap_len > 0, (
                f"Chunks {i-1} and {i} should have a positive overlap. "
                f"Prev end: {prev_chunk_start_index + len(prev_chunk_content)}, Curr start: {current_chunk_start}"
            )
            
            original_overlap_content = SAMPLE_DOC_CONTENT[current_chunk_start : current_chunk_start + overlap_len]
            prev_chunk_overlap_part = prev_chunk_content[len(prev_chunk_content) - overlap_len:]
            curr_chunk_overlap_part = chunk.page_content[:overlap_len]
            
            assert original_overlap_content == prev_chunk_overlap_part, (
                f"Overlap content mismatch for chunk {i-1} end. "
                f"Original: '{original_overlap_content}', PrevChunkPart: '{prev_chunk_overlap_part}'"
            )
            assert original_overlap_content == curr_chunk_overlap_part, (
                f"Overlap content mismatch for chunk {i} start. "
                f"Original: '{original_overlap_content}', CurrChunkPart: '{curr_chunk_overlap_part}'"
            )

        # Verify chunk size (it might be slightly smaller for the last chunk or due to separators)
        assert len(chunk.page_content) <= chunk_size, f"Chunk content length {len(chunk.page_content)} exceeds chunk_size {chunk_size}."
        
        total_content_length += len(chunk.page_content) # This will be > original due to overlap
        
        # Check if the chunk content is part of the original document
        assert chunk.page_content in SAMPLE_DOC_CONTENT, f"Chunk content '{chunk.page_content}' not found in original."

    # Check if the split content, when joined (ignoring overlap for simplicity of this check), resembles the original
    # This isn't perfect due to overlap but gives a rough idea.
    # A better check would be to reconstruct the original string from chunks.
    reconstructed_approx = "".join([c.page_content[:chunk_size-chunk_overlap if i < len(split_docs)-1 else chunk_size] for i,c in enumerate(split_docs)])
    
    # We can't directly compare reconstructed_approx with SAMPLE_DOC_CONTENT due to overlap.
    # We'll check that unique parts of chunks are present.
    unique_parts = set()
    for chunk in split_docs:
        # Attempt to get a part of the chunk that is less likely to be pure overlap
        start = chunk_overlap // 2
        end = len(chunk.page_content) - (chunk_overlap // 2)
        if start < end:
            unique_parts.add(chunk.page_content[start:end])

    for part in unique_parts:
        if part: # ensure part is not empty
             assert part in SAMPLE_DOC_CONTENT, f"Unique part '{part}' not in original content"

    assert split_docs[0].metadata["start_index"] == 0, "First chunk should start at index 0."


def test_split_documents_empty_list():
    """Test splitting an empty list of documents."""
    with patch.object(logger, 'warning') as mock_warning:
        split_docs = split_documents([])
        assert split_docs == [], "Should return an empty list for empty input."
        mock_warning.assert_called_once_with("Received empty list of documents to split. Returning empty list.")


def test_split_documents_no_split_needed():
    """Test splitting a document shorter than chunk_size."""
    short_content = "Short content."
    doc = Document(page_content=short_content, metadata={"source": "short.txt"})
    docs = [doc]
    
    # Use default chunk_size, which should be larger than short_content
    split_docs = split_documents(docs)
    
    assert len(split_docs) == 1, "Document should not have been split."
    assert split_docs[0].page_content == short_content
    assert split_docs[0].metadata["source"] == "short.txt"
    assert split_docs[0].metadata["start_index"] == 0


def test_split_documents_multiple_docs():
    """Test splitting multiple documents."""
    doc1_content = "This is the first document, it is fairly long and should be split into several pieces." * 3
    doc2_content = "This is the second document, also quite long, ensuring it gets split too." * 3
    doc1 = Document(page_content=doc1_content, metadata={"source": "doc1.txt"})
    doc2 = Document(page_content=doc2_content, metadata={"source": "doc2.txt"})
    docs = [doc1, doc2]
    
    chunk_size = 50
    chunk_overlap = 10
    
    split_docs = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    assert len(split_docs) > 2, "Should have more chunks than original documents."
    
    doc1_chunks = [d for d in split_docs if d.metadata["source"] == "doc1.txt"]
    doc2_chunks = [d for d in split_docs if d.metadata["source"] == "doc2.txt"]
    
    assert len(doc1_chunks) > 1, "Doc1 should be split."
    assert len(doc2_chunks) > 1, "Doc2 should be split."
    
    # Verify content and metadata for doc1 chunks
    reconstructed_doc1 = ""
    current_pos_doc1 = 0
    for chunk in sorted(doc1_chunks, key=lambda d: d.metadata["start_index"]):
        assert chunk.metadata["start_index"] >= current_pos_doc1 - chunk_overlap
        assert chunk.page_content in doc1_content
        # This is a simplified reconstruction, real reconstruction needs careful handling of overlap
        # For testing, we check if the start of the chunk matches the original content at start_index
        assert doc1_content[chunk.metadata["start_index"] : chunk.metadata["start_index"]+len(chunk.page_content)] == chunk.page_content
        current_pos_doc1 = chunk.metadata["start_index"] + len(chunk.page_content)
        if reconstructed_doc1 == "": # First chunk
             reconstructed_doc1 = chunk.page_content
        else:
            # Approximate reconstruction by finding the non-overlapping part
            overlap_start_in_prev = reconstructed_doc1.rfind(chunk.page_content[:chunk_overlap])
            if overlap_start_in_prev != -1 and chunk.metadata["start_index"] > 0: # Simple heuristic
                reconstructed_doc1 += chunk.page_content[len(reconstructed_doc1) - (chunk.metadata["start_index"]):]
            else:
                 # Fallback if proper overlap point not found easily, might result in duplication for this test string
                reconstructed_doc1 += chunk.page_content 

    # A more robust check for content reconstruction:
    # The union of all chunk contents (considering their start_index) should cover the original document.
    # And each chunk should be a substring of the original.
    assert doc1_chunks[0].metadata["start_index"] == 0
    assert doc2_chunks[0].metadata["start_index"] == 0
    
    # Similar checks for doc2_chunks
    reconstructed_doc2 = ""
    current_pos_doc2 = 0
    for chunk in sorted(doc2_chunks, key=lambda d: d.metadata["start_index"]):
        assert chunk.metadata["start_index"] >= current_pos_doc2 - chunk_overlap
        assert chunk.page_content in doc2_content
        assert doc2_content[chunk.metadata["start_index"] : chunk.metadata["start_index"]+len(chunk.page_content)] == chunk.page_content
        current_pos_doc2 = chunk.metadata["start_index"] + len(chunk.page_content)


def test_split_documents_custom_params():
    """Test splitting with custom chunk_size and chunk_overlap using default values from module."""
    docs = [SAMPLE_DOC] # Use the global sample doc for this test
    
    # Test with default parameters first to ensure they are used if not provided
    default_split_docs = split_documents(docs)
    assert len(default_split_docs) >= 1 # Should be 1 if doc is shorter than DEFAULT_CHUNK_SIZE
    if len(SAMPLE_DOC_CONTENT) > DEFAULT_CHUNK_SIZE:
        assert len(default_split_docs) > 1
        for chunk in default_split_docs:
            assert len(chunk.page_content) <= DEFAULT_CHUNK_SIZE
            # Overlap is harder to assert directly without knowing the exact split points
    else:
        assert len(default_split_docs) == 1
        assert default_split_docs[0].page_content == SAMPLE_DOC_CONTENT

    # Test with specific custom parameters different from defaults
    custom_chunk_size = 25
    custom_chunk_overlap = 8
    assert custom_chunk_size != DEFAULT_CHUNK_SIZE
    assert custom_chunk_overlap != DEFAULT_CHUNK_OVERLAP
    
    custom_split_docs = split_documents(docs, chunk_size=custom_chunk_size, chunk_overlap=custom_chunk_overlap)

    print("\nDEBUG: Custom Split Docs Info")
    for idx, d_chunk in enumerate(custom_split_docs):
        print(f"  Chunk {idx}: start={d_chunk.metadata.get('start_index')}, len={len(d_chunk.page_content)}, overlap_cfg={custom_chunk_overlap}, content=\'{d_chunk.page_content}'")

    assert len(custom_split_docs) > 1, "Document should be split with custom small chunk_size."
    for chunk in custom_split_docs:
        assert len(chunk.page_content) <= custom_chunk_size, f"Chunk length {len(chunk.page_content)} exceeds custom_chunk_size {custom_chunk_size}"
        assert chunk.metadata["source"] == SAMPLE_DOC_METADATA["source"]
        assert "start_index" in chunk.metadata
    
    # Check properties of custom_split_docs
    if len(custom_split_docs) > 1:
        last_start_index = -1
        for i, chunk_obj in enumerate(custom_split_docs):
            current_start_index = chunk_obj.metadata['start_index']
            chunk_content = chunk_obj.page_content
            chunk_len = len(chunk_content)

            assert current_start_index > last_start_index, \
                f"Chunk {i} start_index {current_start_index} is not greater than last_start_index {last_start_index}"
            last_start_index = current_start_index

            assert current_start_index < len(SAMPLE_DOC_CONTENT), \
                f"Chunk {i} start_index {current_start_index} is out of bounds for doc length {len(SAMPLE_DOC_CONTENT)}"
            
            assert chunk_content == SAMPLE_DOC_CONTENT[current_start_index : current_start_index + chunk_len], \
                f"Chunk {i} content does not match original document slice."
            
            # We no longer make strong assertions about overlap amount for RecursiveCharacterTextSplitter
            # as it can prioritize separators, leading to variable or zero overlap.
            # The key is that parameters were passed and chunks are valid parts of the original.

    # Ensure the print statement for 0 overlap (if it occurred) is outside the loop or handled if needed for debugging.
    # The previous complex overlap logic has been removed from this test case for simplicity, 
    # relying on the basic test (test_split_documents_basic) for more detailed overlap content checking where feasible.

# Example Usage (for testing purposes)
# if __name__ == '__main__':
# ... (rest of the file remains the same) 