import pytest
import os
from unittest.mock import patch, MagicMock

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)

from src.rag_lang.core.loader import _get_loader, load_documents, DEFAULT_LOADER_MAPPING
from src.rag_lang.core.config import logger as loader_logger # Import the logger used by loader

# --- Tests for _get_loader --- 

@pytest.mark.parametrize(
    "file_path, ext, expected_instance_type",
    [
        ("test.pdf", ".pdf", PyPDFLoader),
        ("test.txt", ".txt", TextLoader),
        ("test.md", ".md", UnstructuredMarkdownLoader),
        ("archive.PDF", ".pdf", PyPDFLoader),
        ("notes.TXT", ".txt", TextLoader),
    ]
)
def test_get_loader_known_extensions(mocker, file_path, ext, expected_instance_type):
    """Tests _get_loader with known file extensions, mocking DEFAULT_LOADER_MAPPING."""
    mock_loader_instance = MagicMock(spec=expected_instance_type)
    mock_loader_instance.file_path = file_path
    
    # Mock the loader class that would be retrieved from the mapping
    MockSpecificLoaderClass = MagicMock(return_value=mock_loader_instance) 

    # Patch the DEFAULT_LOADER_MAPPING for the scope of this test function
    mocker.patch.dict(DEFAULT_LOADER_MAPPING, {ext: MockSpecificLoaderClass}, clear=False)
    
    loader_result = _get_loader(file_path)
    
    # Assert that the mocked loader class from the mapping was instantiated
    MockSpecificLoaderClass.assert_called_once_with(file_path)
    # Assert that the function returned the instance created by the mocked class
    assert loader_result is mock_loader_instance
    assert isinstance(loader_result, expected_instance_type)
    assert loader_result.file_path == file_path

@patch.object(loader_logger, 'warning') # Mock the logger used in _get_loader
@patch('src.rag_lang.core.loader.TextLoader') # Also mock TextLoader for the unknown case
def test_get_loader_unknown_extension(MockTextLoader, mock_warning_logger):
    """Tests _get_loader with an unknown file extension."""
    file_path = "document.unknownext"
    
    mock_text_loader_instance = MagicMock(spec=TextLoader)
    mock_text_loader_instance.file_path = file_path
    MockTextLoader.return_value = mock_text_loader_instance
    
    loader = _get_loader(file_path)

    MockTextLoader.assert_called_once_with(file_path) # Should default to TextLoader
    assert loader is mock_text_loader_instance
    assert loader.file_path == file_path
    mock_warning_logger.assert_called_once_with(
        f"No specific loader for extension '.unknownext', using generic TextLoader for {file_path}."
    )

@patch('src.rag_lang.core.loader.TextLoader') # Mock TextLoader for no extension case
def test_get_loader_no_extension(MockTextLoader):
    """Tests _get_loader with a file that has no extension."""
    file_path = "file_without_extension"

    mock_text_loader_instance = MagicMock(spec=TextLoader)
    mock_text_loader_instance.file_path = file_path
    MockTextLoader.return_value = mock_text_loader_instance

    loader = _get_loader(file_path)

    MockTextLoader.assert_called_once_with(file_path)
    assert loader is mock_text_loader_instance
    assert loader.file_path == file_path
    # We are not checking the warning here as it is covered by test_get_loader_unknown_extension indirectly.
    # The main goal is that it correctly uses TextLoader.

# --- Tests for load_documents --- 

# Helper to create a dummy file with content
def create_dummy_file(path, content="Test content"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    return path

@patch('src.rag_lang.core.loader._get_loader') # Mock _get_loader to control loader behavior
def test_load_documents_single_file_success(mock_get_loader, tmp_path):
    """Tests load_documents with a single existing file that loads successfully."""
    file_content = "This is a test document."
    file_path_obj = tmp_path / "test_doc.txt"
    file_path_str = str(create_dummy_file(file_path_obj, file_content))

    # Mock the loader returned by _get_loader
    mock_loader_instance = MagicMock()
    # Simulate the loader's load() method returning a list of Document objects
    mock_document = MagicMock(spec=Document) # Use spec for Document if available or mock attributes
    mock_document.page_content = file_content
    mock_document.metadata = {} # Start with empty metadata
    mock_loader_instance.load.return_value = [mock_document]
    mock_get_loader.return_value = mock_loader_instance

    documents = load_documents(file_path_str)

    mock_get_loader.assert_called_once_with(os.path.abspath(file_path_str))
    mock_loader_instance.load.assert_called_once()
    assert len(documents) == 1
    assert documents[0].page_content == file_content
    assert documents[0].metadata["source"] == os.path.abspath(file_path_str)

@patch('src.rag_lang.core.loader._get_loader')
@patch.object(loader_logger, 'error')
def test_load_documents_single_file_load_error(mock_error_logger, mock_get_loader, tmp_path):
    """Tests load_documents when a single file loader fails to load."""
    file_path_obj = tmp_path / "error_doc.txt"
    file_path_str = str(create_dummy_file(file_path_obj, "error content"))

    mock_loader_instance = MagicMock()
    mock_loader_instance.load.side_effect = Exception("Failed to load file")
    mock_get_loader.return_value = mock_loader_instance

    documents = load_documents(file_path_str)

    assert len(documents) == 0
    mock_error_logger.assert_called_once()
    # Check that the error message contains the file path and the exception message
    args, _ = mock_error_logger.call_args
    assert f"Error loading document from file {os.path.abspath(file_path_str)}" in args[0]
    assert "Failed to load file" in args[0]

def test_load_documents_file_not_exists(tmp_path):
    """Tests load_documents with a file path that does not exist."""
    non_existent_file = str(tmp_path / "not_found.txt")
    documents = load_documents(non_existent_file)
    assert len(documents) == 0
    # Optionally, check for a warning log, though current implementation logs this as "Source path ... is not a valid URL, directory, or file."

# To use Document in mock_document spec
from langchain_core.documents import Document

@patch('src.rag_lang.core.loader._get_loader')
def test_load_documents_directory_success(mock_get_loader, tmp_path):
    """Tests load_documents with a directory containing multiple files and subdirectories."""
    # Create a directory structure
    # tmp_path/dir1/file1.txt
    # tmp_path/dir1/file2.md
    # tmp_path/dir1/subdir/file3.txt

    dir1 = tmp_path / "dir1"
    subdir = dir1 / "subdir"
    os.makedirs(subdir)

    file1_content = "Content of file1.txt"
    file2_content = "# Content of file2.md"
    file3_content = "Content of file3.txt"

    file1_path_str = str(create_dummy_file(dir1 / "file1.txt", file1_content))
    file2_path_str = str(create_dummy_file(dir1 / "file2.md", file2_content))
    file3_path_str = str(create_dummy_file(subdir / "file3.txt", file3_content))
    
    # Store absolute paths for assertion
    abs_file1_path = os.path.abspath(file1_path_str)
    abs_file2_path = os.path.abspath(file2_path_str)
    abs_file3_path = os.path.abspath(file3_path_str)

    # Mock _get_loader to return loaders that produce specific documents
    mock_loader1 = MagicMock()
    doc1 = Document(page_content=file1_content, metadata={})
    mock_loader1.load.return_value = [doc1]

    mock_loader2 = MagicMock()
    doc2 = Document(page_content=file2_content, metadata={})
    mock_loader2.load.return_value = [doc2]

    mock_loader3 = MagicMock()
    doc3 = Document(page_content=file3_content, metadata={})
    mock_loader3.load.return_value = [doc3]

    # Define the side_effect for _get_loader based on the path it receives
    def get_loader_side_effect(path):
        abs_path = os.path.abspath(path)
        if abs_path == abs_file1_path:
            return mock_loader1
        elif abs_path == abs_file2_path:
            return mock_loader2
        elif abs_path == abs_file3_path:
            return mock_loader3
        return MagicMock() # Default mock for any other files (e.g. .DS_Store)
    
    mock_get_loader.side_effect = get_loader_side_effect

    documents = load_documents(str(dir1))

    assert len(documents) == 3
    # Check if all documents were loaded and metadata updated
    loaded_sources = sorted([doc.metadata["source"] for doc in documents])
    expected_sources = sorted([abs_file1_path, abs_file2_path, abs_file3_path])
    assert loaded_sources == expected_sources

    # Verify content as well, ensuring correct doc was associated with correct source from mock
    content_source_map = {doc.metadata["source"]: doc.page_content for doc in documents}
    assert content_source_map[abs_file1_path] == file1_content
    assert content_source_map[abs_file2_path] == file2_content
    assert content_source_map[abs_file3_path] == file3_content
    
    assert mock_get_loader.call_count >= 3 # May be called for other files like .DS_Store
    mock_loader1.load.assert_called_once()
    mock_loader2.load.assert_called_once()
    mock_loader3.load.assert_called_once()


@patch.object(loader_logger, 'error')
@patch('src.rag_lang.core.loader._get_loader')
def test_load_documents_directory_with_file_error(mock_get_loader, mock_error_logger, tmp_path):
    """Tests directory loading when one file fails to load but others succeed."""
    dir_path = tmp_path / "err_dir"
    os.makedirs(dir_path)

    good_file_content = "Good content"
    good_file_path = str(create_dummy_file(dir_path / "good.txt", good_file_content))
    bad_file_path = str(create_dummy_file(dir_path / "bad.txt", "Bad content"))
    abs_good_file_path = os.path.abspath(good_file_path)
    abs_bad_file_path = os.path.abspath(bad_file_path)

    mock_good_loader = MagicMock()
    good_doc = Document(page_content=good_file_content, metadata={})
    mock_good_loader.load.return_value = [good_doc]

    mock_bad_loader = MagicMock()
    mock_bad_loader.load.side_effect = Exception("Failed to load bad file")

    def get_loader_side_effect(path):
        abs_path = os.path.abspath(path)
        if abs_path == abs_good_file_path:
            return mock_good_loader
        elif abs_path == abs_bad_file_path:
            return mock_bad_loader
        return MagicMock()
    mock_get_loader.side_effect = get_loader_side_effect

    documents = load_documents(str(dir_path))

    assert len(documents) == 1
    assert documents[0].page_content == good_file_content
    assert documents[0].metadata["source"] == abs_good_file_path
    
    mock_error_logger.assert_called_once()
    args, _ = mock_error_logger.call_args
    assert f"Error loading file {abs_bad_file_path}" in args[0]
    assert "Failed to load bad file" in args[0]

def test_load_documents_empty_directory(tmp_path):
    """Tests load_documents with an empty directory."""
    empty_dir = tmp_path / "empty_dir"
    os.makedirs(empty_dir)
    documents = load_documents(str(empty_dir))
    assert len(documents) == 0

def test_load_documents_directory_not_exists(tmp_path):
    """Tests load_documents with a directory path that does not exist."""
    non_existent_dir = str(tmp_path / "non_existent_dir")
    documents = load_documents(non_existent_dir)
    assert len(documents) == 0

@patch('src.rag_lang.core.loader.WebBaseLoader') # Mock WebBaseLoader directly
@patch.object(loader_logger, 'info') # To check log messages
def test_load_documents_url_success(mock_info_logger, MockWebBaseLoader):
    """Tests load_documents with a URL that loads successfully."""
    test_url = "http://example.com/testpage"
    mock_web_loader_instance = MagicMock()
    mock_document = Document(page_content="Web page content", metadata={"source": test_url})
    mock_web_loader_instance.load.return_value = [mock_document]
    MockWebBaseLoader.return_value = mock_web_loader_instance

    documents = load_documents(test_url)

    MockWebBaseLoader.assert_called_once_with([test_url])
    mock_web_loader_instance.load.assert_called_once()
    assert len(documents) == 1
    assert documents[0].page_content == "Web page content"
    assert documents[0].metadata["source"] == test_url
    # Check for specific log message (optional, but good for confirming flow)
    # Example: mock_info_logger.assert_any_call(f"Successfully loaded 1 document(s) from URL {test_url}.")

@patch('src.rag_lang.core.loader.WebBaseLoader')
@patch.object(loader_logger, 'error')
def test_load_documents_url_load_error(mock_error_logger, MockWebBaseLoader):
    """Tests load_documents when a URL loader fails."""
    test_url = "http://example.com/badpage"
    mock_web_loader_instance = MagicMock()
    mock_web_loader_instance.load.side_effect = Exception("Failed to load URL")
    MockWebBaseLoader.return_value = mock_web_loader_instance

    documents = load_documents(test_url)

    assert len(documents) == 0
    mock_error_logger.assert_called_once()
    args, _ = mock_error_logger.call_args
    assert f"Error loading documents from URL {test_url}" in args[0]
    assert "Failed to load URL" in args[0]

@patch.object(loader_logger, 'warning')
def test_load_documents_invalid_path(mock_warning_logger, tmp_path):
    """Tests load_documents with a source path that is not a valid file, directory, or URL."""
    # This path should not exist and not be a URL
    invalid_path = "this_is_not_a_valid_path_or_url"
    # To ensure it doesn't accidentally become a relative file, make it part of tmp_path structure if needed
    # but for this test, a clearly non-file string should suffice if os.path.abspath doesn't make it valid.
    # Let's use a path that would be inside tmp_path but we ensure doesn't exist.
    invalid_path_in_tmp = str(tmp_path / "completely_invalid_source_path")

    documents = load_documents(invalid_path_in_tmp)
    assert len(documents) == 0
    
    # The logger warning uses the original path and the resolved absolute path
    abs_invalid_path = os.path.abspath(invalid_path_in_tmp)
    expected_log_message = f"Source path '{invalid_path_in_tmp}' (resolved to '{abs_invalid_path}') is not a valid URL, directory, or file. Returning empty list."
    mock_warning_logger.assert_any_call(expected_log_message) 