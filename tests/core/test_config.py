import pytest
from unittest.mock import patch
import logging # Import the logging module
import os

# Assuming src.rag_lang.core.config defines get_env_variable and logger
from src.rag_lang.core.config import (
    LLMConfig, EmbeddingConfig, ChromaDBConfig, MilvusConfig, QdrantConfig, AppConfig,
    get_env_variable, logger as config_logger
)

# Test cases for get_env_variable
@pytest.mark.parametrize(
    "env_vars,var_name,default,expected_value,expect_warning",
    [
        ({"TEST_VAR": "value_from_env"}, "TEST_VAR", None, "value_from_env", False),
        ({}, "TEST_VAR", "default_value", "default_value", False),
        ({"TEST_VAR": "value_from_env"}, "TEST_VAR", "default_value", "value_from_env", False), # Env var takes precedence
        ({}, "NON_EXISTENT_VAR", None, None, True), # Not found, no default
        ({}, "NON_EXISTENT_VAR", "default_val", "default_val", False), # Not found, with default
    ],
)
def test_get_env_variable(mocker, env_vars, var_name, default, expected_value, expect_warning):
    """Tests the get_env_variable function."""
    mocker.patch.dict(os.environ, env_vars, clear=True)
    mock_warning = mocker.patch.object(config_logger, 'warning') # Use the imported config_logger

    result = get_env_variable(var_name, default)
    assert result == expected_value

    if expect_warning:
        mock_warning.assert_called_once_with(
            f"Environment variable '{var_name}' not found and no default provided."
        )
    else:
        mock_warning.assert_not_called()

def test_llm_config_defaults(mocker):
    """Tests LLMConfig with default values (no env vars set)."""
    mocker.patch.dict(os.environ, {}, clear=True)
    # Mock get_env_variable to control its behavior for this test
    # get_env_variable will be called for api_key and api_base
    mock_getenv = mocker.patch('src.rag_lang.core.config.get_env_variable')
    
    # Simulate get_env_variable returning None for OPENROUTER_API_KEY (as if not set)
    # and the default for OPENROUTER_API_BASE
    def side_effect(var_name, default=None):
        if var_name == "OPENROUTER_API_KEY":
            return None # No API key set
        if var_name == "OPENROUTER_API_BASE":
            return "https://openrouter.ai/api/v1" # Default base URL
        return default
    mock_getenv.side_effect = side_effect
    
    config = LLMConfig()
    
    assert config.api_key is None
    assert config.api_base == "https://openrouter.ai/api/v1"
    assert config.model_name == "openai/gpt-4o" # Default model

    # Check that get_env_variable was called for api_key and api_base
    expected_calls = [
        mocker.call("OPENROUTER_API_KEY"),
        mocker.call("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
    ]
    # The order of calls to get_env_variable by field(default_factory=...) is not guaranteed
    # So, we check for the calls in any order
    mock_getenv.assert_has_calls(expected_calls, any_order=True)

def test_llm_config_with_env_vars(mocker):
    """Tests LLMConfig when environment variables are set."""
    env_vars = {
        "OPENROUTER_API_KEY": "test_api_key_value",
        "OPENROUTER_API_BASE": "https://custom.api.base/v1",
    }
    mocker.patch.dict(os.environ, env_vars, clear=True)

    # We need to mock get_env_variable as it's directly used by the dataclass
    # field factories. If we don't mock it, it will use the actual os.environ,
    # but we want to isolate the test to the dataclass logic itself and how it *uses* get_env_variable.
    # This also makes the test more robust to changes in get_env_variable's internal logic
    # as long as its contract (fetching env vars) is maintained.
    mock_getenv = mocker.patch('src.rag_lang.core.config.get_env_variable')
    
    def side_effect(var_name, default=None):
        return env_vars.get(var_name, default)
    mock_getenv.side_effect = side_effect

    config = LLMConfig(model_name="custom/model")

    assert config.api_key == "test_api_key_value"
    assert config.api_base == "https://custom.api.base/v1"
    assert config.model_name == "custom/model"

    expected_calls = [
        mocker.call("OPENROUTER_API_KEY"),
        mocker.call("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
    ]
    mock_getenv.assert_has_calls(expected_calls, any_order=True)

def test_embedding_config_defaults(mocker):
    """Tests EmbeddingConfig with default values."""
    mocker.patch.dict(os.environ, {}, clear=True)
    mock_getenv = mocker.patch('src.rag_lang.core.config.get_env_variable')

    def side_effect(var_name, default=None):
        if var_name == "OPENAI_API_KEY_EMBEDDING":
            return None # No API key
        if var_name == "OPENAI_API_BASE_EMBEDDING":
            return None # No custom base
        return default
    mock_getenv.side_effect = side_effect

    config = EmbeddingConfig()

    assert config.api_key is None
    assert config.api_base is None
    assert config.model_name == "text-embedding-ada-002"
    assert config.dimension == 1536

    expected_calls = [
        mocker.call("OPENAI_API_KEY_EMBEDDING"),
        mocker.call("OPENAI_API_BASE_EMBEDDING")
    ]
    mock_getenv.assert_has_calls(expected_calls, any_order=True)

def test_embedding_config_with_env_vars(mocker):
    """Tests EmbeddingConfig with environment variables set."""
    env_vars = {
        "OPENAI_API_KEY_EMBEDDING": "test_embedding_key",
        "OPENAI_API_BASE_EMBEDDING": "https://custom.embedding.base/v1",
    }
    mocker.patch.dict(os.environ, env_vars, clear=True)
    mock_getenv = mocker.patch('src.rag_lang.core.config.get_env_variable')
    
    def side_effect(var_name, default=None):
        return env_vars.get(var_name, default)
    mock_getenv.side_effect = side_effect

    config = EmbeddingConfig(model_name="custom/embedding-model", dimension=1024)

    assert config.api_key == "test_embedding_key"
    assert config.api_base == "https://custom.embedding.base/v1"
    assert config.model_name == "custom/embedding-model"
    assert config.dimension == 1024

    expected_calls = [
        mocker.call("OPENAI_API_KEY_EMBEDDING"),
        mocker.call("OPENAI_API_BASE_EMBEDDING")
    ]
    mock_getenv.assert_has_calls(expected_calls, any_order=True)

def test_chromadb_config_defaults(mocker):
    """Tests ChromaDBConfig with default values."""
    # ChromaDBConfig does not directly use get_env_variable in its field factories
    # So, no need to mock get_env_variable for this specific test of ChromaDBConfig defaults
    mocker.patch.dict(os.environ, {}, clear=True) # Ensure no env vars interfere if any were planned for future
    
    config = ChromaDBConfig()
    
    assert config.persist_directory == "./chroma"
    assert config.collection_name == "rag_lang_collection"

def test_chromadb_config_custom_values():
    """Tests ChromaDBConfig with custom initialization values."""
    custom_dir = "/custom/path/chroma"
    custom_collection = "my_special_collection"
    
    config = ChromaDBConfig(
        persist_directory=custom_dir,
        collection_name=custom_collection
    )
    
    assert config.persist_directory == custom_dir
    assert config.collection_name == custom_collection

def test_milvus_config_defaults(mocker):
    """Tests MilvusConfig with default values (some from env)."""
    mocker.patch.dict(os.environ, {}, clear=True)
    mock_getenv = mocker.patch('src.rag_lang.core.config.get_env_variable')

    def side_effect(var_name, default=None):
        if var_name == "MILVUS_HOST":
            return "rtx4080" # Default from get_env_variable
        if var_name == "MILVUS_PORT":
            return "19530"   # Default from get_env_variable
        return default # For any other potential future env vars
    mock_getenv.side_effect = side_effect

    config = MilvusConfig()

    assert config.host == "rtx4080"
    assert config.port == "19530"
    assert config.collection_name == "rag_lang_milvus_collection"
    assert config.index_params is None
    assert config.search_params is None

    expected_calls = [
        mocker.call("MILVUS_HOST", "rtx4080"),
        mocker.call("MILVUS_PORT", "19530")
    ]
    mock_getenv.assert_has_calls(expected_calls, any_order=True)

def test_milvus_config_with_env_vars_and_custom(mocker):
    """Tests MilvusConfig with specific env vars and custom params."""
    env_vars = {
        "MILVUS_HOST": "custom_milvus_host",
        "MILVUS_PORT": "12345",
    }
    mocker.patch.dict(os.environ, env_vars, clear=True)
    mock_getenv = mocker.patch('src.rag_lang.core.config.get_env_variable')

    def side_effect(var_name, default=None):
        # Simulate get_env_variable behavior based on mocked env_vars
        return env_vars.get(var_name, default)
    mock_getenv.side_effect = side_effect

    custom_index_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
    custom_search_params = {"nprobe": 10}

    config = MilvusConfig(
        collection_name="my_milvus_db",
        index_params=custom_index_params,
        search_params=custom_search_params
    )

    assert config.host == "custom_milvus_host"
    assert config.port == "12345"
    assert config.collection_name == "my_milvus_db"
    assert config.index_params == custom_index_params
    assert config.search_params == custom_search_params

    # Ensure get_env_variable was called for host and port
    expected_calls = [
        mocker.call("MILVUS_HOST", "rtx4080"), # Default is passed to get_env_variable
        mocker.call("MILVUS_PORT", "19530")  # Default is passed to get_env_variable
    ]
    mock_getenv.assert_has_calls(expected_calls, any_order=True) 

def test_qdrant_config_defaults(mocker):
    """Tests QdrantConfig with default values (some from env)."""
    mocker.patch.dict(os.environ, {}, clear=True)
    mock_getenv = mocker.patch('src.rag_lang.core.config.get_env_variable')

    def side_effect(var_name, default=None):
        if var_name == "QDRANT_HOST":
            return "rtx4080"
        if var_name == "QDRANT_PORT":
            return "6333"
        if var_name == "QDRANT_GRPC_PORT":
            return "6334"
        if var_name == "QDRANT_API_KEY":
            return None # No API key by default
        # if var_name == "QDRANT_URL": # Not used by default factory in this test path
        #     return None
        return default
    mock_getenv.side_effect = side_effect

    config = QdrantConfig()

    assert config.host == "rtx4080"
    assert config.port == 6333
    assert config.grpc_port == 6334
    assert config.collection_name == "rag_lang_qdrant_collection"
    assert config.api_key is None
    assert config.prefer_grpc is True
    assert config.get_url() == "http://rtx4080:6333"

    expected_getenv_calls = [
        mocker.call("QDRANT_HOST", "rtx4080"),
        mocker.call("QDRANT_PORT", "6333"),
        mocker.call("QDRANT_GRPC_PORT", "6334"),
        mocker.call("QDRANT_API_KEY")
        # mocker.call("QDRANT_URL") # This is commented out in dataclass, so not called by factory
    ]
    mock_getenv.assert_has_calls(expected_getenv_calls, any_order=True)

def test_qdrant_config_with_env_vars_and_custom(mocker):
    """Tests QdrantConfig with specific env vars and custom values."""
    env_vars = {
        "QDRANT_HOST": "q.example.com",
        "QDRANT_PORT": "1234",
        "QDRANT_GRPC_PORT": "1235",
        "QDRANT_API_KEY": "test_q_key",
        # "QDRANT_URL": "http://my.qdrant.url:8080" # Test get_url without this first
    }
    mocker.patch.dict(os.environ, env_vars, clear=True)
    mock_getenv = mocker.patch('src.rag_lang.core.config.get_env_variable')

    def side_effect(var_name, default=None):
        return env_vars.get(var_name, default)
    mock_getenv.side_effect = side_effect

    config = QdrantConfig(
        collection_name="my_q_collection",
        prefer_grpc=False
    )

    assert config.host == "q.example.com"
    assert config.port == 1234
    assert config.grpc_port == 1235
    assert config.collection_name == "my_q_collection"
    assert config.api_key == "test_q_key"
    assert config.prefer_grpc is False
    assert config.get_url() == "http://q.example.com:1234"

# Commented out QDRANT_URL variant as it's not active in the main code's QdrantConfig
# def test_qdrant_config_with_url_env_var(mocker):
#     """Tests QdrantConfig when QDRANT_URL environment variable is set."""
#     env_vars = {
#         "QDRANT_URL": "http://url.from.env:7777"
#     }
#     mocker.patch.dict(os.environ, env_vars, clear=True)
#     mock_getenv = mocker.patch('src.rag_lang.core.config.get_env_variable')

#     def side_effect(var_name, default=None):
#         # Simulate get_env_variable: QDRANT_URL will be picked up
#         # For others, simulate them not being set, so dataclass defaults apply for host/port if URL wasn't used.
#         if var_name == "QDRANT_URL":
#             return env_vars.get("QDRANT_URL")
#         if var_name == "QDRANT_HOST":
#             return default # So dataclass default or direct init is used
#         if var_name == "QDRANT_PORT":
#             return default
#         if var_name == "QDRANT_GRPC_PORT":
#             return default
#         if var_name == "QDRANT_API_KEY":
#             return None
#         return default
#     mock_getenv.side_effect = side_effect

#     config = QdrantConfig() # Rely on QDRANT_URL from env via get_env_variable in factory

#     assert config.get_url() == "http://url.from.env:7777"
#     # Check that other get_env_variable calls were made for host/port/etc for completeness, 
#     # even if QDRANT_URL took precedence for get_url()
#     expected_getenv_calls = [
#         mocker.call("QDRANT_HOST", "rtx4080"),
#         mocker.call("QDRANT_PORT", "6333"),
#         mocker.call("QDRANT_GRPC_PORT", "6334"),
#         mocker.call("QDRANT_API_KEY"),
#         mocker.call("QDRANT_URL")
#     ]
#     mock_getenv.assert_has_calls(expected_getenv_calls, any_order=True)
#     # Verify that if URL is set, it is used, and host/port directly are for fallback if URL is None
#     # Given the current QdrantConfig.get_url() logic, if self.url is set, it returns self.url.
#     # The default_factory for `url` is `get_env_variable("QDRANT_URL")`
#     assert config.url == "http://url.from.env:7777" 

# Tests for AppConfig
def test_app_config_defaults(mocker):
    """Tests AppConfig with default values and correct child config instantiation."""
    mocker.patch.dict(os.environ, {}, clear=True)
    
    mock_getenv = mocker.patch('src.rag_lang.core.config.get_env_variable')
    
    # This side effect will be used by AppConfig for VECTOR_STORE_TYPE
    # and by child configs (LLMConfig, EmbeddingConfig, etc.) when they call get_env_variable
    def side_effect_for_configs(var_name, default=None):
        if var_name == "VECTOR_STORE_TYPE":
            return "chroma" # For AppConfig.vector_store_type
        # For other env vars called by child configs, return their specified default
        # or None if no default is given to get_env_variable by the child config.
        # This simulates the actual behavior of get_env_variable when env vars are not set.
        return default

    mock_getenv.side_effect = side_effect_for_configs

    # We are NOT mocking the child config classes themselves anymore.
    # We let AppConfig instantiate them normally.
    # The mock_getenv above will control how they get their env vars.

    appconfig = AppConfig()

    # Assert that get_env_variable was called for VECTOR_STORE_TYPE via the factory
    # and that the type is correctly set.
    mock_getenv.assert_any_call("VECTOR_STORE_TYPE", "chroma")
    assert appconfig.vector_store_type == "chroma"

    # Assert that child configs are instances of their respective classes
    assert isinstance(appconfig.llm, LLMConfig)
    assert isinstance(appconfig.embedding, EmbeddingConfig)
    assert isinstance(appconfig.chromadb, ChromaDBConfig)
    assert isinstance(appconfig.milvus, MilvusConfig)
    assert isinstance(appconfig.qdrant, QdrantConfig)

    # Further checks for default values of child configs can be added here if needed,
    # but they are more thoroughly tested in their own dedicated test functions.
    # For example, check one key default from each child config that relies on get_env_variable:
    assert appconfig.llm.api_key is None # Relies on get_env_variable("OPENROUTER_API_KEY") returning None (via default)
    assert appconfig.qdrant.host == "rtx4080" # Relies on get_env_variable("QDRANT_HOST", "rtx4080") returning "rtx4080"

@pytest.mark.parametrize(
    "env_value,expected_type",
    [
        ("milvus", "milvus"),
        ("QDRANT", "qdrant"), # Test case insensitivity
        ("CHROMA", "chroma"),
        ("invalid_store", "invalid_store") # Test fallback if not one of the Literals (though type hint would complain)
    ]
)
def test_app_config_vector_store_type_from_env(mocker, env_value, expected_type):
    """Tests AppConfig's vector_store_type initialization from environment variable."""
    mocker.patch.dict(os.environ, {"VECTOR_STORE_TYPE": env_value}, clear=True)
    
    # Mock get_env_variable to directly return the environment variable's value for this test
    mock_getenv = mocker.patch('src.rag_lang.core.config.get_env_variable')
    
    # The factory lambda calls get_env_variable("VECTOR_STORE_TYPE", "chroma").lower()
    # So we need to simulate this behavior for the mock
    def side_effect(var_name, default=None):
        if var_name == "VECTOR_STORE_TYPE":
            return env_value # Return the raw env value for .lower() to operate on
        return default
    mock_getenv.side_effect = side_effect
    
    # Mock child configs as we are only testing vector_store_type logic here
    mocker.patch('src.rag_lang.core.config.LLMConfig')
    mocker.patch('src.rag_lang.core.config.EmbeddingConfig')
    mocker.patch('src.rag_lang.core.config.ChromaDBConfig')
    mocker.patch('src.rag_lang.core.config.MilvusConfig')
    mocker.patch('src.rag_lang.core.config.QdrantConfig')

    appconfig = AppConfig()
    
    assert appconfig.vector_store_type == expected_type.lower() # .lower() is applied in the factory
    mock_getenv.assert_any_call("VECTOR_STORE_TYPE", "chroma")

# Commented out QDRANT_URL variant as it's not active in the main code's QdrantConfig
#     assert config.url == "http://url.from.env:7777" 