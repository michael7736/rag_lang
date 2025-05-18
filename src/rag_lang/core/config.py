import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Literal

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

def get_env_variable(var_name: str, default: Optional[str] = None) -> Optional[str]:
    """Gets an environment variable or returns a default."""
    value = os.environ.get(var_name, default)
    if value is None and default is None: # Only warn if no default is provided and it's not found
        logger.warning(f"Environment variable '{var_name}' not found and no default provided.")
    return value

@dataclass
class LLMConfig:
    """Configuration for the Language Model (Defaults to OpenRouter)."""
    api_key: Optional[str] = field(default_factory=lambda: get_env_variable("OPENROUTER_API_KEY"))
    api_base: Optional[str] = field(default_factory=lambda: get_env_variable("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"))
    model_name: str = "openai/gpt-4o" # Default model on OpenRouter

@dataclass
class EmbeddingConfig:
    """Configuration for the Embedding Model (Defaults to direct OpenAI)."""
    api_key: Optional[str] = field(default_factory=lambda: get_env_variable("OPENAI_API_KEY_EMBEDDING")) # Specific key for OpenAI embeddings
    # api_base for embeddings, if needed (e.g. for Azure OpenAI or proxy). Defaults to OpenAI's direct API base if None.
    api_base: Optional[str] = field(default_factory=lambda: get_env_variable("OPENAI_API_BASE_EMBEDDING")) 
    model_name: str = "text-embedding-ada-002"
    dimension: int = 1536 

@dataclass
class ChromaDBConfig:
    """Configuration for ChromaDB Vector Store."""
    persist_directory: str = "./chroma" # Default local directory for ChromaDB
    collection_name: str = "rag_lang_collection"

@dataclass
class MilvusConfig:
    """Configuration for Milvus Vector Store."""
    host: str = field(default_factory=lambda: get_env_variable("MILVUS_HOST", "rtx4080"))
    port: str = field(default_factory=lambda: get_env_variable("MILVUS_PORT", "19530"))
    collection_name: str = "rag_lang_milvus_collection"
    # Dimension needs to match the embedding model, e.g., 1536 for text-embedding-ada-002
    # We can get this from EmbeddingConfig later if needed dynamically
    index_params: Optional[dict] = None # Default Milvus index params can be used
    search_params: Optional[dict] = None # Default Milvus search params can be used
    # Example for HNSW index, adjust as needed
    # index_params: dict = field(default_factory=lambda: {
    #     "metric_type": "L2",
    #     "index_type": "HNSW",
    #     "params": {"M": 8, "efConstruction": 64},
    # })
    # search_params: dict = field(default_factory=lambda: {"ef": 64})

@dataclass
class QdrantConfig:
    """Configuration for Qdrant Vector Store."""
    host: str = field(default_factory=lambda: get_env_variable("QDRANT_HOST", "rtx4080"))
    port: int = field(default_factory=lambda: int(get_env_variable("QDRANT_PORT", "6333"))) # type: ignore
    grpc_port: int = field(default_factory=lambda: int(get_env_variable("QDRANT_GRPC_PORT", "6334"))) # type: ignore
    collection_name: str = "rag_lang_qdrant_collection"
    # Qdrant can also be run in-memory or with a local file using `path` or `:memory:` for host.
    # url: Optional[str] = field(default_factory=lambda: get_env_variable("QDRANT_URL")) # e.g. http://rtx4080:6333
    api_key: Optional[str] = field(default_factory=lambda: get_env_variable("QDRANT_API_KEY"))
    prefer_grpc: bool = True # LangChain Qdrant client can use gRPC

    def get_url(self) -> str:
        # if self.url:
        #     return self.url
        return f"http://{self.host}:{self.port}"

@dataclass
class AppConfig:
    """Main Application Configuration."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store_type: Literal["chroma", "milvus", "qdrant"] = field(
        default_factory=lambda: get_env_variable("VECTOR_STORE_TYPE", "chroma").lower() # type: ignore
    )
    chromadb: ChromaDBConfig = field(default_factory=ChromaDBConfig)
    milvus: MilvusConfig = field(default_factory=MilvusConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)

# Instantiate the config
app_config = AppConfig()

# Log loaded configuration details
logger.info(f"LLM Config: Model={app_config.llm.model_name}, API Base={app_config.llm.api_base}, Key Loaded: {bool(app_config.llm.api_key)}")
logger.info(f"Embedding Config: Model={app_config.embedding.model_name}, API Base={app_config.embedding.api_base or 'Default OpenAI'}, Key Loaded: {bool(app_config.embedding.api_key)}, Dimension={app_config.embedding.dimension}")
logger.info(f"Selected Vector Store Type: {app_config.vector_store_type}")
if app_config.vector_store_type == "chroma":
    logger.info(f"ChromaDB Config: Directory={app_config.chromadb.persist_directory}, Collection={app_config.chromadb.collection_name}")
elif app_config.vector_store_type == "milvus":
    logger.info(f"Milvus Config: Host={app_config.milvus.host}, Port={app_config.milvus.port}, Collection={app_config.milvus.collection_name}")
elif app_config.vector_store_type == "qdrant":
    cfg = app_config.qdrant
    logger.info(f"Qdrant Config: Host={cfg.host}, Port={cfg.port}, gRPC Port={cfg.grpc_port}, Collection={cfg.collection_name}")

if not app_config.llm.api_key:
    logger.error("API Key for LLM (OPENROUTER_API_KEY) not found. Please ensure it is set.")
if not app_config.embedding.api_key:
    logger.error("API Key for Embeddings (OPENAI_API_KEY_EMBEDDING) not found. Please ensure it is set.")
    # Consider raising an error if critical for startup
    # raise ValueError("API Key for LLM/Embeddings not configured.") 