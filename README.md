# RAG & Knowledge Engine System Reference

A reference implementation for Retrieval-Augmented Generation (RAG) and agent-based knowledge engine systems using Python and LangChain.

Refer to `docs/PRD.md` and `docs/TECH_SPEC.md` (and their `_zh.md` counterparts) for project details.

## Setup

1.  **Create Conda Environment:**
    ```bash
    conda env create -f environment.yml
    conda activate rag_lang
    ```

2.  **Environment Variables:**
    Create a `.env` file in the project root and add your API keys:
    ```dotenv
    # Example for OpenRouter (OpenAI compatible)
    OPENROUTER_API_KEY="your-openrouter-api-key"
    OPENAI_API_KEY="your-openrouter-api-key" # Use the same key for OpenAI compatibility
    OPENAI_API_BASE="https://openrouter.ai/api/v1" # OpenRouter endpoint

    # Example for direct OpenAI
    # OPENAI_API_KEY="your-real-openai-api-key"
    ```
    *Make sure the `OPENAI_API_KEY` is set correctly for both embedding and LLM calls.*

3.  **Install package in editable mode (optional but recommended for development):**
    ```bash
    pip install -e .
    ```

## Usage

The primary interface is through the command-line tool.

### 1. Ingest Documents

Before querying, you need to ingest your documents into the vector store. The tool supports ingesting from a local directory, a single file, or a web URL.

```bash
# Ingest all supported files (.txt, .md, .pdf) from a directory
python -m rag_lang.cli ingest ./path/to/your/data

# Ingest a single file
python -m rag_lang.cli ingest ./path/to/your/document.pdf

# Ingest content from a URL
python -m rag_lang.cli ingest https://example.com/your-page.html

# Specify a different vector store location and collection name (optional)
python -m rag_lang.cli ingest ./data --persist-dir ./my_vector_store --collection my_documents 
```

This command will:
1. Load documents from the specified source.
2. Split them into chunks.
3. Generate embeddings using the configured model (default: OpenAI `text-embedding-ada-002`).
4. Store the chunks and embeddings in a ChromaDB vector store (default location: `./chroma`).

### 2. Query the System

Once documents are ingested, you can ask questions:

```bash
python -m rag_lang.cli query "What is the main topic of the ingested documents?"
```

Replace the question string with your actual query. The system will:
1. Load the vector store.
2. Create a retriever.
3. Create the RAG chain using the configured LLM (default: OpenRouter `openai/gpt-4o`).
4. Retrieve relevant document chunks based on your question.
5. Generate an answer based on the retrieved context.
6. Print the answer to the console.

## Development

*   Follow guidelines in `.cursor/rules/_global.mdc` and `.cursor/rules/_project.mdc`.
*   Run tests: `pytest` (Tests need to be implemented)
*   Run linters: `flake8 src tests`
*   Run type checker: `mypy src` 