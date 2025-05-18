from typing import List, Dict

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI # Specific import for LLM initialization
from langchain_core.retrievers import BaseRetriever

from .config import app_config, logger
from .retriever import get_retriever # To potentially get a default retriever in example

# --- LLM Initialization --- 
def get_llm() -> BaseChatModel:
    """Initializes and returns the LLM based on config."""
    if not app_config.llm.api_key:
        raise ValueError("API Key for LLM is not configured. Set OPENAI_API_KEY (or relevant key for your provider) in your environment.")
    
    logger.info(f"Initializing LLM with model: {app_config.llm.model_name} via API base: {app_config.llm.api_base}")
    # Using ChatOpenAI as it's compatible with OpenRouter and standard OpenAI endpoints
    return ChatOpenAI(
        model=app_config.llm.model_name,
        openai_api_key=app_config.llm.api_key,
        openai_api_base=app_config.llm.api_base,
        temperature=0.7, # Adjust temperature as needed
        streaming=False # Set to True for streaming responses if needed later
    )

# --- RAG Chain Creation --- 

DEFAULT_RAG_TEMPLATE = """Answer the question based only on the following context:

{context}

Question: {question}

Answer:"""

def format_docs(docs: List[Document]) -> str:
    """Formats a list of documents into a single string for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain(retriever: BaseRetriever, llm: BaseChatModel, prompt_template: str = DEFAULT_RAG_TEMPLATE) -> Runnable:
    """Creates the RAG chain using LCEL.

    Args:
        retriever: The retriever instance to fetch context.
        llm: The language model instance to generate the answer.
        prompt_template: The string template for the prompt.

    Returns:
        A LangChain Runnable representing the RAG chain.
    """
    logger.info("Creating RAG chain...")
    prompt = ChatPromptTemplate.from_template(prompt_template)

    rag_chain = (
        # RunnablePassthrough allows passing the original question through
        {"context": retriever | format_docs, "question": RunnablePassthrough()} 
        | prompt
        | llm
        | StrOutputParser() # Parses the LLM ChatMessage output to a string
    )
    logger.info("RAG chain created successfully.")
    return rag_chain

# Example Usage (for testing purposes)
if __name__ == '__main__':
    import os
    # Ensure API keys and vector store exist (run ingestion first)
    if not app_config.llm.api_key or not app_config.embedding.api_key:
        print("Error: API Key (OPENAI_API_KEY) not found. Please set it in .env")
        exit(1)
        
    if not os.path.exists(app_config.vector_store.persist_directory):
         print(f"Error: Vector store at '{app_config.vector_store.persist_directory}' not found. Run ingestion first.")
         exit(1)

    logger.info("--- Testing RAG Chain Creation and Invocation ---")
    try:
        # 1. Get default retriever
        logger.info("Getting retriever...")
        retriever_instance = get_retriever()
        
        # 2. Get default LLM
        logger.info("Getting LLM...")
        llm_instance = get_llm()
        
        # 3. Create RAG chain
        logger.info("Creating RAG chain...")
        rag_chain_instance = get_rag_chain(retriever_instance, llm_instance)
        logger.info(f"RAG Chain Type: {type(rag_chain_instance)}")

        # 4. Invoke the chain (Example based on vector_store.py test data)
        test_question = "What does LangChain help with?"
        logger.info(f"Invoking RAG chain with question: '{test_question}'")
        
        # Simple synchronous invocation
        response = rag_chain_instance.invoke(test_question)
        
        logger.info(f"\n--- RAG Response ---")
        print(response)
        logger.info("--- End RAG Response ---")
        
        # Example of streaming invocation (if LLM streaming=True)
        # logger.info("Testing streaming response...")
        # for chunk in rag_chain_instance.stream(test_question):
        #     print(chunk, end="", flush=True)
        # print("\n--- End Streaming Response ---")

    except Exception as e:
        logger.error(f"Error during RAG chain testing: {e}", exc_info=True) 