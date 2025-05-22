import pytest
from unittest.mock import MagicMock, ANY
from operator import itemgetter
from typing import Any, List # Ensure List is imported

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableConfig
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from src.rag_lang.core.generator import (
    format_docs,
    get_rag_chain,
    DEFAULT_RAG_TEMPLATE,
)
# BaseRetriever might not be strictly needed if MockRunnableRetriever is self-contained
# from src.rag_lang.core.retriever import BaseRetriever 

# Define a custom mock retriever that behaves like a Runnable
class MockTestRetriever(Runnable[Any, List[Document]]):
    def __init__(self):
        self.expected_input: Any = None
        self.return_value: List[Document] = []
        self.call_count = 0
        self.last_input: Any = None

    def invoke(self, input: Any, config: RunnableConfig | None = None) -> List[Document]:
        print(f"DEBUG MockTestRetriever: Invoked with input: {input}")
        self.call_count += 1
        self.last_input = input
        if self.expected_input is not None:
            assert input == self.expected_input, \
                f"MockTestRetriever expected input {self.expected_input}, got {input}"
        return self.return_value

    def configure(self, expected_input: Any, return_value: List[Document]):
        self.expected_input = expected_input
        self.return_value = return_value
        # Reset for each configuration if needed, or manage call counts carefully
        self.call_count = 0 
        self.last_input = None
        return self

@pytest.fixture
def mock_retriever_runnable() -> MockTestRetriever: # Type hint for clarity
    return MockTestRetriever()

@pytest.fixture
def mock_llm() -> MagicMock: # Type hint for clarity
    llm = MagicMock(spec=BaseChatModel)
    llm.invoke.return_value = "Default Mocked LLM Response"
    return llm

# Test for format_docs
def test_format_docs():
    docs = [
        Document(page_content="Content of doc 1"),
        Document(page_content="Content of doc 2"),
    ]
    expected_output = "Content of doc 1\n\nContent of doc 2"
    assert format_docs(docs) == expected_output

    docs_empty = []
    expected_output_empty = ""
    assert format_docs(docs_empty) == expected_output_empty

    docs_single = [Document(page_content="Single doc")]
    expected_output_single = "Single doc"
    assert format_docs(docs_single) == expected_output_single


# Test for get_rag_chain structure and prompt
def test_get_rag_chain_structure_and_prompt(mock_retriever_runnable: MockTestRetriever, mock_llm: MagicMock):
    """Test the structure of the RAG chain and the default prompt."""
    rag_chain = get_rag_chain(retriever=mock_retriever_runnable, llm=mock_llm)

    assert isinstance(rag_chain, Runnable), "RAG chain should be a Runnable instance."

    # Check chain structure (simplified check)
    # This is a bit brittle as it depends on internal LCEL structure representation
    # A more robust check might involve inspecting the 'graph' if available and simple enough
    # For now, we'll rely on the invocation test to ensure components are wired

    # Verify the prompt template used if no custom one is provided
    # Accessing the prompt template within the chain:
    # The chain is: {context: retriever | format_docs, question: passthrough} | prompt | llm | parser
    # So, the prompt is the second major component of the final sequence.
    # rag_chain.steps[1] would typically be the ChatPromptTemplate
    
    # Let's find the ChatPromptTemplate in the Runnable sequence
    prompt_template_in_chain = None
    if hasattr(rag_chain, 'middle'): # For sequences like RunnableSequence
        for item in rag_chain.middle:
            if isinstance(item, ChatPromptTemplate):
                prompt_template_in_chain = item
                break
    elif hasattr(rag_chain, 'steps'): # Older or different Runnable structures
         if len(rag_chain.steps) > 1 and isinstance(rag_chain.steps[1], ChatPromptTemplate):
            prompt_template_in_chain = rag_chain.steps[1]


    assert prompt_template_in_chain is not None, "Could not find ChatPromptTemplate in the chain."
    # MODIFIED: Access template string correctly from ChatPromptTemplate
    actual_template_str = ""
    if prompt_template_in_chain.messages and hasattr(prompt_template_in_chain.messages[0], 'prompt') and hasattr(prompt_template_in_chain.messages[0].prompt, 'template'):
        actual_template_str = prompt_template_in_chain.messages[0].prompt.template
    
    assert actual_template_str == DEFAULT_RAG_TEMPLATE, \
        f"RAG chain should use the DEFAULT_RAG_TEMPLATE by default. Found: {actual_template_str}"

    # Test with a custom prompt
    custom_prompt_template = "Custom Context: {context} Question: {question} Answer:"
    # Configure the mock retriever for this specific chain creation, though it's not strictly necessary
    # as this test mainly checks prompt structure, not retriever interaction.
    mock_retriever_runnable.configure(expected_input=ANY, return_value=[]) 
    rag_chain_custom_prompt = get_rag_chain(
        retriever=mock_retriever_runnable, llm=mock_llm, prompt_template=custom_prompt_template
    )
    
    custom_prompt_in_chain = None
    if hasattr(rag_chain_custom_prompt, 'middle'):
        for item in rag_chain_custom_prompt.middle:
            if isinstance(item, ChatPromptTemplate):
                custom_prompt_in_chain = item
                break
    elif hasattr(rag_chain_custom_prompt, 'steps'):
         if len(rag_chain_custom_prompt.steps) > 1 and isinstance(rag_chain_custom_prompt.steps[1], ChatPromptTemplate):
            custom_prompt_in_chain = rag_chain_custom_prompt.steps[1]

    assert custom_prompt_in_chain is not None, "Could not find ChatPromptTemplate in the custom prompt chain."
    # MODIFIED: Access template string correctly for custom prompt
    actual_custom_template_str = ""
    if custom_prompt_in_chain.messages and hasattr(custom_prompt_in_chain.messages[0], 'prompt') and hasattr(custom_prompt_in_chain.messages[0].prompt, 'template'):
        actual_custom_template_str = custom_prompt_in_chain.messages[0].prompt.template

    assert actual_custom_template_str == custom_prompt_template, \
        f"RAG chain should use the provided custom_prompt_template. Found: {actual_custom_template_str}"


def test_get_rag_chain_invocation_flow(mock_retriever_runnable: MockTestRetriever, mock_llm: MagicMock):
    """Test the full invocation flow, focusing on LLM input and final output."""
    test_question = "What is the RAG lang project?"
    mocked_retriever_docs = [
        Document(page_content="Document 1 about RAG lang project."),
        Document(page_content="Document 2: More details on RAG lang.")
    ]
    # Configure the mock retriever for this specific invocation
    mock_retriever_runnable.configure(expected_input=test_question, return_value=mocked_retriever_docs)

    rag_chain = get_rag_chain(retriever=mock_retriever_runnable, llm=mock_llm)

    final_llm_response = "The RAG lang project helps build advanced RAG systems."
    mock_llm.invoke.return_value = final_llm_response

    response = rag_chain.invoke({"question": test_question})

    # 1. Assert our mock retriever was called correctly
    assert mock_retriever_runnable.call_count == 1, "MockTestRetriever was not called once."
    assert mock_retriever_runnable.last_input == test_question, \
        f"MockTestRetriever called with {mock_retriever_runnable.last_input}, expected {test_question}"

    # 2. Assert LLM was called
    mock_llm.invoke.assert_called_once()

    # Print details of the call to LLM for debugging
    print(f"DEBUG: mock_llm.invoke.call_args: {mock_llm.invoke.call_args}")

    # 3. Verify the input to the LLM contains the formatted context and question
    # ((llm_input_args,), _) = mock_llm.invoke.call_args # This was causing ValueError
    # Assuming the first positional argument to invoke is the prompt object
    llm_input_prompt_object = mock_llm.invoke.call_args[0][0]
    llm_prompt_str = llm_input_prompt_object.to_string()

    expected_context_str = format_docs(mocked_retriever_docs)
    assert expected_context_str in llm_prompt_str, \
        f"Formatted context '{expected_context_str}' not found in LLM input '{llm_prompt_str}'"
    assert test_question in llm_prompt_str, \
        f"Question '{test_question}' not found in LLM input '{llm_prompt_str}'"
    
    # Check if the default template structure is somewhat present
    # This is a loose check, as the exact formatting depends on ChatPromptTemplate internals
    assert "Answer:" in llm_prompt_str # Part of DEFAULT_RAG_TEMPLATE

    # 4. Assert final response is as expected (comes from mocked LLM)
    assert response == final_llm_response


def test_get_rag_chain_custom_prompt(mock_retriever_runnable: MockTestRetriever, mock_llm: MagicMock):
    """Test RAG chain with a custom prompt, focusing on LLM input and final output."""
    # Reset mocks if they are not function-scoped by default (pytest usually re-creates them)
    # mock_retriever_runnable.invoke.reset_mock() # REMOVED: mock_retriever_runnable.invoke is not a MagicMock
    mock_llm.invoke.reset_mock() # mock_llm.invoke IS a MagicMock

    custom_prompt_str = "My Custom Prompt! Context: {context} --- Question: {question} --- Answer Please:"
    test_question = "How does the custom prompt work?"
    mocked_retriever_docs_custom = [Document(page_content="Custom context for this specific test.")]
    
    # Configure mock retriever
    mock_retriever_runnable.configure(expected_input=test_question, return_value=mocked_retriever_docs_custom)
    mock_llm.invoke.reset_mock() # Reset LLM mock for fresh assertions in this test

    rag_chain = get_rag_chain(retriever=mock_retriever_runnable, llm=mock_llm, prompt_template=custom_prompt_str)
    
    final_llm_response_custom = "The custom prompt works by new formatting!"
    mock_llm.invoke.return_value = final_llm_response_custom

    response = rag_chain.invoke({"question": test_question})

    # Assert retriever
    assert mock_retriever_runnable.call_count == 1
    assert mock_retriever_runnable.last_input == test_question
    
    # Assert LLM
    mock_llm.invoke.assert_called_once()
    llm_input_prompt_object_custom = mock_llm.invoke.call_args[0][0]
    llm_prompt_str = llm_input_prompt_object_custom.to_string()

    expected_context_custom_str = format_docs(mocked_retriever_docs_custom)
    assert expected_context_custom_str in llm_prompt_str
    assert test_question in llm_prompt_str
    assert "My Custom Prompt!" in llm_prompt_str # Check for unique part of custom template
    assert "--- Answer Please:" in llm_prompt_str

    assert response == final_llm_response_custom 