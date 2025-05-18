# Project Plan: RAG & Knowledge Engine System Reference

*Status: As of [Current Date - Please Update]*

## 1. Overall Goal

Develop a reference implementation for Retrieval-Augmented Generation (RAG) and agent-based knowledge engine systems using Python and LangChain. The system should start with a functional baseline and iteratively incorporate advanced techniques, serving as a learning resource and adaptable foundation.

Refer to:
*   `../.cursor/rules/_project.mdc` (Overall Project Vision & Approach)
*   `PRD.md` / `PRD_zh.md` (Product Requirements)
*   `TECH_SPEC.md` / `TECH_SPEC_zh.md` (Technical Specifications)
*   `RESEARCH.md` / `RESEARCH_zh.md` (Optimization & Research Notes)

## 2. Completed Milestones

*   **M1: Project Initialization & Setup:**
    *   [x] Defined project structure (`src`, `docs`, `tests`, `data`).
    *   [x] Configured Conda environment (`environment.yml`).
    *   [x] Setup project metadata and dependencies (`pyproject.toml`).
    *   [x] Established development guidelines (`_global.mdc`, `_project.mdc`).
    *   [x] Created initial documentation (`PRD.md`, `TECH_SPEC.md`, `README.md`, `PLAN.md`, translations).
    *   [x] Implemented configuration loading (`core/config.py`).
*   **M2: Baseline RAG Pipeline - Ingestion:**
    *   [x] Implemented document loading (`core/loader.py`) supporting directories, files (txt, md, pdf), and URLs.
    *   [x] Implemented text splitting (`core/splitter.py`) using `RecursiveCharacterTextSplitter`.
    *   [x] Implemented vector store creation (`core/vector_store.py`) using `OpenAIEmbeddings` and `ChromaDB`.
    *   [x] Integrated ingestion steps into a pipeline (`pipelines/baseline_rag.py::run_ingestion_pipeline`).
    *   [x] Exposed ingestion via CLI (`cli.py::ingest`).
*   **M3: Baseline RAG Pipeline - Query:**
    *   [x] Implemented retriever creation (`core/retriever.py`) loading from ChromaDB.
    *   [x] Implemented RAG chain generation (`core/generator.py`) using LCEL, prompt templating, and configured LLM (OpenRouter/OpenAI).
    *   [x] Integrated query components into a pipeline (`pipelines/baseline_rag.py::get_baseline_rag_pipeline`) with component caching.
    *   [x] Exposed querying via CLI (`cli.py::query`).
*   **M4: Documentation & Planning Update:**
    *   [x] Created `WORKSPACE_OVERVIEW.md` and `RESEARCH.md` (and `_zh` versions).
    *   [x] Updated `PLAN.md` based on research findings.
*   **M5: Multi-Vector Store Support & Testing:**
    *   [x] Added support for Milvus vector database.
    *   [x] Added support for Qdrant vector database.
    *   [x] Successfully tested ingestion and query with ChromaDB, Milvus, and Qdrant.
    *   [ ] Weaviate support temporarily paused due to `weaviate-client` v4 and LangChain compatibility issues.

## 3. Current Status

A functional baseline RAG pipeline is implemented and accessible via the CLI. It supports document ingestion and querying using ChromaDB, Milvus, and Qdrant as vector store backends. LLM and Embedding models can be configured separately (e.g., OpenRouter for LLM, direct OpenAI for embeddings).

Weaviate integration is on hold pending resolution of client/LangChain wrapper compatibility.

## 4. Backlog / Next Steps

Items are roughly prioritized. Refer to `docs/RESEARCH.md` for more details on some of these techniques.

*   **P1: Testing (High Priority):**
    *   [ ] Implement unit tests for core components (`loader`, `splitter`, `vector_store`, `retriever`, `generator`) for all supported DBs.
    *   [ ] Implement integration tests for ingestion and query pipelines for all supported DBs.
    *   [ ] Setup CI/CD pipeline (e.g., GitHub Actions) to run tests automatically.
*   **P2: Evaluation Framework (HLR-F04) (High Priority):**
    *   [ ] Define key RAG evaluation metrics (e.g., using RAGAs: `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`).
    *   [ ] Select or build an evaluation dataset relevant to ingested content.
    *   [ ] Implement basic evaluation scripts callable via CLI.
*   **P3: Resolve Weaviate Integration:**
    *   [ ] Monitor `langchain-community` and `langchain-weaviate` for updates regarding `weaviate-client` v4 compatibility.
    *   [ ] If no timely official fix, investigate deeper into manual data handling with `weaviate-client` v4 as outlined in `RESEARCH.md` or pin to `weaviate-client` v3 as a temporary measure if critical.
*   **P4: Advanced Retrieval Strategies (HLR-F02):**
    *   [ ] Implement Re-ranking:
        *   [ ] Integrate a lightweight cross-encoder or `CohereRerank`.
    *   [ ] Explore and implement Hybrid Search (e.g., BM25 + semantic).
    *   [ ] Investigate Parent Document Retriever strategy.
    *   [ ] Explore Context-aware & Dynamic Chunking/Indexing (e.g., RAPTOR-like).
*   **P5: Query Understanding/Transformation (HLR-F01):**
    *   [ ] Implement Multi-Query Retriever.
    *   [ ] Explore and implement HyDE.
*   **P6: Agentic Components (HLR-F03):**
    *   [ ] Study `e2e` Agentic RAG examples.
    *   [ ] Re-factor current RAG chain into a LangGraph structure.
    *   [ ] Implement simple agent for iterative retrieval or self-correction.
*   **P7: Enhancements & Refinements:**
    *   [ ] Improve error handling and user feedback in CLI.
    *   [ ] Add chat history support to the query interface.
    *   [ ] Implement Contextual Compression (`ContextualCompressionRetriever`).
    *   [ ] Enhance answer generation with source attribution.
    *   [ ] Consider API interface (FastAPI).

## 5. Open Questions / Decisions

*   Finalize choice of specific evaluation dataset and initial metrics for P2.
*   Decision on Weaviate path forward (wait for LangChain updates vs. deeper manual integration vs. temporary v3 rollback).
*   (Add other open questions as they arise)

## 6. Detailed RAG Optimization & Research Plan (Phased Approach)

**Core Goal:** Build a modular, scalable, efficient RAG reference implementation capable of handling large-scale long documents, with advanced knowledge engine capabilities.

**Phase 0: Foundation & Evaluation Framework (Partially Completed/Ongoing)**

*   **Task 0.1:** Complete unit tests for core modules (`config`, `loader`, `splitter`, `generator`, `vector_store` - skip `vector_store` for now), ensuring code quality and stability. (Status: `config`, `loader`, `splitter` completed)
*   **Task 0.2:** **Establish Baseline RAG Evaluation Framework (Critical!)**
    *   **Sub-task 0.2.1:** Prepare/collect one or more QA datasets containing long documents (general or domain-specific).
    *   **Sub-task 0.2.2:** Define core evaluation metrics, including at least:
        *   Retrieval Quality: Context Precision, Context Recall, MRR (Mean Reciprocal Rank)
        *   Generation Quality: Faithfulness, Answer Relevancy, Answer Correctness (if ground truth available)
    *   **Sub-task 0.2.3:** Integrate evaluation tools (e.g., RAGAs, LangChain Evaluation) or write custom evaluation scripts.
    *   **Sub-task 0.2.4:** Evaluate the current baseline RAG system (simple chunking + basic retrieval + LLM generation) to get initial performance data.
*   **Deliverables:**
    *   Stable core codebase.
    *   Runnable evaluation pipeline and baseline performance report.

**Phase 1: Indexing & Retrieval Optimization (for Long Documents)**

*   **Research & Experimentation Focus:**
    *   **1.1. Advanced Chunking Strategies:**
        *   **1.1.1. Parent Document Retriever / Hierarchical Indexing:** Implement and evaluate LangChain's `ParentDocumentRetriever` or similar hierarchical indexing. Compare its effectiveness against naive chunking on long documents.
        *   **1.1.2. Sentence Window Retrieval:** Experiment with the sentence window method, evaluating its performance in providing focused context.
        *   **(Optional) 1.1.3. RAPTOR / Proposition-Based Chunking:** Preliminary research into its principles and potential implementation complexity.
    *   **1.2. Query Transformation/Expansion:**
        *   **1.2.1. HyDE:** Implement and evaluate HyDE.
        *   **1.2.2. Multi-Query Retriever:** Implement and evaluate Multi-Query.
    *   **1.3. Re-ranking:**
        *   **1.3.1. Cross-Encoder Re-ranker:** Integrate a lightweight cross-encoder (e.g., from `sentence-transformers`) for re-ranking.
        *   **(Optional) 1.3.2. LLM-based Re-ranker:** Experiment with using an LLM for intelligent re-ranking of retrieved results.
    *   **1.4. Hybrid Search:**
        *   **1.4.1. BM25 + Semantic Search:** Research and implement a solution combining BM25 (or other sparse retrieval) with current vector search, exploring result fusion strategies (RRF).
*   **Evaluation:** After completing each sub-task, perform quantitative evaluation using the framework established in Phase 0, comparing performance before and after optimization.
*   **Deliverables:**
    *   Experimental report on the performance of different indexing and retrieval strategies for long documents.
    *   1-2 validated advanced indexing/retrieval techniques integrated into the RAG system.

**Phase 2: Agentic RAG & Modular Architecture Evolution**

*   **Research & Experimentation Focus:**
    *   **2.1. Modular Refactoring:**
        *   **2.1.1.** Analyze the current RAG pipeline, identifying components that can be modularized (refer to Modular RAG definition in `RESEARCH.md`).
        *   **2.1.2.** Refactor the existing RAG chain into a more flexible graph structure using LangGraph or a similar orchestration tool.
    *   **2.2. Initial Agentic Capabilities Integration:**
        *   **2.2.1. Basic Routing Capability:** Implement a simple routing node, e.g., deciding whether to answer directly or initiate RAG based on query length or keywords.
        *   **2.2.2. Iterative Retrieval & Reflection (Self-Correction/Reflection Loop):**
            *   Design a simple agentic loop: Retrieve -> Generate -> LLM evaluates answer quality/confidence -> If unsatisfactory, rewrite query/adjust strategy and re-retrieve.
            *   Refer to `Self_RAG.ipynb` and `Corrective_RAG.ipynb` in the `e2e` directory.
    *   **2.3. Tool Usage:**
        *   **2.3.1.** Encapsulate different retrieval strategies implemented in Phase 1 (e.g., hybrid search, HyDE-enhanced search) as LangChain tools.
        *   **2.3.2.** Allow the Agent to select appropriate retrieval tools based on analysis.
*   **Evaluation:** Assess the effectiveness of Agentic RAG in handling complex queries and improving robustness. Evaluate the flexibility gained from modularity.
*   **Deliverables:**
    *   A prototype modular RAG system based on LangGraph.
    *   A RAG Agent with initial capabilities for agentic reflection and tool use.

**Phase 3: Generation Enhancement, Knowledge Management & Productionization Considerations**

*   **Research & Experimentation Focus:**
    *   **3.1. Context Management & Compression:**
        *   **3.1.1.** Experiment with LangChain's `ContextualCompressionRetriever`.
        *   **3.1.2.** Research other context selection and compression techniques.
    *   **3.2. Answer Attribution & Trustworthiness:**
        *   **3.2.1.** Improve prompt engineering to ensure the LLM explicitly cites source document snippets when generating answers.
        *   **3.2.2.** Implement functionality to display attribution information in the frontend (if applicable).
    *   **3.3. Dynamic Knowledge Base Update & Maintenance:**
        *   **3.3.1.** Design and preliminarily implement an incremental indexing mechanism for documents.
        *   **3.3.2.** Consider strategies for knowledge base version control.
    *   **3.4. Productionization Readiness:**
        *   **3.4.1. Performance Analysis:** Detailed analysis of system latency and cost bottlenecks.
        *   **3.4.2. Error Handling & Robustness:** Enhance the system's error handling mechanisms.
        *   **(Optional) 3.4.3. Knowledge Distillation:** If performance and cost are major bottlenecks, investigate the possibility of using knowledge distillation to fine-tune smaller models.
*   **Evaluation:** Assess the impact of context management on generation quality and efficiency, the accuracy of attribution, and the efficiency of knowledge base updates.
*   **Deliverables:**
    *   A more robust, maintainable RAG system with production potential.
    *   Documentation on system performance, cost, and maintenance strategies.

**Phase 4: Future Exploration (Select based on project progress and interest)**

*   Multi-Modal RAG
*   Deeper synergy with fine-tuning (retriever fine-tuning, generator fine-tuning)
*   More advanced Agent collaboration and planning
*   Personalized RAG

**Guiding Principles:**

*   **Iterative Development:** Each phase should also be iterative, with small, rapid steps.
*   **Documentation:** Detailed recording of experiments, results, issues encountered, and solutions.
*   **Code Reusability & Modularity:** Strive to write reusable, high-cohesion, low-coupling modules.
*   **Continuous Integration/Continuous Evaluation (CI/CE):** Ideally, every code change should trigger automated tests and calculation of core evaluation metrics.

This plan is extensive. You can adjust the depth and breadth of each phase based on actual time and resources. The key is to first establish a solid evaluation foundation (Phase 0) and then systematically explore and integrate selected optimization techniques. 