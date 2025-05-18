# 项目计划: RAG 与知识引擎系统参考实现

*状态：截至 [当前日期 - 请更新]*

## 1. 总体目标

使用 Python 和 LangChain 开发一个检索增强生成 (RAG) 和基于 Agent 的知识引擎系统的参考实现。该系统应从一个功能基线开始，并迭代地集成先进技术，作为学习资源和可适应的基础。

参考文档：
*   `../.cursor/rules/_project.mdc` (总体项目愿景和方法)
*   `PRD.md` / `PRD_zh.md` (产品需求)
*   `TECH_SPEC.md` / `TECH_SPEC_zh.md` (技术规格)
*   `RESEARCH.md` / `RESEARCH_zh.md` (优化方案与研究笔记)

## 2. 已完成的里程碑

*   **M1: 项目初始化与设置：**
    *   [x] 定义了项目结构 (`src`, `docs`, `tests`, `data`)。
    *   [x] 配置了 Conda 环境 (`environment.yml`)。
    *   [x] 设置了项目元数据和依赖 (`pyproject.toml`)。
    *   [x] 建立了开发指南 (`_global.mdc`, `_project.mdc`)。
    *   [x] 创建了初始文档 (`PRD.md`, `TECH_SPEC.md`, `README.md`, `PLAN.md`, 翻译版本)。
    *   [x] 实现了配置加载 (`core/config.py`)。
*   **M2: 基线 RAG 流水线 - 摄入：**
    *   [x] 实现了文档加载 (`core/loader.py`)，支持目录、文件（txt, md, pdf）和 URL。
    *   [x] 实现了文本分割 (`core/splitter.py`)，使用 `RecursiveCharacterTextSplitter`。
    *   [x] 实现了向量存储创建 (`core/vector_store.py`)，使用 `OpenAIEmbeddings` 和 `ChromaDB`。
    *   [x] 将摄入步骤集成到流水线中 (`pipelines/baseline_rag.py::run_ingestion_pipeline`)。
    *   [x] 通过 CLI 提供了摄入功能 (`cli.py::ingest`)。
*   **M3: 基线 RAG 流水线 - 查询：**
    *   [x] 实现了检索器创建 (`core/retriever.py`)，从 ChromaDB 加载。
    *   [x] 实现了 RAG 链生成 (`core/generator.py`)，使用 LCEL、提示模板和配置的 LLM (OpenRouter/OpenAI)。
    *   [x] 将查询组件集成到流水线中 (`pipelines/baseline_rag.py::get_baseline_rag_pipeline`)，并带有组件缓存。
    *   [x] 通过 CLI 提供了查询功能 (`cli.py::query`)。
*   **M4: 文档与计划更新：**
    *   [x] 创建了 `WORKSPACE_OVERVIEW.md` 和 `RESEARCH.md` (及其 `_zh` 中文版)。
    *   [x] 基于研究发现更新了 `PLAN.md`。
*   **M5: 多向量数据库支持与测试：**
    *   [x] 添加了对 Milvus 向量数据库的支持。
    *   [x] 添加了对 Qdrant 向量数据库的支持。
    *   [x] 成功测试了使用 ChromaDB, Milvus 和 Qdrant 进行数据摄入和查询。
    *   [ ] Weaviate 支持由于 `weaviate-client` v4 与 LangChain 兼容性问题暂时暂停。

## 3. 当前状态

已实现一个功能性的基线 RAG 流水线，可通过 CLI 访问。它支持使用 ChromaDB、Milvus 和 Qdrant 作为向量存储后端进行文档摄入和查询。LLM 和嵌入模型可以分开配置（例如，LLM 使用 OpenRouter，嵌入直接使用 OpenAI）。

Weaviate 的集成工作因客户端/LangChain 包装器兼容性问题而暂停。

## 4. 待办事项 / 后续步骤

项目大致按优先级排序。有关某些技术的更多详细信息，请参阅 `docs/RESEARCH.md`。

*   **P1: 测试 (高优先级)：**
    *   [ ] 为核心组件 (`loader`, `splitter`, `vector_store`, `retriever`, `generator`) 针对所有支持的数据库实现单元测试。
    *   [ ] 为所有支持的数据库的摄入和查询流水线实现集成测试。
    *   [ ] 设置 CI/CD 流水线（例如 GitHub Actions）以自动运行测试。
*   **P2: 评估框架 (HLR-F04) (高优先级)：**
    *   [ ] 定义关键 RAG 评估指标（例如，使用 RAGAs：`faithfulness` 忠实度, `answer_relevancy` 答案相关性, `context_precision` 上下文精度, `context_recall` 上下文召回率）。
    *   [ ] 选择或构建与摄入内容相关的评估数据集。
    *   [ ] 实现可通过 CLI 调用的基础评估脚本。
*   **P3: 解决 Weaviate 集成问题：**
    *   [ ] 关注 `langchain-community` 和 `langchain-weaviate` 关于 `weaviate-client` v4 兼容性的更新。
    *   [ ] 如果没有及时的官方修复，则按照 `RESEARCH.md` 中的方案深入研究手动处理 `weaviate-client` v4 数据，或作为临时措施在关键情况下固定使用 `weaviate-client` v3。
*   **P4: 高级检索策略 (HLR-F02)：**
    *   [ ] 实现重排序 (Re-ranking)：
        *   [ ] 集成一个轻量级的交叉编码器或 `CohereRerank`。
    *   [ ] 探索并实现混合搜索 (例如 BM25 + 语义搜索)。
    *   [ ] 研究父文档检索器 (Parent Document Retriever) 策略。
    *   [ ] 探索上下文感知与动态分块/索引 (例如，类 RAPTOR 方法)。
*   **P5: 查询理解/转换 (HLR-F01)：**
    *   [ ] 实现多查询检索器 (Multi-Query Retriever)。
    *   [ ] 探索并实现 HyDE。
*   **P6: Agent 组件 (HLR-F03)：**
    *   [ ] 学习 `e2e` 目录中的 Agentic RAG 示例。
    *   [ ] 将当前的 RAG 链重构为 LangGraph 结构。
    *   [ ] 实现简单的 Agent 用于迭代检索或自我校正。
*   **P7: 增强与优化：**
    *   [ ] 改进 CLI 中的错误处理和用户反馈。
    *   [ ] 为查询接口添加聊天历史支持。
    *   [ ] 实现上下文压缩 (`ContextualCompressionRetriever`)。
    *   [ ] 增强答案生成，加入来源追溯功能。
    *   [ ] 考虑 API 接口 (FastAPI)。

## 5. 待解决问题 / 待定决策

*   最终确定 P2 的具体评估数据集和初始指标选择。
*   关于 Weaviate 的后续方案决策（等待 LangChain 更新 vs. 更深入的手动集成 vs. 临时回退到 v3）。
*   （在此添加其他出现的问题）

## 6. RAG 详细优化与研究计划 (分阶段方法)

**核心目标:** 构建一个模块化、可扩展、高效的 RAG 参考实现，能够处理大规模长文档，并具备先进的知识引擎能力。

**阶段 0: 基础夯实与评估框架 (部分完成/进行中)**

*   **任务 0.1:** 完成核心模块 (`config`, `loader`, `splitter`, `generator`, `vector_store` - 暂时跳过 `vector_store`) 的单元测试，确保代码质量和稳定性。(状态: `config`, `loader`, `splitter` 已完成)
*   **任务 0.2:** **建立基线 RAG 评估框架 (关键!)**
    *   **子任务 0.2.1:** 准备/收集一个或多个包含长文档的问答数据集（通用领域或特定领域）。
    *   **子任务 0.2.2:** 定义核心评估指标，至少包括：
        *   检索质量: 上下文精度 (Context Precision), 上下文召回率 (Context Recall), 平均倒数排名 (MRR)
        *   生成质量: 忠实度 (Faithfulness), 答案相关性 (Answer Relevancy), 答案正确性 (Answer Correctness) (若有标准答案)
    *   **子任务 0.2.3:** 集成评估工具（如 RAGAs, LangChain Evaluation）或编写自定义评估脚本。
    *   **子任务 0.2.4:** 对当前基线 RAG 系统（简单分块 + 基础检索 + LLM 生成）进行评估，获得初始性能数据。
*   **交付成果:**
    *   稳定的核心代码库。
    *   可运行的评估流水线和基线性能报告。

**阶段 1: 索引与检索优化 (针对长文档)**

*   **研究与实验重点:**
    *   **1.1. 高级分块策略:**
        *   **1.1.1. 父文档检索器 / 分层索引 (Parent Document Retriever / Hierarchical Indexing):** 实现并评估 LangChain 的 `ParentDocumentRetriever` 或类似的分层索引策略。比较其与朴素分块在长文档上的效果。
        *   **1.1.2. 句子窗口检索 (Sentence Window Retrieval):** 实验句子窗口方法，评估其在提供精确上下文方面的表现。
        *   **(可选) 1.1.3. RAPTOR / 基于命题分块 (Proposition-Based Chunking):** 初步研究其原理和潜在实现复杂度。
    *   **1.2. 查询转换/扩展 (Query Transformation/Expansion):**
        *   **1.2.1. HyDE:** 实现并评估 HyDE。
        *   **1.2.2. 多查询检索器 (Multi-Query Retriever):** 实现并评估 Multi-Query。
    *   **1.3. 重排序 (Re-ranking):**
        *   **1.3.1. 交叉编码器重排序器 (Cross-Encoder Re-ranker):** 集成一个轻量级交叉编码器（如来自 `sentence-transformers`）进行重排序。
        *   **(可选) 1.3.2. 基于 LLM 的重排序器 (LLM-based Re-ranker):** 实验使用 LLM 对检索结果进行智能排序。
    *   **1.4. 混合搜索 (Hybrid Search):**
        *   **1.4.1. BM25 + 语义搜索:** 研究并实现将 BM25 (或其他稀疏检索) 与当前向量搜索结合的方案，探索结果融合策略 (RRF)。
*   **评估:** 每个子任务完成后，使用阶段 0 建立的评估框架进行量化评估，对比优化前后的性能。
*   **交付成果:**
    *   关于不同索引和检索策略在长文档场景下表现的实验报告。
    *   集成到 RAG 系统中的、经过验证有效的1-2种高级索引/检索技术。

**阶段 2: Agentic RAG 与模块化架构演进**

*   **研究与实验重点:**
    *   **2.1. 模块化重构:**
        *   **2.1.1.** 分析当前 RAG 流水线，识别可模块化的组件（参考 `RESEARCH_zh.md` 中模块化 RAG 的定义）。
        *   **2.1.2.** 使用 LangGraph 或类似流程编排工具，将现有 RAG 链重构为更灵活的图结构。
    *   **2.2. Agentic 能力初步集成:**
        *   **2.2.1. 基本路由能力:** 实现一个简单的路由节点，例如，根据查询长度或关键词决定是直接回答还是启动 RAG。
        *   **2.2.2. 迭代式检索与反思 (Self-Correction/Reflection Loop):**
            *   设计一个简单的 Agentic 循环：检索 -> 生成 -> LLM评估答案质量/置信度 -> 若不满意，则改写查询/调整策略后重新检索。
            *   参考 `e2e` 目录中的 `Self_RAG.ipynb` 和 `Corrective_RAG.ipynb`。
    *   **2.3. 工具使用:**
        *   **2.3.1.** 将阶段 1 中实现的不同检索策略（如混合搜索、HyDE增强的搜索）封装为 LangChain 工具。
        *   **2.3.2.** 允许 Agent 根据分析选择合适的检索工具。
*   **评估:** 评估 Agentic RAG 在处理复杂查询、提高鲁棒性方面的效果。评估模块化带来的灵活性。
*   **交付成果:**
    *   一个基于 LangGraph 的模块化 RAG 系统原型。
    *   初步实现 Agentic 反思和工具使用能力的 RAG Agent。

**阶段 3: 生成增强、知识管理与生产化考量**

*   **研究与实验重点:**
    *   **3.1. 上下文管理与压缩:**
        *   **3.1.1.** 实验 LangChain 的 `ContextualCompressionRetriever`。
        *   **3.1.2.** 研究其他上下文选择和压缩技术。
    *   **3.2. 答案溯源与可信度:**
        *   **3.2.1.** 改进提示工程，确保 LLM 生成答案时能明确引用来源文档片段。
        *   **3.2.2.** 实现前端展示溯源信息的功能（如果适用）。
    *   **3.3. 动态知识库更新与维护:**
        *   **3.3.1.** 设计并初步实现文档的增量索引机制。
        *   **3.3.2.** 考虑知识库版本控制的策略。
    *   **3.4. 生产化准备:**
        *   **3.4.1. 性能分析:** 详细分析系统的延迟和成本瓶颈。
        *   **3.4.2. 错误处理与鲁棒性:** 增强系统的错误处理机制。
        *   **(可选) 3.4.3. 知识蒸馏:** 如果性能和成本是主要瓶颈，研究使用知识蒸馏微调更小模型的可能性。
*   **评估:** 评估上下文管理对生成质量和效率的影响，溯源的准确性，以及知识库更新的效率。
*   **交付成果:**
    *   更鲁棒、可维护、具备生产潜力的 RAG 系统。
    *   关于系统性能、成本和维护策略的文档。

**阶段 4: 未来探索 (根据项目进展和兴趣选择)**

*   多模态 RAG
*   与微调的深度协同（检索器微调、生成器微调）
*   更高级的 Agent 协作和规划
*   个性化 RAG

**指导原则:**

*   **迭代开发:** 每个阶段内部也应该是迭代的，小步快跑。
*   **文档记录:** 详细记录实验过程、结果、遇到的问题和解决方案。
*   **代码复用与模块化:** 尽可能编写可复用、高内聚低耦合的模块。
*   **持续集成/持续评估 (CI/CE):** 理想情况下，每次代码变更都应触发自动化测试和核心评估指标的计算。

此计划内容广泛，您可以根据实际时间和资源调整每个阶段的深度和广度。关键是先建立好坚实的评估基础 (阶段 0)，然后系统地探索和集成选定的优化技术。 