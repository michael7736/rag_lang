# 技术规格书: RAG 与知识引擎系统参考实现

## 1. 引言

### 1.1 概述
本文档详细说明了 RAG 与知识引擎系统参考实现的技术设计和架构。它扩展了 `PRD.md` 中概述的需求，并与 `../.cursor/rules/_project.mdc` 中的项目目标以及 `../.cursor/rules/_global.mdc` 中的全局指南保持一致。

### 1.2 目标
*   定义系统架构和组件交互。
*   为基线实现指定技术栈和库。
*   概述数据摄入、检索和生成的流程。
*   为迭代开发和未来增强提供基础。

## 2. 系统架构

### 2.1 高层示意图
```mermaid
graph TD
    A[用户界面 (CLI/API)] --> B{查询处理器};
    B --> C[检索器];
    C --> D[向量存储库];
    C --> E[LLM 用于生成];
    E --> A;

    F[文档源] --> G{摄入流水线};
    G --> H[文档加载器];
    H --> I[文本分割器];
    I --> J[嵌入模型];
    J --> D;
```
*示意图将在组件实现过程中不断完善。*

### 2.2 组件
*   **摄入流水线 (Ingestion Pipeline):** 处理文档的加载、处理和嵌入。
    *   **文档加载器 (Document Loader):** 从各种来源读取文档。
    *   **文本分割器 (Text Splitter):** 将文档分割成易于管理的小块。
    *   **嵌入模型 (Embedding Model):** 将文本块转换为向量嵌入。
*   **向量存储库 (Vector Store):** 存储和索引文档嵌入，以实现高效检索。
*   **查询处理器 (Query Processor):** 处理用户查询，可能包括查询转换。
*   **检索器 (Retriever):** 根据处理后的查询从向量存储库中获取相关的文档块。
*   **LLM 用于生成 (LLM for Generation):** 基于检索到的上下文和原始查询合成答案。
*   **用户界面 (User Interface):** 提供用户与系统交互的方式。

## 3. 技术栈 (基线 - 迭代 1)

*   **编程语言:** Python (依据 `_global.mdc`)
*   **包管理:** Conda (依据 `_global.mdc`)
*   **核心框架:** LangChain (`langchain-core`, `langchain-community`, `langgraph` 依据 `_global.mdc`)
*   **文档加载器:** LangChain 社区加载器 (例如 `PyPDFLoader`, `TextLoader`)。
*   **文本分割器:** LangChain 分割器 (例如 `RecursiveCharacterTextSplitter`)。
*   **嵌入模型:** 通过 LangChain 配置 (例如 `HuggingFaceEmbeddings`, `OpenAIEmbeddings`)。初始默认：`HuggingFaceEmbeddings` (例如 `sentence-transformers/all-MiniLM-L6-v2`)。
*   **向量存储库:** 通过 LangChain 配置。初始默认：`ChromaDB` 以方便本地开发。
*   **LLM:** 通过 LangChain 配置 (例如 `ChatOpenAI`, `HuggingFaceHub`)。初始默认：一个小型、可本地运行的模型（如果可行），或一个广泛可用的 API 模型。
*   **测试:** `pytest` (依据 `_global.mdc`)
*   **代码检查/格式化:** `flake8`, `mypy` (依据 `_global.mdc`)

## 4. 数据流

### 4.1 摄入流
1.  从数据源（例如本地目录）提供文档。
2.  `文档加载器` 读取内容。
3.  `文本分割器` 将内容分解为文本块。
4.  `嵌入模型` 为每个文本块生成向量嵌入。
5.  嵌入和相应的文本存储在 `向量存储库` 中。

### 4.2 查询流
1.  用户通过 `用户界面` 提交查询。
2.  `查询处理器` （初始为简单传递）准备查询。
3.  `检索器` 使用查询嵌入在 `向量存储库` 中查找相似的文档块。
4.  检索到的文本块（上下文）和原始查询传递给 `LLM 用于生成`。
5.  LLM 生成答案。
6.  答案通过 `用户界面` 返回给用户。

## 5. 模块化与接口
*   组件将设计为具有清晰接口的 Python 类/函数。
*   将广泛利用 LangChain 的 Runnable 协议 (`langchain-core.runnables`) 来链接组件并确保可互换性。
*   将使用配置文件（例如 YAML 或 Python 字典）来管理 LLM 选择、嵌入模型和其他参数。

## 6. 日志与可观察性
*   利用 Python 的 `logging` 模块。
*   将记录关键操作、错误和组件交互。
*   可以集成 LangChain 的跟踪/调试功能（例如 LangSmith，如果配置）以获得更深入的洞察。

## 7. 未来增强 (技术视角)
*   **查询转换:** 实现 HyDE、多查询等模块。
*   **高级检索:** 集成混合搜索、重排序组件。
*   **Agent 框架:** 探索 LangGraph 以实现复杂的 Agent 行为。
*   **评估:** 开发用于衡量上下文相关性、答案忠实度等指标的脚本。

## 8. 待解决问题 / 待最终确定的设计选择
*   具体的初始默认 LLM 模型（平衡性能和可访问性）。
*   在向量存储库中与嵌入一起存储元数据的详细模式 (schema)。 