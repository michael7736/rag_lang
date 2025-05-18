# RAG 与知识引擎系统参考实现

一个使用 Python 和 LangChain 构建的检索增强生成 (RAG) 和基于 Agent 的知识引擎系统的参考实现。

有关项目详情，请参阅 `docs/PRD.md` 和 `docs/TECH_SPEC.md` (及其对应的 `_zh.md` 中文版本)。

## 安装设置

1.  **创建 Conda 环境：**
    ```bash
    conda env create -f environment.yml
    conda activate rag_lang
    ```

2.  **环境变量：**
    在项目根目录下创建一个 `.env` 文件，并添加您的 API 密钥：
    ```dotenv
    # OpenRouter (OpenAI 兼容) 示例
    OPENROUTER_API_KEY="your-openrouter-api-key"
    OPENAI_API_KEY="your-openrouter-api-key" # OpenAI 兼容性，使用相同的密钥
    OPENAI_API_BASE="https://openrouter.ai/api/v1" # OpenRouter 端点

    # 直接使用 OpenAI 示例
    # OPENAI_API_KEY="your-real-openai-api-key"
    ```
    *请确保为嵌入和 LLM 调用正确设置了 `OPENAI_API_KEY`。*

3.  **以可编辑模式安装包（可选，但推荐开发时使用）：**
    ```bash
    pip install -e .
    ```

## 使用方法

主要通过命令行工具进行交互。

### 1. 摄入文档

在查询之前，您需要将文档摄入到向量存储中。该工具支持从本地目录、单个文件或 Web URL 摄入。

```bash
# 从目录中摄入所有支持的文件（.txt, .md, .pdf）
python -m rag_lang.cli ingest ./path/to/your/data

# 摄入单个文件
python -m rag_lang.cli ingest ./path/to/your/document.pdf

# 从 URL 摄入内容
python -m rag_lang.cli ingest https://example.com/your-page.html

# 指定不同的向量存储位置和集合名称（可选）
python -m rag_lang.cli ingest ./data --persist-dir ./my_vector_store --collection my_documents 
```

此命令将：
1. 从指定来源加载文档。
2. 将它们分割成块。
3. 使用配置的模型（默认为 OpenAI `text-embedding-ada-002`）生成嵌入。
4. 将文本块和嵌入存储在 ChromaDB 向量存储中（默认位置：`./chroma`）。

### 2. 查询系统

文档摄入后，您可以提问：

```bash
python -m rag_lang.cli query "已摄入文档的主要主题是什么？"
```

将问句字符串替换为您的实际查询。系统将：
1. 加载向量存储。
2. 创建一个检索器。
3. 使用配置的 LLM（默认为 OpenRouter `openai/gpt-4o`）创建 RAG 链。
4. 根据您的问题检索相关的文档块。
5. 基于检索到的上下文生成答案。
6. 将答案打印到控制台。

## 开发

*   遵循 `.cursor/rules/_global.mdc` 和 `.cursor/rules/_project.mdc` 中的指南。
*   运行测试：`pytest` (测试有待实现)
*   运行代码检查工具：`flake8 src tests`
*   运行类型检查工具：`mypy src` 