# LangExtract-Cli Web Demo

一个最小可运行的 Web 程序：在网页输入文本，调用 `langextract` + OpenAI 模型输出结构化 JSON。

## 1. 安装依赖（uv）

```bash
# 首次使用可先安装 uv
# Windows: winget install --id=astral-sh.uv -e
# macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync
```

## 2. 配置 API Key / 兼容接口地址

推荐使用环境变量：

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."
$env:OPENAI_BASE_URL="https://your-provider.example.com/v1"

# macOS/Linux
# export OPENAI_API_KEY="sk-..."
# export OPENAI_BASE_URL="https://your-provider.example.com/v1"
```

也可以在网页中临时填写 API Key 和 Base URL。

## 3. 启动服务

```bash
uv run python app.py
```

浏览器打开：`http://127.0.0.1:5000`

## 4. 说明

- 默认模型：`gpt-4o-mini`
- 使用 `langextract.extract(...)` 进行结构化抽取
- 支持 OpenAI 官方和 OpenAI 兼容接口（可配置 `OPENAI_BASE_URL`）
- 为 OpenAI 兼容配置 `fence_output=True` 与 `use_schema_constraints=False`
- 依赖管理与运行统一使用 `uv`
