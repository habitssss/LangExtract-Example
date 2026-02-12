"""Flask Web 应用：将自由文本通过 langextract 转成结构化结果。"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any

from dotenv import load_dotenv
from flask import Flask, render_template, request
import langextract as lx
import langextract.prompt_validation as pv
import langextract.tokenizer as tokenizer_lib
from langextract.providers.openai import OpenAILanguageModel

load_dotenv()

app = Flask(__name__)


def build_default_examples() -> list[Any]:
    """构造默认 few-shot 示例。

    Returns:
        list[Any]: 可传给 ``lx.extract`` 的示例列表，用于约束输出结构。
    """
    return [
        lx.data.ExampleData(
            text=(
                "2025年3月12日，Alice 与 Bob 在上海签署了一份价值 120 万人民币的"
                "软件采购合同，甲方为星云科技。"
            ),
            extractions=[
                lx.data.Extraction(
                    extraction_class="Time",
                    extraction_text="2025年3月12日",
                    attributes={"normalized": "2025-03-12"},
                ),
                lx.data.Extraction(
                    extraction_class="Person",
                    extraction_text="Alice",
                    attributes={"role": "signer"},
                ),
                lx.data.Extraction(
                    extraction_class="Person",
                    extraction_text="Bob",
                    attributes={"role": "signer"},
                ),
                lx.data.Extraction(
                    extraction_class="Location",
                    extraction_text="上海",
                    attributes={},
                ),
                lx.data.Extraction(
                    extraction_class="Money",
                    extraction_text="120 万人民币",
                    attributes={"currency": "CNY", "amount": 1200000},
                ),
                lx.data.Extraction(
                    extraction_class="Organization",
                    extraction_text="星云科技",
                    attributes={"role": "buyer"},
                ),
                lx.data.Extraction(
                    extraction_class="Event",
                    extraction_text="签署了一份价值 120 万人民币的软件采购合同",
                    attributes={"type": "contract_signing"},
                ),
            ],
        )
    ]


def to_serializable(data: Any) -> Any:
    """将 langextract 返回对象递归转换为可 JSON 序列化结构。

    Args:
        data (Any): 任意 Python 对象（包含 dataclass / pydantic / list / dict 等）。

    Returns:
        Any: 可被 ``json.dumps`` 序列化的对象。
    """
    if data is None or isinstance(data, (str, int, float, bool)):
        return data

    if isinstance(data, list):
        return [to_serializable(item) for item in data]

    if isinstance(data, tuple):
        return [to_serializable(item) for item in data]

    if isinstance(data, dict):
        return {str(k): to_serializable(v) for k, v in data.items()}

    # pydantic v2
    if hasattr(data, "model_dump") and callable(data.model_dump):
        return to_serializable(data.model_dump())

    # dataclass
    if dataclasses.is_dataclass(data):
        return to_serializable(dataclasses.asdict(data))

    # 兜底：尝试对象属性字典
    if hasattr(data, "__dict__"):
        return to_serializable(vars(data))

    return str(data)


def extract_structured(
    input_text: str,
    prompt_description: str,
    model_id: str,
    api_key: str | None,
    base_url: str | None,
) -> Any:
    """调用 langextract 执行结构化抽取。

    Args:
        input_text (str): 待抽取的原始文本。
        prompt_description (str): 抽取任务描述。
        model_id (str): 模型 ID，如 ``gpt-4o-mini`` 或兼容 OpenAI 协议的模型名。
        api_key (str | None): API Key；为空时回退环境变量 ``OPENAI_API_KEY``。
        base_url (str | None): 兼容接口地址；为空时回退环境变量 ``OPENAI_BASE_URL``。

    Returns:
        Any: ``lx.extract`` 的原始返回对象（通常是 AnnotatedDocument 或其列表）。
    """
    final_api_key = (api_key or "").strip() or os.getenv("OPENAI_API_KEY")
    if not final_api_key:
        raise ValueError("缺少 API Key：请在页面填写，或设置环境变量 OPENAI_API_KEY。")

    final_base_url = (base_url or "").strip() or os.getenv("OPENAI_BASE_URL")

    cleaned_model_id = model_id.strip()
    if not cleaned_model_id:
        raise ValueError("模型 ID 不能为空。")

    # 直接构造 OpenAI 兼容模型，避免 provider 名称解析差异导致失败。
    model = OpenAILanguageModel(
        model_id=cleaned_model_id,
        api_key=final_api_key,
        base_url=final_base_url or None,
        # 对部分 OpenAI 兼容网关更友好：避免强制 JSON mode(response_format=json_object)。
        format_type=lx.data.FormatType.YAML,
    )

    result = lx.extract(
        text_or_documents=input_text,
        prompt_description=prompt_description,
        examples=build_default_examples(),
        model=model,
        # 与 model.format_type 保持一致，避免解析阶段按 JSON 解析 YAML 文本。
        format_type=lx.data.FormatType.YAML,
        fence_output=True,
        use_schema_constraints=False,
        # 中文文本建议使用 UnicodeTokenizer，降低对齐失败概率。
        tokenizer=tokenizer_lib.UnicodeTokenizer(),
        # 关闭示例对齐告警，避免非关键 warning 干扰排障。
        prompt_validation_level=pv.PromptValidationLevel.OFF,
        # 关闭进度条，避免看起来“卡住”。
        show_progress=False,
    )
    return result


@app.get("/")
def home() -> str:
    """渲染首页。

    Returns:
        str: HTML 页面。
    """
    return render_template(
        "index.html",
        result_json=None,
        error=None,
        model_id="gpt-4o-mini",
        base_url=os.getenv("OPENAI_BASE_URL", ""),
    )


@app.post("/extract")
def extract_view() -> str:
    """处理表单提交并返回抽取结果。

    Returns:
        str: HTML 页面（包含 JSON 结果或错误信息）。
    """
    input_text = request.form.get("input_text", "")
    prompt_description = request.form.get("prompt_description", "").strip()
    model_id = request.form.get("model_id", "gpt-4o-mini")
    api_key = request.form.get("api_key", "")
    base_url = request.form.get("base_url", "")

    if not input_text.strip():
        return render_template(
            "index.html",
            result_json=None,
            error="输入文本不能为空。",
            input_text=input_text,
            model_id=model_id,
            base_url=base_url,
        )

    try:
        result = extract_structured(
            input_text=input_text,
            prompt_description=prompt_description,
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
        )
        result_json = json.dumps(to_serializable(result), ensure_ascii=False, indent=2)
        return render_template(
            "index.html",
            result_json=result_json,
            error=None,
            input_text=input_text,
            model_id=model_id,
            base_url=base_url,
        )
    except Exception as exc:  # noqa: BLE001
        return render_template(
            "index.html",
            result_json=None,
            error=str(exc),
            input_text=input_text,
            model_id=model_id,
            base_url=base_url,
        )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
