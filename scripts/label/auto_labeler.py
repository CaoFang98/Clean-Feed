#!/usr/bin/env python3
"""
统一任务化自动打标脚本。

支持：
1. 调用本地 Ollama 或远程大模型 API
2. 指定任意输入 JSONL 与文本字段
3. 输出为：
   - review: 供人工复核的 predicted_task_annotations
   - train: 直接可训练的 task_annotations
   - hybrid: 同时保留两者
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from task_config import DEFAULT_TASK_ORDER, TASK_DEFINITIONS, build_task_annotation

DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "zhihu_cleaned_data.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "task_prelabeled_dataset.jsonl"

LOCAL_PROVIDER = "local"
LOCAL_API_URL = "http://localhost:11434/api/chat"
LOCAL_MODEL = "qwen3:8b"

REMOTE_API_BASE_URL = {
    "doubao": "https://ark.cn-beijing.volces.com/api/coding/v3",
    "qwen": "https://dashscope.aliyuncs.com/api/v1/chat/completions",
    "openai": "https://api.openai.com/v1/chat/completions",
}
REMOTE_DEFAULT_MODEL = {
    "doubao": "ark-code-latest",
    "qwen": "qwen-turbo",
    "openai": "gpt-3.5-turbo",
}
REMOTE_COST_PER_1K_TOKENS = {
    "doubao": 0.0008,
    "qwen": 0.001,
    "openai": 0.01,
}

TOKENIZER = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CleanFeed 统一任务化自动打标工具")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="可选，未传时默认按 provider-model 自动生成带后缀的输出文件名",
    )
    parser.add_argument("--text-field", type=str, default="model_input")
    parser.add_argument("--provider", choices=[LOCAL_PROVIDER, "doubao", "qwen", "openai"], default=LOCAL_PROVIDER)
    parser.add_argument("--model", type=str, default=None, help="可选，覆盖默认模型名")
    parser.add_argument("--limit", type=int, default=500, help="-1 表示处理全部")
    parser.add_argument("--request-delay", type=float, default=0.1)
    parser.add_argument("--review-threshold", type=float, default=0.8)
    parser.add_argument("--output-format", choices=["review", "train", "hybrid"], default="hybrid")
    return parser.parse_args()


def count_tokens(text: str) -> int:
    global TOKENIZER
    if TOKENIZER is None:
        try:
            import tiktoken
        except ImportError as exc:
            raise RuntimeError("缺少依赖 tiktoken，请先安装后再运行自动打标脚本") from exc

        TOKENIZER = tiktoken.get_encoding("cl100k_base")
    return len(TOKENIZER.encode(text))


def get_model_name(provider: str, model_override: str | None) -> str:
    if model_override:
        return model_override
    if provider == LOCAL_PROVIDER:
        return LOCAL_MODEL
    return REMOTE_DEFAULT_MODEL[provider]


def get_api_key(provider: str) -> str:
    env_key = f"{provider.upper()}_API_KEY"
    api_key = os.getenv(env_key)
    if not api_key:
        raise ValueError(f"请先设置环境变量 {env_key}=你的API密钥")
    return api_key


def get_request_url(provider: str) -> str:
    url = REMOTE_API_BASE_URL[provider].rstrip("/")
    if provider != "doubao":
        return url

    normalized = url
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"


def sanitize_path_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    sanitized = re.sub(r"-{2,}", "-", sanitized).strip("-")
    return sanitized or "unknown"


def build_annotator_suffix(provider: str, model_name: str) -> str:
    return f"{sanitize_path_component(provider)}-{sanitize_path_component(model_name)}"


def resolve_output_path(output_path: Path | None, provider: str, model_name: str) -> Path:
    if output_path is not None:
        return output_path

    suffix = build_annotator_suffix(provider, model_name)
    return DEFAULT_OUTPUT_PATH.with_name(f"{DEFAULT_OUTPUT_PATH.stem}.{suffix}{DEFAULT_OUTPUT_PATH.suffix}")


def build_prompt(text: str) -> str:
    task_lines = []
    schema_lines = []
    for task_id in DEFAULT_TASK_ORDER:
        task_def = TASK_DEFINITIONS[task_id]
        reasons = "、".join(task_def["reason_options"])
        task_lines.append(
            f"- {task_id}: {task_def['description']}；reason 只能从这些选项里选：{reasons}"
        )
        schema_lines.append(
            f'    "{task_id}": {{"label": true/false/null, "reason": "原因或空字符串", "confidence": 0.xx}}'
        )

    task_desc = "\n".join(task_lines)
    task_schema = ",\n".join(schema_lines)
    return f"""
你是内容审核助手，请对下面输入按多个独立任务分别判断。

任务定义：
{task_desc}

输出要求：
1. 只输出 JSON，不要附加解释。
2. JSON 格式必须是：
{{
  "task_results": {{
{task_schema}
  }}
}}
3. 如果无法判断，可返回 label=null。

待判断内容：
{text[:1200]}
"""


def parse_task_results(content: str, annotator_name: str) -> dict:
    json_match = re.search(r"\{.*\}", content, re.DOTALL)
    if not json_match:
        raise ValueError("返回格式错误")

    data = json.loads(json_match.group(0))
    raw_results = data.get("task_results", {})
    parsed = {}

    for task_id in DEFAULT_TASK_ORDER:
        raw = raw_results.get(task_id, {})
        label = raw.get("label")
        if label not in [True, False, None]:
            label = None

        confidence = raw.get("confidence", 0.0)
        try:
            confidence = round(float(confidence), 2)
        except Exception:
            confidence = 0.0

        parsed[task_id] = build_task_annotation(
            label,
            reason=raw.get("reason", ""),
            annotator=annotator_name,
            source="auto",
            confidence=confidence,
        )
    return parsed


def build_failed_result(reason: str) -> dict:
    return {
        task_id: build_task_annotation(
            None,
            reason=reason,
            annotator="system",
            source="auto",
            confidence=0.0,
        )
        for task_id in DEFAULT_TASK_ORDER
    }


def request_local_model(prompt: str, model_name: str) -> tuple[str, int, int]:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("缺少依赖 requests，请先安装后再调用本地模型") from exc

    response = requests.post(
        LOCAL_API_URL,
        json={
            "model": model_name,
            "temperature": 0.1,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )
    if response.status_code != 200:
        raise ValueError(f"模型请求失败: {response.status_code}")

    content = response.json()["message"]["content"].strip()
    return content, count_tokens(prompt), count_tokens(content)


def request_remote_model(prompt: str, provider: str, model_name: str) -> tuple[str, int, int]:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("缺少依赖 requests，请先安装后再调用远程模型 API") from exc

    request_url = get_request_url(provider)
    response = requests.post(
        request_url,
        headers={
            "Authorization": f"Bearer {get_api_key(provider)}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_name,
            "temperature": 0.1,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )
    if response.status_code != 200:
        detail = response.text.strip().replace("\n", " ")
        detail = detail[:240] if detail else "empty response body"
        if provider == "doubao" and response.status_code == 404:
            raise ValueError(
                "API请求失败: 404。请确认豆包接口地址和模型 ID 是否匹配；"
                f"当前 URL={request_url}，当前 model={model_name}。"
                "当前脚本会自动在 Coding Plan 地址后补上 /chat/completions。若仍然 404，通常是模型未开通、无权限，"
                "或当前账号不支持用该 model 标识访问。"
            )
        raise ValueError(f"API请求失败: {response.status_code}; {detail}")

    result = response.json()
    if provider in {"doubao", "openai"}:
        content = result["choices"][0]["message"]["content"].strip()
        usage = result.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", count_tokens(prompt))
        completion_tokens = usage.get("completion_tokens", count_tokens(content))
        return content, prompt_tokens, completion_tokens

    content = result["output"]["choices"][0]["message"]["content"].strip()
    usage = result.get("usage", {})
    prompt_tokens = usage.get("input_tokens", count_tokens(prompt))
    completion_tokens = usage.get("output_tokens", count_tokens(content))
    return content, prompt_tokens, completion_tokens


def auto_label_text(text: str, provider: str, model_name: str) -> tuple[dict, int, int]:
    prompt = build_prompt(text)

    if provider == LOCAL_PROVIDER:
        content, prompt_tokens, completion_tokens = request_local_model(prompt, model_name)
    else:
        content, prompt_tokens, completion_tokens = request_remote_model(prompt, provider, model_name)

    annotator_name = model_name if provider == LOCAL_PROVIDER else provider
    return parse_task_results(content, annotator_name), prompt_tokens, completion_tokens


def resolve_text(item: dict, text_field: str) -> str:
    if text_field in item:
        return item[text_field]
    if text_field == "model_input":
        return item.get("model_input", item.get("content", ""))
    raise KeyError(f"字段不存在: {text_field}")


def should_review(predicted: dict, review_threshold: float) -> bool:
    for annotation in predicted.values():
        confidence = annotation.get("confidence") or 0.0
        if annotation.get("label") is None or confidence < review_threshold:
            return True
    return False


def build_output_item(
    item: dict,
    predicted: dict,
    *,
    provider: str,
    model_name: str,
    output_format: str,
    review_threshold: float,
) -> dict:
    output_item = dict(item)
    review_flag = should_review(predicted, review_threshold)

    if output_format in {"review", "hybrid"}:
        output_item["predicted_task_annotations"] = copy.deepcopy(predicted)
    if output_format in {"train", "hybrid"}:
        output_item["task_annotations"] = copy.deepcopy(predicted)

    output_item["need_review"] = review_flag
    output_item["auto_label_meta"] = {
        "provider": provider,
        "model": model_name,
        "output_format": output_format,
        "review_threshold": review_threshold,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    return output_item


def label_dataset(
    *,
    input_path: Path,
    output_path: Path,
    text_field: str,
    provider: str,
    model_name: str,
    limit: int,
    request_delay: float,
    review_threshold: float,
    output_format: str,
) -> dict:
    try:
        from tqdm import tqdm
    except ImportError as exc:
        raise RuntimeError("缺少依赖 tqdm，请先安装后再运行自动打标脚本") from exc

    with open(input_path, "r", encoding="utf-8") as f:
        all_data = [json.loads(line.strip()) for line in f if line.strip()]

    process_count = len(all_data) if limit == -1 or limit > len(all_data) else limit
    selected_items = all_data[:process_count]

    print("=" * 88)
    print("CleanFeed 统一任务化自动打标工具")
    print("=" * 88)
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"文本字段: {text_field}")
    print(f"调用方式: {provider}")
    print(f"模型: {model_name}")
    print(f"输出格式: {output_format}")
    print(f"处理条数: {process_count}/{len(all_data)}")

    results = []
    stats = {task_id: {"positive": 0, "negative": 0, "unknown": 0} for task_id in DEFAULT_TASK_ORDER}
    total_prompt_tokens = 0
    total_completion_tokens = 0
    failure_count = 0

    for item in tqdm(selected_items, desc="自动打标进度"):
        try:
            text = resolve_text(item, text_field)
            predicted, prompt_tokens, completion_tokens = auto_label_text(text, provider, model_name)
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
        except Exception as exc:
            failure_count += 1
            predicted = build_failed_result(f"预标失败: {str(exc)[:30]}")

        for task_id, annotation in predicted.items():
            if annotation["label"] is True:
                stats[task_id]["positive"] += 1
            elif annotation["label"] is False:
                stats[task_id]["negative"] += 1
            else:
                stats[task_id]["unknown"] += 1

        results.append(
            build_output_item(
                item,
                predicted,
                provider=provider,
                model_name=model_name,
                output_format=output_format,
                review_threshold=review_threshold,
            )
        )

        if request_delay > 0:
            time.sleep(request_delay)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    need_review_count = sum(1 for item in results if item["need_review"])

    print("\n✅ 自动打标完成")
    print(f"失败条数: {failure_count}")
    print(f"需要人工复核: {need_review_count}/{len(results)}")

    for task_id in DEFAULT_TASK_ORDER:
        task_def = TASK_DEFINITIONS[task_id]
        task_stats = stats[task_id]
        print(f"\n任务 {task_id} ({task_def['name']}):")
        print(f"  positive: {task_stats['positive']}")
        print(f"  negative: {task_stats['negative']}")
        print(f"  unknown: {task_stats['unknown']}")

    if provider != LOCAL_PROVIDER:
        total_tokens = total_prompt_tokens + total_completion_tokens
        total_cost = round(total_tokens / 1000 * REMOTE_COST_PER_1K_TOKENS[provider], 4)
        print(f"\n总 token: {total_tokens}")
        print(f"估算成本: {total_cost} 元")

    print(f"\n结果已保存到: {output_path}")

    return {
        "rows": len(results),
        "failures": failure_count,
        "need_review": need_review_count,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "stats": stats,
    }


def main():
    args = parse_args()
    model_name = get_model_name(args.provider, args.model)
    output_path = resolve_output_path(args.output_path, args.provider, model_name)
    label_dataset(
        input_path=args.input_path,
        output_path=output_path,
        text_field=args.text_field,
        provider=args.provider,
        model_name=model_name,
        limit=args.limit,
        request_delay=args.request_delay,
        review_threshold=args.review_threshold,
        output_format=args.output_format,
    )


if __name__ == "__main__":
    main()
