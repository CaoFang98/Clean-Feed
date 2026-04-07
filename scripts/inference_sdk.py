#!/usr/bin/env python3
"""
CleanFeed LoRA 推理与验证工具。

支持：
1. 加载 base model + LoRA adapter 做单条推理
2. 对一份 JSONL 数据集进行批量验证
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Optional

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CleanFeed LoRA 推理与验证工具")
    parser.add_argument("--base_model_name", required=True, help="HF 模型名或本地 snapshot 路径")
    parser.add_argument(
        "--lora_model_path",
        default=None,
        help="LoRA adapter 目录，如 generation_model/.../final；不传时评估纯 base model",
    )
    parser.add_argument("--task_id", default="is_low_quality", help="任务 ID")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--text", default=None, help="直接验证一条文本")
    parser.add_argument("--data_path", type=Path, default=None, help="验证数据集路径（JSONL）")
    parser.add_argument(
        "--annotation_field",
        default="auto",
        choices=["auto", "task_annotations", "predicted_task_annotations"],
        help="金标字段；auto 会优先选择当前任务下有 true/false 标签的字段",
    )
    parser.add_argument(
        "--text_field",
        default="auto",
        choices=["auto", "text", "model_input", "content"],
        help="验证文本字段；auto 会按 text -> model_input -> content 选择",
    )
    parser.add_argument("--limit", type=int, default=20, help="-1 表示全部")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--show_samples", type=int, default=5)
    return parser.parse_args()


def resolve_runtime(device: str, dtype: str):
    import torch

    if device == "auto":
        if torch.backends.mps.is_available():
            resolved_device = "mps"
        elif torch.cuda.is_available():
            resolved_device = "cuda"
        else:
            resolved_device = "cpu"
    else:
        resolved_device = device

    if dtype == "auto":
        if resolved_device == "cuda":
            resolved_dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        elif resolved_device == "mps":
            resolved_dtype = "float16"
        else:
            resolved_dtype = "float32"
    else:
        resolved_dtype = dtype

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return resolved_device, resolved_dtype, dtype_map[resolved_dtype]


class CleanFeedDetector:
    def __init__(
        self,
        *,
        base_model_name: str,
        lora_model_path: str | None,
        task_id: str,
        device: str = "auto",
        dtype: str = "auto",
        local_files_only: bool = False,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.task_id = task_id
        self.lora_model_path = lora_model_path
        self.device, self.dtype_name, torch_dtype = resolve_runtime(device, dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, local_files_only=local_files_only)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            local_files_only=local_files_only,
        )
        if lora_model_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(
                self.base_model,
                lora_model_path,
                local_files_only=local_files_only,
            )
        else:
            self.model = self.base_model
        if self.device != "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()

    def build_prompt(self, text: str) -> str:
        return f"""请对以下内容执行任务 {self.task_id}，只输出 JSON：
{{"task_id":"{self.task_id}","label":true/false/null,"reason":"原因","evidence":"证据片段","confidence":0.xx}}

内容：
{text[:1200]}
输出："""

    def detect(self, text: str, max_new_tokens: int = 200) -> Dict:
        import time

        start_time = time.perf_counter()
        prompt = self.build_prompt(text)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True,
        ).strip()
        result = self._parse_response(response)
        result["raw_response"] = response
        result["process_time"] = round(time.perf_counter() - start_time, 4)
        return result

    def _parse_response(self, response: str) -> Dict:
        default_result = {
            "task_id": self.task_id,
            "label": None,
            "reason": "解析失败",
            "evidence": "",
            "confidence": 0.0,
        }
        try:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if not json_match:
                return default_result
            data = json.loads(json_match.group(0))
            label = data.get("label")
            if label not in [True, False, None]:
                label = None
            confidence = data.get("confidence", 0.0)
            try:
                confidence = float(confidence)
            except Exception:
                confidence = 0.0
            return {
                "task_id": data.get("task_id", self.task_id),
                "label": label,
                "reason": data.get("reason", ""),
                "evidence": data.get("evidence", ""),
                "confidence": confidence,
            }
        except Exception:
            return default_result


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def resolve_text_field(items: list[dict], text_field: str) -> str:
    if not items:
        raise ValueError("数据集为空")
    if text_field != "auto":
        return text_field
    for candidate in ["text", "model_input", "content"]:
        if candidate in items[0]:
            return candidate
    raise ValueError("数据中找不到可用的文本字段")


def count_labeled_rows(items: Iterable[dict], annotation_field: str, task_id: str) -> int:
    count = 0
    for item in items:
        label = (item.get(annotation_field) or {}).get(task_id, {}).get("label")
        if label in [True, False]:
            count += 1
    return count


def resolve_annotation_field(items: list[dict], annotation_field: str, task_id: str) -> str:
    if annotation_field != "auto":
        return annotation_field
    candidates = ["task_annotations", "predicted_task_annotations"]
    counts = {field: count_labeled_rows(items, field, task_id) for field in candidates}
    best = max(counts, key=counts.get)
    if counts[best] == 0:
        raise ValueError(f"任务 {task_id} 没有可用的 true/false 标签")
    return best


def evaluate_dataset(
    detector: CleanFeedDetector,
    *,
    data_path: Path,
    task_id: str,
    annotation_field: str,
    text_field: str,
    limit: int,
    max_new_tokens: int,
    show_samples: int,
):
    try:
        from tqdm import tqdm
    except ImportError as exc:
        raise RuntimeError("缺少依赖 tqdm，请先安装后再运行验证") from exc

    items = load_jsonl(data_path)
    resolved_annotation_field = resolve_annotation_field(items, annotation_field, task_id)
    resolved_text_field = resolve_text_field(items, text_field)

    labeled_items = []
    for item in items:
        gold = (item.get(resolved_annotation_field) or {}).get(task_id, {}).get("label")
        if gold in [True, False]:
            labeled_items.append(item)
    if limit != -1:
        labeled_items = labeled_items[:limit]

    if not labeled_items:
        raise ValueError("没有可评估的样本")

    tp = tn = fp = fn = unknown = 0
    samples = []
    total_time = 0.0

    progress = tqdm(labeled_items, desc=f"验证 {task_id}", unit="sample")
    for item in progress:
        text = item.get(resolved_text_field, "")
        gold = (item.get(resolved_annotation_field) or {}).get(task_id, {}).get("label")
        pred = detector.detect(text, max_new_tokens=max_new_tokens)
        pred_label = pred["label"]
        total_time += pred["process_time"]

        if pred_label is None:
            unknown += 1
        elif pred_label is True and gold is True:
            tp += 1
        elif pred_label is False and gold is False:
            tn += 1
        elif pred_label is True and gold is False:
            fp += 1
        elif pred_label is False and gold is True:
            fn += 1

        if len(samples) < show_samples:
            samples.append(
                {
                    "sample_id": item.get("sample_id", ""),
                    "gold": gold,
                    "pred": pred_label,
                    "confidence": pred["confidence"],
                    "reason": pred["reason"],
                    "text": text[:180],
                }
            )

        processed = tp + tn + fp + fn + unknown
        avg_time = total_time / processed if processed else 0.0
        progress.set_postfix(
            avg_time=f"{avg_time:.2f}s",
            unknown=unknown,
        )

    decided = tp + tn + fp + fn
    accuracy = (tp + tn) / decided if decided else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    coverage = decided / len(labeled_items)

    print("=" * 72)
    print("CleanFeed LoRA 验证结果")
    print("=" * 72)
    print(f"数据文件: {data_path}")
    print(f"任务: {task_id}")
    print(f"标签字段: {resolved_annotation_field}")
    print(f"文本字段: {resolved_text_field}")
    print(f"样本数: {len(labeled_items)}")
    print(f"device: {detector.device}, dtype: {detector.dtype_name}")
    print(f"accuracy: {accuracy:.4f}")
    print(f"precision: {precision:.4f}")
    print(f"recall: {recall:.4f}")
    print(f"f1: {f1:.4f}")
    print(f"coverage: {coverage:.4f}")
    print(f"unknown: {unknown}")
    print(f"avg_time_per_item: {total_time / len(labeled_items):.4f}s")

    if samples:
        print("\n样例:")
        for idx, sample in enumerate(samples, start=1):
            print(f"\n[{idx}] sample_id={sample['sample_id']}")
            print(f"gold={sample['gold']} pred={sample['pred']} confidence={sample['confidence']:.2f}")
            print(f"reason={sample['reason']}")
            print(f"text={sample['text']}")


def main():
    args = parse_args()
    detector = CleanFeedDetector(
        base_model_name=args.base_model_name,
        lora_model_path=args.lora_model_path,
        task_id=args.task_id,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )

    print(f"📌 base model: {args.base_model_name}")
    print(f"📌 lora path: {args.lora_model_path or '(none, base model only)'}")
    print(f"📌 device: {detector.device}")
    print(f"📌 dtype: {detector.dtype_name}")

    if args.data_path is not None:
        evaluate_dataset(
            detector,
            data_path=args.data_path,
            task_id=args.task_id,
            annotation_field=args.annotation_field,
            text_field=args.text_field,
            limit=args.limit,
            max_new_tokens=args.max_new_tokens,
            show_samples=args.show_samples,
        )
        return

    if args.text is None:
        raise SystemExit("请提供 --text 或 --data_path")

    print(json.dumps(detector.detect(args.text, max_new_tokens=args.max_new_tokens), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
